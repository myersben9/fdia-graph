"""Federated per-bus localizers: the paper's BusMLP and BusCNN trained by FedAvg over K clients
[MCM17], [FED26], with records never leaving a client.

Each client (a utility) holds every training record but reads only its own buses: its features are
built from its own meters (`kcl="local"`, the default, recomputes the power-balance channel from the
flow meters the client owns, the from-bus end of each branch), its loss covers only its own buses,
and a CNN convolves over its own buses in bus order (plus an optional read-only halo of other
clients' buses after them). What crosses a client boundary: the per-channel moments once (for one
standardization), the model weights every round (one average), and the per-bus confusion counts
once (for the paper's validation threshold). All three are the formulas of `formulas.federated` and
`formulas.metrics`.

With K = 1 and no gradient clip the fit is the centralized one, weight for weight.

    loc = FedBusCNN(K=3, rounds=60, local_epochs=3).fit(train, val=val)   # needs the [federated] extra
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..formulas.federated import channel_moments
from ..localization.learned import BusCNN, BusMLP, LearnedLocalizer, LocalTrainer, predict, standardization
from ..models.federated import Partition, RoundLog
from .aggregate import fedavg_state, state_bytes
from .partition import compute_nodes, spectral_partition

if TYPE_CHECKING:
    from ..dataset import FdiaGraph


@dataclass
class _Client:
    """One client's view: its compute buses (own first, then halo), how many are its own, its
    staged standardized features and labels, and its local copy of the model and trainer."""

    nodes: np.ndarray
    owned: int
    net: Any
    trainer: LocalTrainer
    Xt: Any = None
    Yt: Any = None
    rng: Any = None  # the client's own torch RNG state (dropout), carried from run to run


def _check_settings(K: int, rounds: int, local_epochs: int, halo: int, grad_clip: Optional[float]) -> None:
    """The federated knobs a fit cannot recover from."""
    if K < 1 or rounds < 1 or local_epochs < 1 or halo < 0:
        raise ValueError(
            f"need K, rounds, local_epochs >= 1 and halo >= 0, got {K}, {rounds}, {local_epochs}, {halo}"
        )
    if grad_clip is not None and not (np.isfinite(grad_clip) and grad_clip > 0):
        raise ValueError(f"grad_clip must be None or a finite positive norm, got {grad_clip}")


class FederatedLocalizer(LearnedLocalizer):
    """FedAvg over a K-way partition of the buses; the encoder comes from the class it is mixed
    with (`FedBusMLP`, `FedBusCNN`).

    K, rounds, local_epochs : the paper's defaults are K = 2, 60 rounds of 3 local epochs
    partition               : a Partition to use instead of `spectral_partition(K)`
    halo                    : hops of other clients' buses each client reads (0, the paper's runs)
    grad_clip               : gradient-norm bound in local training (1.0, the paper's harness)
    kcl                     : "local" (each client's own flow meters) or "global" (the paper's
                              central computation, which reads the neighbour's tie-line meters)
    Every other keyword is the encoder's (hidden, layers, dropout, lr, weight_decay, batch_size,
    pos_weight, seed, attackable_only, device, features, kernel).
    """

    def __init__(
        self,
        K: int = 2,
        rounds: int = 60,
        local_epochs: int = 3,
        partition: Optional[Partition] = None,
        halo: int = 0,
        grad_clip: Optional[float] = 1.0,
        kcl: str = "local",
        **kw: Any,
    ) -> None:
        if "epochs" in kw:
            raise ValueError("a federated fit trains rounds x local_epochs; pass those, not epochs")
        super().__init__(**kw)
        _check_settings(K, rounds, local_epochs, halo, grad_clip)
        if partition is not None and partition.K != K:
            raise ValueError(f"the partition has {partition.K} clients but K={K}")
        if kcl not in ("local", "global"):
            raise ValueError(f"kcl must be 'local' or 'global', got {kcl!r}")
        if "jac" in self.features:
            raise ValueError("the Jacobian features need the whole system's estimator, which no client has")
        self.K, self.rounds, self.local_epochs = K, rounds, local_epochs
        self.partition, self.halo, self.grad_clip, self.kcl = partition, halo, grad_clip, kcl
        self.epochs = rounds * local_epochs  # the local passes over the data each client makes
        self.history: list[RoundLog] = []

    # ---- features per client --------------------------------------------------------------
    def _client_features(self, d: dict[str, np.ndarray], k: int) -> np.ndarray:
        """Client k's raw feature block [n, N, F]: with kcl="local" the power balance counts only
        the flows metered at buses the client owns (the from-bus end of each branch)."""
        if self.kcl == "global" or self.features == "meas":
            return self._features(d)
        own_edge = self._part.assignment[d["edge_index"][0]] == k
        local = dict(d)
        local["edge_x"] = d["edge_x"] * own_edge[None, :, None]
        return self._features(local)

    # ---- LocalizerBase hooks --------------------------------------------------------------
    def _fit_stats(self, d: dict[str, np.ndarray], ben: np.ndarray, ds: FdiaGraph) -> None:
        torch = self._torch_seeded()
        ei = ds.edge_index_np
        self._part = self.partition or spectral_partition(ei, int(ds.N), self.K)
        if len(self._part.assignment) != int(ds.N):
            raise ValueError(
                f"the partition covers {len(self._part.assignment)} buses, the dataset has {ds.N}"
            )
        views = [compute_nodes(self._part, ei, k, self.halo) for k in range(self.K)]
        blocks, moments = [], []
        for k, (nodes, owned) in enumerate(views):  # one client's grid-wide block at a time
            X = self._client_features(d, k)
            moments.append(channel_moments(X[:, nodes[:owned]]))
            blocks.append(X[:, nodes])  # keep only the buses the client computes on
            del X
        # one standardization for everyone: each client's moments over its own buses, pooled
        self.mu, self.sd = standardization(moments)
        Y = d["y"].astype(np.float32)
        self.N = Y.shape[1]
        self._attackable = d["y"].any(axis=0)  # each client knows its own buses' labels
        self.dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._clients = self._make_clients(blocks, views, Y)
        self.net = self._run_rounds()
        self.net.eval()

    def _torch_seeded(self) -> Any:
        import torch

        torch.manual_seed(self.seed)
        return torch

    def _make_clients(
        self, blocks: list[np.ndarray], views: list[tuple[np.ndarray, int]], Y: np.ndarray
    ) -> list[_Client]:
        """One client per partition part, all starting from one initialization, each with its own
        dropout RNG: client 0 continues the stream the initialization drew from (so K = 1 is the
        centralized fit), client k > 0 starts from seed + k."""
        init = self._build(self.N).to(self.dev)
        clients = []
        for k, (X, (nodes, owned)) in enumerate(zip(blocks, views)):
            net = init if k == 0 else copy.deepcopy(init)
            c = _Client(
                nodes, owned, net, LocalTrainer(net, self._optim(), self.dev, self.seed + k, self.grad_clip)
            )
            c.rng = self._rng_state() if k == 0 else self._seeded_rng_state(self.seed + k)
            Xs = ((X - self.mu) / self.sd).astype(np.float32)
            c.Xt, c.Yt = c.trainer.stage(Xs, np.ascontiguousarray(Y[:, nodes]))
            clients.append(c)
        return clients

    # ---- per-client RNG streams -----------------------------------------------------------
    def _cuda(self) -> bool:
        return str(self.dev).startswith("cuda")

    def _rng_state(self) -> tuple[Any, Any]:
        """The process RNG state now (CPU, and the device's when training on a GPU)."""
        import torch

        return torch.get_rng_state(), (torch.cuda.get_rng_state(self.dev) if self._cuda() else None)

    def _seeded_rng_state(self, seed: int) -> tuple[Any, Any]:
        """The RNG state a fresh `manual_seed(seed)` gives, without disturbing the current one."""
        import torch

        with torch.random.fork_rng(devices=[torch.device(self.dev)] if self._cuda() else []):
            torch.manual_seed(seed)
            return self._rng_state()

    def _train_client(self, c: _Client) -> float:
        """One client's local epochs on its own RNG stream, which it keeps for the next round; the
        process stream is left as it was."""
        import torch

        with torch.random.fork_rng(devices=[torch.device(self.dev)] if self._cuda() else []):
            torch.set_rng_state(c.rng[0])
            if c.rng[1] is not None:
                torch.cuda.set_rng_state(c.rng[1], self.dev)
            loss = c.trainer.run(c.Xt, c.Yt, self.local_epochs, owned=c.owned)
            c.rng = self._rng_state()
        return loss

    def _run_rounds(self) -> Any:
        """The FedAvg loop: every client trains its local copy from the global weights on its own
        buses, the server averages (uniformly: every client holds every record)."""
        glob = {k: v.detach().clone() for k, v in self._clients[0].net.state_dict().items()}
        size = state_bytes(glob)
        self.history = []
        for r in range(self.rounds):
            states, losses = [], []
            for c in self._clients:
                c.net.load_state_dict(glob)
                losses.append(self._train_client(c))
                states.append(c.net.state_dict())
            glob = fedavg_state(states, [1.0] * len(self._clients))
            n = len(self._clients)
            self.history.append(RoundLog(r, float(np.mean(losses)), losses, n * size, n * size))
        net = self._clients[0].net
        net.load_state_dict(glob)
        return net

    def _score(self, d: dict[str, np.ndarray], ds: FdiaGraph) -> np.ndarray:
        """Each client scores its own buses with the global model on its own features; the columns
        are stitched into one [n, N] matrix."""
        out = np.zeros((len(d["y"]), self.N), np.float64)
        for k, c in enumerate(self._clients):
            Xs = ((self._client_features(d, k)[:, c.nodes] - self.mu) / self.sd).astype(np.float32)
            out[:, c.nodes[: c.owned]] = predict(self.net, Xs, self.dev)[:, : c.owned]
        if self.attackable_only:
            out[:, ~self._attackable] = 0.0
        return out


class FedBusMLP(FederatedLocalizer, BusMLP):
    """The paper's per-bus MLP trained by FedAvg (reads each bus alone, so the partition only
    decides whose data trains it)."""


class FedBusCNN(FederatedLocalizer, BusCNN):
    """The paper's 1-D CNN trained by FedAvg, each client convolving over its own buses."""
