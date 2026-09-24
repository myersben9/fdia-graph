"""Federated per-bus localizers: the paper's BusMLP and BusCNN trained by FedAvg over K clients
[MCM17], [FED26], with records never leaving a client.

Each client (a utility) holds every training record but reads only its own buses: its features are
built from its own meters (`kcl="local"`, the default, recomputes the power-balance channel from the
flow meters the client owns, the from-bus end of each branch; the other channels are each bus's own
readings and their scan-to-scan change), its loss covers only its own buses, and a CNN convolves
over its own buses in bus order. Two opt-in exceptions read beyond a client's own meters:
`halo > 0` adds the halo buses' own meters as read-only context, placed after its own buses, and
the Jacobian feature sets (`features="full14+jac"` or `"jac"`) append the 8-channel Jacobian block,
the whole system's estimator applied to every meter's change, computed once centrally per pass and
shared with every client. What crosses a client boundary: the per-channel moments once (for one
standardization), the model weights every round (one average), the per-bus confusion counts once
(for the paper's validation threshold), and with a Jacobian feature set the central block. All three are the formulas of `formulas.federated` and
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
from ..formulas.metrics import perbus_counts, tau_from_counts
from ..localization.learned import (
    BusCNN,
    BusMLP,
    LearnedLocalizer,
    LocalTrainer,
    full14,
    predict,
    standardization,
)
from ..models.federated import Partition, RoundLog
from .aggregate import fedavg_state, state_bytes
from .partition import check_partition, compute_nodes, spectral_partition

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
    attackable: Any = None  # [owned] bool: which of its own buses carry an attack label in its train records

    @property
    def own(self) -> np.ndarray:
        """The client's own buses (the first `owned` compute buses)."""
        return self.nodes[: self.owned]


def _is_int(v: Any) -> bool:
    """A true integer (numpy's included), not a bool or a float that happens to be whole."""
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _check_settings(K: int, rounds: int, local_epochs: int, halo: int, grad_clip: Optional[float]) -> None:
    """The federated knobs a fit cannot recover from."""
    counts = (K, rounds, local_epochs, halo)
    if not all(_is_int(v) for v in counts):
        raise ValueError(f"K, rounds, local_epochs and halo must be integers, got {counts}")
    if min(K, rounds, local_epochs) < 1 or halo < 0:
        raise ValueError(f"need K, rounds, local_epochs >= 1 and halo >= 0, got {counts}")
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
        self.K, self.rounds, self.local_epochs = K, rounds, local_epochs
        self.partition, self.halo, self.grad_clip, self.kcl = partition, halo, grad_clip, kcl
        self.epochs = rounds * local_epochs  # the local passes over the data each client makes
        self.history: list[RoundLog] = []

    # ---- features per client --------------------------------------------------------------
    def _client_features(
        self, d: dict[str, np.ndarray], k: int, jac: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Client k's raw feature block [n, N, F]: with kcl="local" the power balance counts only
        the flows metered at buses the client owns (the from-bus end of each branch). The Jacobian
        block of the "jac" feature sets is the one exception to client-local features: it is the
        whole system's estimator applied to every meter's change, computed once centrally per pass
        (`_central`) and handed in as `jac`; computed here when not given."""
        if self.features == "meas":
            return self._features(d)
        if self.features == "jac":  # no client-local channel to build
            return self._jac.transform(d)["bus"] if jac is None else jac
        local = d
        if self.kcl == "local":
            own_edge = self._part.assignment[d["edge_index"][0]] == k
            local = dict(d)
            local["edge_x"] = d["edge_x"] * own_edge[None, :, None]
        if "jac" not in self.features:
            return self._features(local)
        # a single client's call builds the central block here; a pass over the clients hands it in
        block: np.ndarray = self._jac.transform(d)["bus"] if jac is None else jac
        return np.concatenate([full14(local), block], -1)

    def _check_units(self, ds: FdiaGraph) -> None:
        """The Jacobian block converts physical units itself, so a per-unit view is refused."""
        if "jac" in self.features:
            from ..se.base import require_physical

            require_physical(ds)

    def _central(self, d: dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """The per-bus Jacobian block [n, N, 8] of these records, or None without a "jac" feature
        set. A pass over the clients computes it once and hands it to each; nothing is kept after."""
        return self._jac.transform(d)["bus"] if "jac" in self.features else None

    # ---- LocalizerBase hooks --------------------------------------------------------------
    def _fit_stats(self, d: dict[str, np.ndarray], ben: np.ndarray, ds: FdiaGraph) -> None:
        torch = self._torch_seeded()
        ei = ds.edge_index_np
        if "jac" in self.features:  # the central estimator's physics, fitted on this split
            from ..se.jacobian import JacobianFeatures

            self._jac = JacobianFeatures().fit(ds)
        self._part = self.partition or spectral_partition(ei, int(ds.N), self.K)
        check_partition(self._part, int(ds.N))
        views = [compute_nodes(self._part, ei, k, self.halo) for k in range(self.K)]
        blocks, moments, jac = [], [], self._central(d)
        for k, (nodes, owned) in enumerate(views):  # one client's grid-wide block at a time
            X = self._client_features(d, k, jac)
            moments.append(channel_moments(X[:, nodes[:owned]]))
            blocks.append(X[:, nodes])  # keep only the buses the client computes on
            del X
        # one standardization for everyone: each client's moments over its own buses, pooled
        self.mu, self.sd = standardization(moments)
        Y = d["y"].astype(np.float32)
        self.N = Y.shape[1]
        self.dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._clients = self._make_clients(blocks, views, Y)
        # each client finds its own attackable buses in its own labels; the per-bus thresholds that
        # LearnedLocalizer.fit sets from this mask are each bus's own, so it is assembled, not pooled
        self._attackable = np.zeros(self.N, bool)
        for c in self._clients:
            c.attackable = d["y"][:, c.own].any(axis=0)
            self._attackable[c.own] = c.attackable
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
            c.Xt, c.Yt = c.trainer.stage(Xs, np.ascontiguousarray(Y[:, nodes[:owned]]))  # own labels only
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
        for c in self._clients:  # the final broadcast: every client deploys the averaged model
            c.net.load_state_dict(glob)
        return self._clients[0].net

    def _client_scores(
        self, d: dict[str, np.ndarray], k: int, jac: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Client k's attack probabilities on its own buses [n, owned], from its own features and its
        own attackable mask."""
        c = self._clients[k]
        Xs = ((self._client_features(d, k, jac)[:, c.nodes] - self.mu) / self.sd).astype(np.float32)
        p = predict(self.net, Xs, self.dev)[:, : c.owned]
        if self.attackable_only:
            p[:, ~c.attackable] = 0.0
        return p

    def _score(self, d: dict[str, np.ndarray], ds: FdiaGraph) -> np.ndarray:
        """Each client scores its own buses; the columns are stitched into one [n, N] matrix."""
        self._check_units(ds)
        out, jac = np.zeros((len(d["y"]), self.N), np.float64), self._central(d)
        for k, c in enumerate(self._clients):
            out[:, c.own] = self._client_scores(d, k, jac)
        return out

    def tune_threshold(self, val: FdiaGraph) -> FederatedLocalizer:
        """The papers' validation tau with labels kept local: each client counts true positives,
        false positives and false negatives on its own buses at every candidate tau, and only those
        per-bus counts meet (`formulas.metrics.tau_from_counts`), the same tau as a pooled count."""
        self._check_units(val)
        d = self._pull(val, extra=["y"])
        taus = np.linspace(0.05, 0.95, 19)
        tp, fp, fn = (np.zeros((len(taus), self.N)) for _ in range(3))
        active, jac = np.zeros(self.N, bool), self._central(d)
        for k, c in enumerate(self._clients):
            p, t = self._client_scores(d, k, jac), d["y"][:, c.own].astype(bool)
            for i, tau in enumerate(taus):
                tp[i, c.own], fp[i, c.own], fn[i, c.own] = perbus_counts(p > tau, t)
            active[c.own] = t.any(axis=0)
        if not active.any():
            raise ValueError("tune_threshold needs attacked records in val")
        self.tau = tau_from_counts(tp, fp, fn, active, taus)
        self.thr = np.full(self.N, self.tau)
        if self.attackable_only:
            self.thr[~self._attackable] = np.inf
        return self


class FedBusMLP(FederatedLocalizer, BusMLP):
    """The paper's per-bus MLP trained by FedAvg (reads each bus alone, so the partition only
    decides whose data trains it)."""


class FedBusCNN(FederatedLocalizer, BusCNN):
    """The paper's 1-D CNN trained by FedAvg, each client convolving over its own buses."""
