"""Learned per-bus localizers: the graph-free encoders that gave the best localization numbers in
the federated localization paper, on that paper's 14-dim per-bus feature vector.

Both classes are LocalizerBase methods, so they drop into the same calibration and metrics as the
threshold arms. Two things differ from the threshold arms and follow the paper instead:

- fit() TRAINS on every record in the dataset it is given, attacked ones included, so the protocol
  is set by what you load: ``fg.load(sys, split="train", families=[0, 1, 2])`` is the papers'
  zero-shot protocol (benign + Aq + Ad in train, As/Ar unseen until test).
- fit(train, val) with a validation split picks ONE global probability threshold that maximizes
  mean per-bus F1 on val, the papers' rule. Without val, the base class's benign false-alarm
  calibration applies, which keeps the learned arms comparable to the threshold arms.

Needs torch: pip install "fdia-graph[torch]". Trains on the GPU when one is visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..choices import Choice
from ..formulas.federated import Moments, channel_moments, pool_moments
from ..formulas.metrics import perbus_counts, tau_from_counts
from ..models.scores import GridScores
from ..models.training import OptimConfig  # noqa: F401  re-exported beside its user
from .base import LocalizerBase

if TYPE_CHECKING:
    from ..dataset import FdiaGraph

N_FEAT = 14  # the papers' per-bus vector: 4 readings + 4 mask + 2 KCL + 2 delta + 2 swing


# Feature sets, named after the Jacobian-informed digest's ablation: A measurements only, B the
# papers' 14-dim vector (measurements + temporal change), C = B + the Jacobian block, D = the
# Jacobian block alone. The Jacobian block is fdia_graph.se.jacobian's 8 per-bus features.
# "+prev" appends the previous frame's swing (2 channels, a timeline's prev_swing): on a timeline the
# frame after a one-frame attack carries the same jump back with the opposite sign, and only the
# frame before tells the attack's jump from that return.
class Features(Choice):
    """The per-bus vector a learned localizer reads (channels in FEATURE_SETS)."""

    MEAS = "meas"
    FULL14 = "full14"
    FULL14_PREV = "full14+prev"
    FULL14_JAC = "full14+jac"
    FULL14_PREV_JAC = "full14+prev+jac"
    JAC = "jac"


# The channels per bus of each feature set.
FEATURE_SETS = {
    Features.MEAS: 8,
    Features.FULL14: 14,
    Features.FULL14_PREV: 16,
    Features.FULL14_JAC: 22,
    Features.FULL14_PREV_JAC: 24,
    Features.JAC: 8,
}


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError("learned localizers need torch: pip install 'fdia-graph[torch]'") from e


def kcl_residual(node_x: np.ndarray, edge_x: np.ndarray, ei: np.ndarray) -> np.ndarray:
    """Per-bus partial power balance [n, N, 2]: metered incident inflow minus the bus injection.

    Metering is sparse, so this is a true Kirchhoff balance only at buses whose incident branches
    are all metered; elsewhere it is a partial residual. The papers keep it raw rather than masking
    it (the meter-mask channels let the model discount the partial buses), which scored higher.
    node_x column order is [|V|, P_inj, Q_inj, theta]; edge_x is [P_from, Q_from].
    """
    n, N = node_x.shape[:2]
    inflow = np.zeros((N, n, 2), np.float64)  # bus-first so the scatter indexes one axis
    flows = edge_x.transpose(1, 0, 2)  # [E, n, 2]
    np.add.at(inflow, ei[1], flows)  # arrives at the to-bus
    np.add.at(inflow, ei[0], -flows)  # leaves the from-bus
    return inflow.transpose(1, 0, 2) - node_x[:, :, 1:3]


def full14(d: dict[str, np.ndarray]) -> np.ndarray:
    """The papers' [n, N, 14] per-bus feature vector, in the order every trained model expects."""
    nx = d["node_x"].astype(np.float64)
    kcl = kcl_residual(nx, d["edge_x"].astype(np.float64), d["edge_index"])
    return np.concatenate([nx, d["node_m"].astype(np.float64), kcl, d["temporal_delta"], d["swing"]], axis=-1)


class LearnedLocalizer(LocalizerBase):
    """Shared training and scoring for the learned arms; subclasses supply the encoder.

    Knobs default to the paper's best run: 4 layers of width 128, dropout 0.1, AdamW at lr 5e-4
    and weight decay 0.01, batch 256, plain BCE (pos_weight 1.0), seed 123. The paper trained
    60 federated rounds of 3 local epochs on half-size client shards, roughly 90 full-data epochs;
    ``epochs=60`` is a practical default and ``epochs=90`` matches that budget.

    attackable_only (default True, the paper's rule) never flags a bus that carries no attack
    label in the training records, so buses that can never be attacked contribute no false alarms.
    """

    def __init__(
        self,
        fa_target: float = 0.01,
        hidden: int = 128,
        layers: int = 4,
        dropout: float = 0.1,
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        batch_size: int = 256,
        epochs: int = 60,
        pos_weight: float = 1.0,
        seed: int = 123,
        attackable_only: bool = True,
        device: Optional[str] = None,
        features: str = "full14",
    ) -> None:
        super().__init__(fa_target=fa_target)
        if layers < 1 or hidden < 8:
            raise ValueError(f"need layers >= 1 and hidden >= 8, got {layers}, {hidden}")
        chosen = Features(features)
        self.features = chosen.value  # which per-bus vector the encoder sees (see FEATURE_SETS)
        self.n_feat = FEATURE_SETS[chosen]
        self.hidden, self.layers, self.dropout = hidden, layers, dropout
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.epochs, self.pos_weight, self.seed = epochs, pos_weight, seed
        self.attackable_only = attackable_only
        self.device = device  # None -> cuda if available, else cpu
        self.tau: Optional[float] = None  # set by tune_threshold (the papers' global threshold)

    # ---- subclass hook --------------------------------------------------------------------
    def _build(self, N: int) -> Any:
        """Return an nn.Module mapping [B, N, 14] standardized features to [B, N] logits."""
        raise NotImplementedError

    # ---- LocalizerBase hooks --------------------------------------------------------------
    def _fields(self) -> list[str]:
        # Only what the chosen feature set reads: measurement-only and Jacobian-only models run on
        # datasets without the temporal fields; the Jacobian block needs the previous frame.
        f = ["node_x", "node_m", "edge_x", "y"]
        if "full14" in self.features:
            f += ["temporal_delta", "swing"]
        if "prev" in self.features:
            f += ["prev_swing"]
        if "jac" in self.features:
            f += ["prev_node_x", "prev_edge_x", "prev_timestep"]
        return f

    def _features(self, d: dict[str, np.ndarray], jac: Optional[np.ndarray] = None) -> np.ndarray:
        """The per-bus vector for the chosen feature set, [n, N, n_feat], raw (standardized later):
        each bus's own channels, then the Jacobian block when the set has one (`jac` when handed
        in, else built from d)."""
        block: Optional[np.ndarray] = None
        if "jac" in self.features:
            block = self._jac.transform(d)["bus"] if jac is None else jac  # [n, N, 8], fdia_graph.se.jacobian
            if self.features == "jac":
                return block
        own = self._own_channels(d)
        return own if block is None else np.concatenate([own, block], -1)

    def _own_channels(self, d: dict[str, np.ndarray]) -> np.ndarray:
        """The channels each bus builds from its own readings: the measurements and mask, or the
        papers' 14, followed by the previous frame's swing for a "+prev" set."""
        if self.features == "meas":
            return np.concatenate([d["node_x"].astype(np.float64), d["node_m"].astype(np.float64)], -1)
        own = full14(d)
        return (
            np.concatenate([own, d["prev_swing"].astype(np.float64)], -1) if "prev" in self.features else own
        )

    def _fit_stats(self, d: dict[str, np.ndarray], ben: np.ndarray, ds: FdiaGraph) -> None:
        torch = _torch()
        if "jac" in self.features:  # the Jacobian block needs the [se] physics, fitted on this split
            from ..se.jacobian import JacobianFeatures

            self._jac = JacobianFeatures().fit(ds)
        X = self._features(d)
        # Standardize every channel on the training records, mask and swing included, exactly as
        # the paper's cache builder does; sd is floored so a constant channel cannot blow up. The
        # moments pool across parts, which is how a federated fit shares them.
        self.mu, self.sd = standardization([channel_moments(X)])
        Xs = ((X - self.mu) / self.sd).astype(np.float32)
        Y = d["y"].astype(np.float32)
        self.N = Xs.shape[1]
        self._attackable = d["y"].any(axis=0)  # buses that carry an attack label in train

        torch.manual_seed(self.seed)
        self.dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self._build(self.N).to(self.dev)
        trainer = LocalTrainer(self.net, self._optim(), self.dev, self.seed)
        trainer.run(*trainer.stage(Xs, Y), self.epochs)
        self.net.eval()

    def _optim(self) -> OptimConfig:
        """The training knobs a LocalTrainer needs, from this localizer's settings."""
        return OptimConfig(self.lr, self.weight_decay, self.batch_size, self.pos_weight)

    def _score(self, d: dict[str, np.ndarray], ds: FdiaGraph) -> np.ndarray:
        if "jac" in self.features:  # the Jacobian block converts physical units itself
            from ..se.base import require_physical

            require_physical(ds)
        Xs = ((self._features(d) - self.mu) / self.sd).astype(np.float32)
        out = predict(self.net, Xs, self.dev)
        if self.attackable_only:
            out[:, ~self._attackable] = 0.0  # probability 0 can never cross a threshold in (0, 1]
        return out

    # ---- fitting ----------------------------------------------------------------------------
    def fit(self, ds: FdiaGraph, val: Optional[FdiaGraph] = None) -> LearnedLocalizer:
        """Train on every record in ds; calibrate thresholds on benign records (base protocol),
        or, when val is given, pick the papers' single validation-best probability threshold."""
        self.grid_tau: Optional[float] = None  # a grid threshold belongs to one fit
        super().fit(ds)
        if self.attackable_only:
            self.thr[~self._attackable] = np.inf
        if val is not None:
            self.tune_threshold(val)
        return self

    def tune_threshold(self, val: FdiaGraph) -> LearnedLocalizer:
        """The papers' rule: one global tau on a 0.05..0.95 grid maximizing mean per-bus F1 on val,
        the mean taken over buses that carry an attack label in val."""
        d = self._pull(val, extra=["y"])
        p, t = self._score(d, val), d["y"].astype(bool)
        active = t.any(axis=0)
        if not active.any():
            raise ValueError("tune_threshold needs attacked records in val")
        taus = np.linspace(0.05, 0.95, 19)
        tp, fp, fn = (np.stack(c) for c in zip(*(perbus_counts(p > tau, t) for tau in taus)))
        self.tau = tau_from_counts(tp, fp, fn, np.asarray(active), taus)
        self.thr = np.full(p.shape[1], self.tau)
        if self.attackable_only:
            self.thr[~self._attackable] = np.inf
        return self

    def _grid_score(self, d: dict[str, np.ndarray], ds: FdiaGraph) -> np.ndarray:
        """A record's grid score: its highest probability over the attackable buses."""
        p = self._score(d, ds)
        return p[:, self._attackable].max(axis=1) if self._attackable.any() else np.zeros(len(p))

    def tune_grid_threshold(self, val: FdiaGraph) -> LearnedLocalizer:
        """The federated paper's grid rule: one tau on the 0.05..0.95 grid maximizing record-level
        F1 (attacked vs benign) on val, applied to the grid score."""
        d = self._pull(val, extra=["y"])
        g, attacked = self._grid_score(d, val), np.asarray(d["y"].any(axis=1))
        if attacked.all() or not attacked.any():
            raise ValueError("tune_grid_threshold needs both attacked and benign records in val")
        taus = np.linspace(0.05, 0.95, 19)
        tp, fp, fn = (
            np.stack(c) for c in zip(*(perbus_counts((g > t)[:, None], attacked[:, None]) for t in taus))
        )
        self.grid_tau = tau_from_counts(tp, fp, fn, np.ones(1, bool), taus)
        return self

    def score_grid(self, ds: FdiaGraph) -> GridScores:
        """Record-level detection at `grid_tau` (from `tune_grid_threshold`): the benign false-alarm
        rate and the detection rate, over every attacked record and per family."""
        from ..dataset import FAMILIES

        tau = getattr(self, "grid_tau", None)
        if tau is None:
            raise ValueError("call tune_grid_threshold(val) first")
        d = self._pull(ds, extra=["family", "y"])
        flag, fam = self._grid_score(d, ds) > tau, d["family"]
        by = {n: float(flag[fam == f].mean()) for f, n in FAMILIES.items() if f and (fam == f).any()}
        return GridScores(
            tau=float(tau),
            false_alarm=float(flag[fam == 0].mean()) if (fam == 0).any() else 0.0,
            detection_rate=float(flag[fam != 0].mean()) if (fam != 0).any() else 0.0,
            by_family=by,
        )


def standardization(parts: list[Moments]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and sd from pooled moments (`formulas.federated.pool_moments`), the sd
    floored at 1e-3 so a constant channel cannot blow up."""
    _, mu, var = pool_moments(parts)
    return mu, np.clip(np.sqrt(var), 1e-3, None)


def predict(net: Any, Xs: np.ndarray, dev: str, chunk: int = 4096) -> np.ndarray:
    """Per-bus attack probabilities [n, N] of standardized features [n, N, F], in chunks so the
    big systems stay inside a bounded device footprint."""
    torch = _torch()
    net.eval()  # inference: dropout off even straight after a training run
    out = np.empty(Xs.shape[:2], np.float64)
    with torch.no_grad():
        for i in range(0, len(Xs), chunk):
            xb = torch.from_numpy(Xs[i : i + chunk]).to(dev)
            out[i : i + chunk] = torch.sigmoid(net(xb)).cpu().numpy()
    return out


class LocalTrainer:
    """The training loop of a learned localizer, kept apart so a federated client runs the same
    code: AdamW, BCE with logits, and a seeded batch order that persists across calls to `run`
    (the optimizer too), so several short runs equal one long one.

    `owned`, when given, restricts the loss to the first `owned` buses of every sample (a client's
    own buses, its halo after them); `clip` bounds the gradient norm. Neither is used centrally.
    """

    def __init__(self, net: Any, cfg: OptimConfig, dev: str, seed: int, clip: Optional[float] = None) -> None:
        if clip is not None and not (np.isfinite(clip) and clip > 0):
            raise ValueError(f"clip must be None or a finite positive norm, got {clip}")
        torch = _torch()
        self.net, self.cfg, self.dev, self.clip = net, cfg, dev, clip
        self.opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=dev))
        self.gen = torch.Generator().manual_seed(seed)

    def stage(self, Xs: np.ndarray, Y: np.ndarray) -> tuple[Any, Any]:
        """The split as CPU tensors, pinned when training on a GPU: only each batch crosses to the
        device, so the big systems train inside a bounded device footprint."""
        torch = _torch()
        Xt, Yt = torch.from_numpy(Xs), torch.from_numpy(Y)
        if self.dev != "cpu":
            Xt, Yt = Xt.pin_memory(), Yt.pin_memory()
        return Xt, Yt

    def run(self, Xt: Any, Yt: Any, epochs: int, owned: Optional[int] = None) -> float:
        """Train `epochs` passes over the staged split; returns the mean batch loss of the last."""
        torch = _torch()
        self.net.train()
        losses: list[float] = []
        for _ in range(epochs):
            losses = []
            perm = torch.randperm(len(Xt), generator=self.gen)
            for i in range(0, len(Xt), self.cfg.batch_size):
                losses.append(self._step(Xt, Yt, perm[i : i + self.cfg.batch_size], owned))
        return float(np.mean(losses)) if losses else 0.0

    def _step(self, Xt: Any, Yt: Any, j: Any, owned: Optional[int]) -> float:
        """One optimizer step on the batch rows `j`."""
        torch = _torch()
        xb = Xt[j].to(self.dev, non_blocking=True)
        yb = Yt[j].to(self.dev, non_blocking=True)
        self.opt.zero_grad()
        logits = self.net(xb)
        if owned is not None:  # a client learns only its own buses
            logits, yb = logits[:, :owned], yb[:, :owned]
        loss = self.loss_fn(logits, yb)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
        self.opt.step()
        return float(loss.detach())


class BusCNN(LearnedLocalizer):
    """The paper's best localizer: a 1-D convolution across the bus axis (kernel 3, padding same).

    Each layer mixes a bus's 14-dim vector with its index-neighbors in bus order, without reading
    the graph. Four layers of 128 channels, GroupNorm after every second convolution, ReLU, dropout,
    then a linear head per bus. About 154k parameters; macro-F1 0.963 / 0.963 / 0.952 on
    IEEE 14 / 118 / 300 in the paper's zero-shot protocol (v0.4.1 data).
    """

    def __init__(self, kernel: int = 3, **kw: Any) -> None:
        super().__init__(**kw)
        self.kernel = kernel

    def _build(self, N: int) -> Any:
        return _cnn_net(self.n_feat, self.hidden, self.layers, self.kernel, self.dropout)


class BusMLP(LearnedLocalizer):
    """The paper's lightweight arm: an identical per-bus MLP applied to every bus on its own.

    Reads nothing but the bus's own 14 numbers, so it is the cleanest statement of the temporal
    feature's power. Four layers of 128 units, LayerNorm after every second layer, ReLU, dropout,
    linear head. About 52k parameters; macro-F1 0.963 / 0.957 / 0.933 on IEEE 14 / 118 / 300 in
    the paper's zero-shot protocol (v0.4.1 data).
    """

    def _build(self, N: int) -> Any:
        return _mlp_net(self.n_feat, self.hidden, self.layers, self.dropout)


def _cnn_net(n_feat: int, hidden: int, layers: int, kernel: int, dropout: float) -> Any:
    """The BusCNN network: 1-D convolutions across the bus axis (kernel `kernel`, padding same),
    GroupNorm after every second convolution, ReLU, dropout, a linear head per bus."""
    nn = _torch().nn

    class _CNN(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self, F: int, H: int, L: int, k: int, p: float) -> None:
            super().__init__()
            nn = _torch().nn
            self.convs = nn.ModuleList(
                [nn.Conv1d(F if i == 0 else H, H, k, padding="same") for i in range(L)]
            )
            self.norms = nn.ModuleList([nn.GroupNorm(max(H // 8, 1), H) for _ in range(L // 2)])
            self.drop = nn.Dropout(p)
            self.head = nn.Linear(H, 1)

        def forward(self, x: Any) -> Any:  # [B, N, F] -> [B, N]
            torch = _torch()
            x = x.permute(0, 2, 1)  # Conv1d wants [B, C, N]: the bus axis is the sequence
            for i, conv in enumerate(self.convs):
                x = conv(x)
                if (i + 1) % 2 == 0:
                    x = self.norms[i // 2](x)
                x = self.drop(torch.relu(x))
            return self.head(x.permute(0, 2, 1)).squeeze(-1)

    return _CNN(n_feat, hidden, layers, kernel, dropout)


def _mlp_net(n_feat: int, hidden: int, layers: int, dropout: float) -> Any:
    """The BusMLP network: the same per-bus MLP applied to every bus on its own, LayerNorm after
    every second layer, ReLU, dropout, a linear head."""
    nn = _torch().nn

    class _MLP(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self, F: int, H: int, L: int, p: float) -> None:
            super().__init__()
            nn = _torch().nn
            self.lins = nn.ModuleList([nn.Linear(F if i == 0 else H, H) for i in range(L)])
            self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(L // 2)])
            self.drop = nn.Dropout(p)
            self.head = nn.Linear(H, 1)

        def forward(self, x: Any) -> Any:  # [B, N, F] -> [B, N]
            torch = _torch()
            for i, lin in enumerate(self.lins):
                x = lin(x)
                if (i + 1) % 2 == 0:
                    x = self.norms[i // 2](x)
                x = self.drop(torch.relu(x))
            return self.head(x).squeeze(-1)

    return _MLP(n_feat, hidden, layers, dropout)
