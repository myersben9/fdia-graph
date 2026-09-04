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

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from .base import LocalizerBase, _perbus_f1

if TYPE_CHECKING:
    from ..dataset import FdiaGraph

N_FEAT = 14  # the papers' per-bus vector: 4 readings + 4 mask + 2 KCL + 2 delta + 2 swing
# Feature sets, named after the Jacobian-informed digest's ablation: A measurements only, B the
# papers' 14-dim vector (measurements + temporal change), C = B + the Jacobian block, D = the
# Jacobian block alone. The Jacobian block is fdia_graph.se.jacobian's 8 per-bus features.
FEATURE_SETS = {"meas": 8, "full14": 14, "full14+jac": 22, "jac": 8}


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


def full14(d: Dict[str, np.ndarray]) -> np.ndarray:
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
        if features not in FEATURE_SETS:
            raise ValueError(f"features must be one of {sorted(FEATURE_SETS)}, got {features!r}")
        self.features = features  # which per-bus vector the encoder sees (see FEATURE_SETS)
        self.n_feat = FEATURE_SETS[features]
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
    def _fields(self) -> List[str]:
        base = ["node_x", "node_m", "edge_x", "temporal_delta", "swing", "y"]
        return base + ["timestep"] if "jac" in self.features else base

    def _features(self, d: Dict[str, np.ndarray]) -> np.ndarray:
        """The per-bus vector for the chosen feature set, [n, N, n_feat], raw (standardized later)."""
        if self.features == "meas":
            return np.concatenate([d["node_x"].astype(np.float64), d["node_m"].astype(np.float64)], -1)
        if self.features == "full14":
            return full14(d)
        jac = self._jac.transform(d)["bus"]  # [n, N, 8] from fdia_graph.se.jacobian
        return jac if self.features == "jac" else np.concatenate([full14(d), jac], -1)

    def _fit_stats(self, d: Dict[str, np.ndarray], ben: np.ndarray, ds: "FdiaGraph") -> None:
        torch = _torch()
        if "jac" in self.features:  # the Jacobian block needs the [se] physics, fitted on this split
            from ..se.jacobian import JacobianFeatures

            self._jac = JacobianFeatures().fit(ds)
        X = self._features(d)
        # Standardize every channel on the training records, mask and swing included, exactly as
        # the paper's cache builder does; sd is floored so a constant channel cannot blow up.
        self.mu = X.mean(axis=(0, 1))
        self.sd = np.clip(X.std(axis=(0, 1)), 1e-3, None)
        Xs = ((X - self.mu) / self.sd).astype(np.float32)
        Y = d["y"].astype(np.float32)
        self.N = Xs.shape[1]
        self._attackable = d["y"].any(axis=0)  # buses that carry an attack label in train

        torch.manual_seed(self.seed)
        self.dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self._build(self.N).to(self.dev)
        # The full split stays on the CPU (pinned when a GPU is used) and only each batch crosses to
        # the device, so the big systems train inside a bounded device footprint like scoring does.
        Xt, Yt = torch.from_numpy(Xs), torch.from_numpy(Y)
        if self.dev != "cpu":
            Xt, Yt = Xt.pin_memory(), Yt.pin_memory()
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight, device=self.dev))
        gen = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt), generator=gen)
            for i in range(0, len(Xt), self.batch_size):
                j = perm[i : i + self.batch_size]
                xb = Xt[j].to(self.dev, non_blocking=True)
                yb = Yt[j].to(self.dev, non_blocking=True)
                opt.zero_grad()
                loss_fn(self.net(xb), yb).backward()
                opt.step()
        self.net.eval()

    def _score(self, d: Dict[str, np.ndarray]) -> np.ndarray:
        torch = _torch()
        Xs = ((self._features(d) - self.mu) / self.sd).astype(np.float32)
        out = np.empty(Xs.shape[:2], np.float64)
        with torch.no_grad():
            for i in range(0, len(Xs), 4096):  # bounded device memory on the big systems
                xb = torch.from_numpy(Xs[i : i + 4096]).to(self.dev)
                out[i : i + 4096] = torch.sigmoid(self.net(xb)).cpu().numpy()
        if self.attackable_only:
            out[:, ~self._attackable] = 0.0  # probability 0 can never cross a threshold in (0, 1]
        return out

    # ---- fitting ----------------------------------------------------------------------------
    def fit(self, ds: "FdiaGraph", val: Optional["FdiaGraph"] = None) -> "LearnedLocalizer":
        """Train on every record in ds; calibrate thresholds on benign records (base protocol),
        or, when val is given, pick the papers' single validation-best probability threshold."""
        super().fit(ds)
        if self.attackable_only:
            self.thr[~self._attackable] = np.inf
        if val is not None:
            self.tune_threshold(val)
        return self

    def tune_threshold(self, val: "FdiaGraph") -> "LearnedLocalizer":
        """The papers' rule: one global tau on a 0.05..0.95 grid maximizing mean per-bus F1 on val,
        the mean taken over buses that carry an attack label in val."""
        d = self._pull(val, extra=["y"])
        p, t = self._score(d), d["y"].astype(bool)
        active = t.any(axis=0)
        if not active.any():
            raise ValueError("tune_threshold needs attacked records in val")
        taus = np.linspace(0.05, 0.95, 19)
        f1 = [_perbus_f1(p > tau, t)[active].mean() for tau in taus]
        self.tau = float(taus[int(np.argmax(f1))])
        self.thr = np.full(p.shape[1], self.tau)
        if self.attackable_only:
            self.thr[~self._attackable] = np.inf
        return self


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
        torch = _torch()
        nn = torch.nn
        H, L, k, p, F = self.hidden, self.layers, self.kernel, self.dropout, self.n_feat

        class _CNN(nn.Module):  # type: ignore[misc,name-defined]
            def __init__(self) -> None:
                super().__init__()
                self.convs = nn.ModuleList(
                    [nn.Conv1d(F if i == 0 else H, H, k, padding="same") for i in range(L)]
                )
                self.norms = nn.ModuleList([nn.GroupNorm(max(H // 8, 1), H) for _ in range(L // 2)])
                self.drop = nn.Dropout(p)
                self.head = nn.Linear(H, 1)

            def forward(self, x: Any) -> Any:  # [B, N, 14] -> [B, N]
                x = x.permute(0, 2, 1)  # Conv1d wants [B, C, N]: the bus axis is the sequence
                for i, conv in enumerate(self.convs):
                    x = conv(x)
                    if (i + 1) % 2 == 0:
                        x = self.norms[i // 2](x)
                    x = self.drop(torch.relu(x))
                return self.head(x.permute(0, 2, 1)).squeeze(-1)

        return _CNN()


class BusMLP(LearnedLocalizer):
    """The paper's lightweight arm: an identical per-bus MLP applied to every bus on its own.

    Reads nothing but the bus's own 14 numbers, so it is the cleanest statement of the temporal
    feature's power. Four layers of 128 units, LayerNorm after every second layer, ReLU, dropout,
    linear head. About 52k parameters; macro-F1 0.963 / 0.957 / 0.933 on IEEE 14 / 118 / 300 in
    the paper's zero-shot protocol (v0.4.1 data).
    """

    def _build(self, N: int) -> Any:
        torch = _torch()
        nn = torch.nn
        H, L, p, F = self.hidden, self.layers, self.dropout, self.n_feat

        class _MLP(nn.Module):  # type: ignore[misc,name-defined]
            def __init__(self) -> None:
                super().__init__()
                self.lins = nn.ModuleList([nn.Linear(F if i == 0 else H, H) for i in range(L)])
                self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(L // 2)])
                self.drop = nn.Dropout(p)
                self.head = nn.Linear(H, 1)

            def forward(self, x: Any) -> Any:  # [B, N, 14] -> [B, N]
                for i, lin in enumerate(self.lins):
                    x = lin(x)
                    if (i + 1) % 2 == 0:
                        x = self.norms[i // 2](x)
                    x = self.drop(torch.relu(x))
                return self.head(x).squeeze(-1)

        return _MLP()
