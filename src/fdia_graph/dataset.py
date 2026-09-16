"""FdiaGraph: a PyTorch-ready dataset over one ml_only_ieee{C}.h5 shard.

Static graph read once; per-record tensors sliced lazily so the whole file is never loaded. Each item:
  node_x [N,4]=[|V|,P_inj,Q_inj,theta]  node_m [N,4] availability mask
  edge_x [E,2]=[P_from,Q_from]          edge_m [E,2] availability mask
  y [N] per-bus attack label  + family / stealthy / gap / seq_id / timestep
Masked measurements (mask==0) are already zeroed; the model consumes the masks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from types import ModuleType
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from torch_geometric.data import Data

# numpy + h5py are the import-time deps (both required); torch/pandas/pandapower are lazy (optional extras).
import warnings

import numpy as np
import h5py

from .formulas.network import BranchModel, branch_admittances, branch_flows, complex_voltages

# On-disk `data/family` codes -> display name; the SDK speaks in codes.
FAMILIES = {0: "benign", 1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 5: "At", 6: "Al"}
STEALTHY_FAMILIES = {1, 5, 6}  # Aq, At, Al — evade classical bad-data detection
_FAMILY_ALIAS = {"Ao": 1, "SLS": 1, "ramp": 5, "LRA": 6}  # backward-compatible family-name aliases
_SPLIT = {"train": 0, "val": 1, "test": 2}  # on-disk `data/split` codes (precomputed)
_HELDOUT_TRAIN_EXCLUDE = {
    3,
    4,
}  # As, Ar reserved for test-only in the unseen-attack protocol (Boyaci et al. 2022)


def _torch() -> ModuleType:
    # Lazy torch import (+ install hint) so `import fdia_graph` stays light.
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError("PyTorch is required: pip install 'fdia-graph[torch]'") from e


# v0.5.0+ static per-branch physics, bus shunts and per-bus attributes stored under graph/.
_STATIC_PHYSICS = (
    "edge_r",
    "edge_x",
    "edge_b",
    "edge_g",
    "edge_gs",
    "edge_bs",
    "edge_tap",
    "edge_shift",
    "edge_status",
    "edge_is_trafo",
    "bus_shunt_g",
    "bus_shunt_b",
    "bus_type",
    "bus_vmin",
    "bus_vmax",
    "bus_base_kv",
    "bus_is_zero_inj",
    "bus_has_gen",
    "bus_base_pd",
    "bus_base_qd",
    "bus_attackable",
)
# Per-record tensors collate() stacks into a batch (those the record carries), and the scalars it gathers.
_BATCH_STACKED = (
    "node_x",
    "node_m",
    "edge_x",
    "edge_m",
    "y",
    "temporal_delta",
    "swing",
    "clean",
    "edge_clean",
    "edge_clean_full",
)
_BATCH_SCALARS = ("family", "stealthy", "seq_id", "timestep")
# Layers stored once per POOL timestep (not per record), resolved through data/timestep.
_CLEAN_LAYERS = ("clean", "edge_clean", "edge_clean_full")
# Which unit conversion each returned array takes under units="pu" (masks, labels, swing: none).
_UNIT_KIND = {
    "node_x": "node",
    "clean": "node",
    "edge_x": "edge",
    "edge_clean": "edge",
    "edge_clean_full": "edge",
    "temporal_delta": "td",
}


def family_ids(families: Sequence[Union[str, int]]) -> List[int]:
    """Family names (with the legacy aliases) or raw integer codes as integer codes."""
    if isinstance(next(iter(families)), str):
        return [k for k, v in FAMILIES.items() if v in families] + [
            _FAMILY_ALIAS[n] for n in families if n in _FAMILY_ALIAS
        ]
    return [int(f) for f in families]


class _RecordFilter(NamedTuple):
    """Which records a FdiaGraph view keeps: the partition, the families, gap records, the
    unseen-attack protocol (As/Ar held out of train and val)."""

    split: Optional[str]
    families: Optional[Sequence[Union[str, int]]]
    include_gaps: bool
    heldout: bool


def _record_mask(
    fam: np.ndarray, gap: np.ndarray, sp: Optional[np.ndarray], filt: _RecordFilter, path: str
) -> np.ndarray:
    """The kept record positions (sorted, unique): drop gap records unless asked, restrict to the
    requested partition, apply the unseen-attack protocol, and keep only the requested families."""
    keep = np.ones(len(fam), bool)
    if not filt.include_gaps:
        keep &= gap == 0  # drop gap (missing/skipped scan) records unless asked
    if filt.split is not None:
        if sp is None:
            raise ValueError(f"{path} has no split; run the split step first")
        keep &= sp == _SPLIT[filt.split]
        if filt.heldout and _SPLIT[filt.split] in (0, 1):  # test keeps As/Ar
            keep &= ~np.isin(fam, list(_HELDOUT_TRAIN_EXCLUDE))
    if filt.families is not None:
        keep &= np.isin(fam, family_ids(filt.families))
    return np.nonzero(keep)[0]


class FdiaGraph:
    """torch.utils.data.Dataset over an .h5 shard. Use `.loader(...)` for a ready DataLoader, or index items."""

    def __init__(
        self,
        path: str,
        split: Optional[str] = None,
        families: Optional[Sequence[Union[str, int]]] = None,
        include_gaps: bool = False,
        heldout: bool = False,
        format: str = "torch",
        units: str = "physical",
        preload: bool = False,
    ) -> None:
        # ONE pass over the small metadata arrays (family/gap/split) picks the kept rows;
        # the big measurement arrays are read only in __getitem__.
        self.path, self.format = path, format  # file + output flavor ("torch"/"pyg")
        # Unit system for RETURNED measurements. File stores engineering units (V p.u., P/Q MW/MVAr, theta deg);
        # units="pu" converts losslessly on the fly (P/Q + branch flows / baseMVA, theta deg->rad, V already p.u.),
        # so one shard serves both physical and normalized views. temporal_delta scales with power (->p.u.);
        # swing is a dimensionless z-score, never rescaled.
        if units not in ("physical", "pu"):
            raise ValueError("units must be 'physical' or 'pu'")
        self.units = units
        self._f = None  # lazy per-worker h5py handle, opened on first __getitem__
        with h5py.File(path, "r") as f:  # read-only; metadata copied out before block exit
            self._read_header(f)
            self._read_static_graph(f)
            self._read_layers(f)
            self.slack = self._reference_bus()
            fam = f["data/family"][:]
            gap = f["data/gap"][:]  # per-record metadata copied to RAM for filtering
            sp = f["data/split"][:] if "data/split" in f else None  # split code, or None on unsplit files
        # Kept row positions; SORTED+UNIQUE by construction, which lets to_numpy() use h5py fancy-indexing.
        self.idx = _record_mask(fam, gap, sp, _RecordFilter(split, families, include_gaps, heldout), path)
        self._mem: Optional[Dict[str, np.ndarray]] = None
        if preload and len(self.idx):
            self._preload(path)

    def _read_header(self, f: h5py.File) -> None:
        """Dims and the power base. IEEE case (14/118/300) falls back to N, then 0, for older files."""
        self.system = int(f.attrs.get("system", f.attrs.get("N", 0)))
        self.N = int(f.attrs["N"])
        self.E = int(f.attrs["E"])  # fixed graph size: bus count N, branch count E
        self.baseMVA = float(f.attrs.get("baseMVA", 100.0))  # p.u. base; v0.4.1+, default 100 MVA

    def _read_static_graph(self, f: h5py.File) -> None:
        """The static graph, read ONCE and cached as numpy (same for every record): edge index, the
        deprecated reactance, and the v0.5.0+ per-unit branch physics, bus shunts and per-bus
        attributes, each None when the file predates the schema."""
        self.edge_index_np = f["graph/edge_index"][:].astype(np.int64)
        self.edge_reactance_np = f["graph/edge_reactance"][:].astype(np.float32)
        self._phys = {}
        for _k in _STATIC_PHYSICS:
            self._phys[_k] = f[f"graph/{_k}"][:].astype(np.float64) if f"graph/{_k}" in f else None
        self.has_physics = self._phys["edge_x"] is not None
        # Forward-compat: v0.6.0 PER-RECORD data/edge_status will override static graph/edge_status;
        # None on v0.5.0 shards, so v0.5.0 loaders already read v0.6.0 shards correctly.
        self.edge_status_per_record = f["data/edge_status"][:] if "data/edge_status" in f else None

    def _read_layers(self, f: h5py.File) -> None:
        """Which optional layers the file carries: the temporal features (v0.3+, v0.4.1+) and the
        noiseless attack-free truth (v0.7.2+), stored ONCE per pool timestep and resolved per record
        via data/timestep. The clean layer is small ([Tpool,N,4] / [Tpool,E,2]) so it lives in RAM."""
        self.has_temporal = "temporal_delta" in f["data"]
        self.has_swing = "swing" in f["data"]
        self._clean_np = f["clean/node_clean"][:] if "clean/node_clean" in f else None
        self._eclean_np = f["clean/edge_clean"][:] if "clean/edge_clean" in f else None
        self.has_clean = self._clean_np is not None
        # edge_clean_full ([Tpool,E,2], the same flows on EVERY branch) is derived from the clean
        # state through Yf on first use, so it needs the clean layer and the branch physics.
        self._eclean_full_np: Optional[np.ndarray] = None
        self.has_clean_full = self.has_clean and all(
            self._phys.get(k) is not None
            for k in ("edge_r", "edge_x", "edge_b", "edge_g", "edge_tap", "edge_shift")
        )

    def _reference_bus(self) -> Optional[int]:
        """The slack bus. Explicit bus-type metadata wins when the shard carries it (ppc code 3 =
        REF); otherwise it is the one bus whose clean angle is pinned across the pool, which needs
        the clean layer (v0.7.2+) and more than one pool timestep. None when neither applies."""
        bt = self._phys.get("bus_type")
        ref = np.where(bt == 3)[0] if bt is not None else np.empty(0, int)
        if len(ref) == 1:
            return int(ref[0])
        if self._clean_np is not None and len(self._clean_np) > 1:
            # Tolerance rather than exact zero: a pinned angle can carry float32 round-off.
            pinned = np.where(self._clean_np[:, :, 3].std(axis=0) < 1e-6)[0]
            if len(pinned) == 1:
                return int(pinned[0])
        return None

    def _preload(self, path: str) -> None:
        """Optional RAM cache: bulk-read the kept rows once so __getitem__ skips per-record h5py
        overhead (the real .loader() bottleneck). About 350 MB for an ieee118 split. One contiguous
        slice plus a numpy subset beats h5py point reads (splits are near-contiguous)."""
        with h5py.File(path, "r") as f:
            dg = f["data"]
            keys = (
                ["node_x", "node_m", "edge_x", "edge_m", "y", "family", "stealthy", "seq_id", "timestep"]
                + (["temporal_delta"] if self.has_temporal else [])
                + (["swing"] if self.has_swing else [])
            )
            lo, hi = int(self.idx[0]), int(self.idx[-1]) + 1
            rel = self.idx - lo
            self._mem = {k: dg[k][lo:hi][rel] for k in keys if k in dg}

    @property
    def edge_index(self) -> torch.Tensor:
        return _torch().as_tensor(self.edge_index_np)  # [2,E] long connectivity for message passing

    @property
    def edge_reactance(self) -> torch.Tensor:
        # DEPRECATED: mixes ohms (lines) with vk_percent (transformers), not comparable. Use edge_x (per-unit).
        return _torch().as_tensor(self.edge_reactance_np)

    def _p(self, key: str) -> torch.Tensor:
        v = self._phys.get(key)
        if v is None:
            raise AttributeError(f"{key} needs a v0.5.0+ shard; this file predates the physics schema")
        return _torch().as_tensor(v)

    # Per-unit branch physics, ppc order (lines then transformers), aligned with edge_index. The
    # canonical names are branch_*; the older edge_* spellings still work but warn, because `edge_x`
    # on the dataset (series reactance) collided with `edge_x` on every record (the branch flows).
    @property
    def branch_r(self) -> torch.Tensor:
        return self._p("edge_r")  # series resistance

    @property
    def branch_x(self) -> torch.Tensor:
        return self._p("edge_x")  # series reactance

    @property
    def branch_b(self) -> torch.Tensor:
        return self._p("edge_b")  # charging susceptance

    @property
    def branch_g(self) -> torch.Tensor:
        return self._p("edge_g")  # charging conductance, transformer iron losses

    def _old_name(self, attr: str) -> torch.Tensor:
        # Deprecated edge_* spelling of a branch_* property: same tensor, plus a one-line warning.
        note = " (on records and batches edge_x is the [E,2] branch flows)" if attr == "x" else ""
        warnings.warn(
            f"ds.edge_{attr} is deprecated, use ds.branch_{attr}{note}", DeprecationWarning, stacklevel=3
        )
        return getattr(self, f"branch_{attr}")

    @property
    def edge_r(self) -> torch.Tensor:
        return self._old_name("r")

    @property
    def edge_x(self) -> torch.Tensor:
        return self._old_name("x")

    @property
    def edge_b(self) -> torch.Tensor:
        return self._old_name("b")

    @property
    def edge_g(self) -> torch.Tensor:
        return self._old_name("g")

    def _series_adm(self, imag: bool) -> torch.Tensor:
        # Series admittance 1/(r+jx). Stored on v0.7.0+ shards; derived from r,x for older ones so edge_attr
        # keeps working on every physics shard.
        key = "edge_bs" if imag else "edge_gs"
        v = self._phys.get(key)
        if v is None:
            r, x = self._phys.get("edge_r"), self._phys.get("edge_x")
            if r is None:
                return self._p(key)  # no physics at all -> standard "needs v0.5.0+" error
            z = np.asarray(r) + 1j * np.asarray(x)
            ys = np.zeros_like(z, complex)
            nz = np.abs(z) > 1e-12
            ys[nz] = 1.0 / z[nz]
            v = np.imag(ys) if imag else np.real(ys)
        return _torch().as_tensor(v)

    @property
    def branch_gs(self) -> torch.Tensor:
        return self._series_adm(False)  # series conductance, Re(1/(r+jx))

    @property
    def branch_bs(self) -> torch.Tensor:
        return self._series_adm(True)  # series susceptance, Im(1/(r+jx))

    @property
    def branch_tap(self) -> torch.Tensor:
        return self._p("edge_tap")  # transformer turns ratio, 1.0 for lines

    @property
    def branch_shift(self) -> torch.Tensor:
        return self._p("edge_shift")  # phase shift, degrees

    @property
    def edge_gs(self) -> torch.Tensor:
        return self._old_name("gs")

    @property
    def edge_bs(self) -> torch.Tensor:
        return self._old_name("bs")

    @property
    def edge_tap(self) -> torch.Tensor:
        return self._old_name("tap")

    @property
    def edge_shift(self) -> torch.Tensor:
        return self._old_name("shift")

    @property
    def edge_status(self) -> torch.Tensor:
        return self._p("edge_status")  # 1 in service, 0 out (static; see edge_status_per_record)

    @property
    def edge_is_trafo(self) -> torch.Tensor:
        return self._p("edge_is_trafo")

    # Static per-bus attributes, pandapower == ppc bus order (verified identity for case14/118/300).
    @property
    def bus_type(self) -> torch.Tensor:
        return self._p("bus_type")  # 1 PQ, 2 PV, 3 reference

    @property
    def bus_vmin(self) -> torch.Tensor:
        return self._p("bus_vmin")  # voltage limits, pu

    @property
    def bus_vmax(self) -> torch.Tensor:
        return self._p("bus_vmax")

    @property
    def bus_base_kv(self) -> torch.Tensor:
        return self._p("bus_base_kv")  # nominal voltage, kV

    @property
    def bus_is_zero_inj(self) -> torch.Tensor:
        return self._p("bus_is_zero_inj")  # generator's own zero-injection set

    @property
    def bus_has_gen(self) -> torch.Tensor:
        return self._p("bus_has_gen")  # generator or slack on the bus

    @property
    def bus_base_pd(self) -> torch.Tensor:
        return self._p("bus_base_pd")  # base-case load, MW

    @property
    def bus_base_qd(self) -> torch.Tensor:
        return self._p("bus_base_qd")  # base-case load, MVAr

    @property
    def bus_attackable(self) -> torch.Tensor:
        return self._p("bus_attackable")  # carries |p_mw|>0 load (IEEE-300 141/183 rule)

    # Shunts are a BUS property, on the Ybus diagonal, so they were not expressible in an edge schema.
    @property
    def bus_shunt_g(self) -> torch.Tensor:
        return self._p("bus_shunt_g")

    @property
    def bus_shunt_b(self) -> torch.Tensor:
        return self._p("bus_shunt_b")

    @property
    def edge_attr(self) -> torch.Tensor:
        """[E,8] stacked per-unit branch features: series impedance (r,x), branch shunt (b,g), series
        admittance (gs,bs = 1/(r+jx)), and transformer tap/shift. Drop-in replacement for edge_reactance."""
        t = _torch()
        return t.stack(
            [
                self.branch_r,
                self.branch_x,
                self.branch_b,
                self.branch_g,
                self.branch_gs,
                self.branch_bs,
                self.branch_tap,
                self.branch_shift,
            ],
            dim=1,
        )

    def _admittances(self) -> Dict[str, np.ndarray]:
        """Ybus [N,N], Yf [E,N] and Yt [E,N] from the stored branch physics, built once and cached.

        Same branch model as pandapower's makeYbus: series admittance, charging split half per end,
        tap ratio and phase shift on the from side, in-service status; bus shunts on the Ybus
        diagonal. Rows of Yf/Yt follow edge_index, columns and Ybus follow node_x bus order. So for a
        complex bus voltage vector V, `V[f] * conj(Yf @ V)` is the from-end branch flow and
        `V * conj(Ybus @ V)` the bus injection, both per-unit. Static: one topology per shard.
        """
        cached = getattr(self, "_adm", None)
        if cached is not None:
            return cached
        need = ("edge_r", "edge_x", "edge_b", "edge_g", "edge_tap", "edge_shift")
        missing = [k for k in need if self._phys.get(k) is None]
        if missing:
            raise AttributeError(
                f"ybus/yf/yt need a v0.5.0+ shard with branch physics; missing graph/{', '.join(missing)}"
            )
        p = self._phys
        Y, Yf, Yt = branch_admittances(
            BranchModel(
                p["edge_r"],
                p["edge_x"],
                p["edge_b"],
                p["edge_g"],
                p["edge_tap"],
                p["edge_shift"],
                p["edge_status"],
            ),
            self.edge_index_np,
            self.N,
            bus_shunt_g=p.get("bus_shunt_g"),
            bus_shunt_b=p.get("bus_shunt_b"),
            base_mva=self.baseMVA,
        )
        self._adm = {"ybus": Y, "yf": Yf, "yt": Yt}
        return self._adm

    def _clean_flows_full(self) -> Optional[np.ndarray]:
        """[Tpool,E,2] = [P_from, Q_from] exact AC from-end flows on EVERY branch, from the clean state.

        `edge_clean` (stored) zeroes unmetered branches to mirror `edge_x`; a graph model that supervises
        every edge needs the flow on the unmetered ones too. This is the same physics the generator
        used for `edge_clean` (`V[from] * conj(Yf @ V) * baseMVA` with the clean voltages), computed
        once from the shard's own Yf and cached, so it equals `edge_clean` wherever a flow meter
        exists. Physical units (MW, MVAr) like the stored layer; None without the clean layer or the
        branch physics.
        """
        if self._eclean_full_np is not None:
            return self._eclean_full_np
        if not self.has_clean_full or self._clean_np is None:
            return None
        # Only |V| and theta enter the flow; the same physics primitive the generator emits with.
        V = complex_voltages(
            self._clean_np[:, :, 0].astype(np.float64), self._clean_np[:, :, 3].astype(np.float64)
        )
        Sf = branch_flows(
            V, self.yf_np, self.edge_index_np[0], self.baseMVA
        )  # [Tpool,E] from-end complex flow
        self._eclean_full_np = np.stack([Sf.real, Sf.imag], axis=2).astype(np.float32)
        return self._eclean_full_np

    @property
    def ybus_np(self) -> np.ndarray:
        """Full nodal admittance matrix Ybus [N,N], complex per-unit, node_x bus order (see _admittances)."""
        return self._admittances()["ybus"]

    @property
    def yf_np(self) -> np.ndarray:
        """From-end branch admittance matrix Yf [E,N]: `V[f] * conj(Yf @ V)` is the from-end flow."""
        return self._admittances()["yf"]

    @property
    def yt_np(self) -> np.ndarray:
        """To-end branch admittance matrix Yt [E,N]: `V[t] * conj(Yt @ V)` is the to-end flow."""
        return self._admittances()["yt"]

    @property
    def ybus(self) -> torch.Tensor:
        """ybus_np as a complex128 torch tensor [N,N]."""
        return _torch().as_tensor(self.ybus_np)

    @property
    def yf(self) -> torch.Tensor:
        """yf_np as a complex128 torch tensor [E,N]."""
        return _torch().as_tensor(self.yf_np)

    @property
    def yt(self) -> torch.Tensor:
        """yt_np as a complex128 torch tensor [E,N]."""
        return _torch().as_tensor(self.yt_np)

    def _h(self) -> h5py.File:
        # Open+cache the h5py handle on first access (not __init__) so each DataLoader worker gets its
        # OWN handle — fork-safe, no shared-handle crash across processes.
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f

    def __len__(self) -> int:
        return len(self.idx)  # number of records this filtered view exposes

    def _to_units(self, arr: np.ndarray, kind: str) -> np.ndarray:
        """Convert a physical-unit array to self.units (no-op unless units=='pu'). kind:
        'node' [...,4]=[V,P,Q,theta] -> P,Q divided by baseMVA, theta deg->rad, V left (already p.u.);
        'edge' [...,2]=[P_from,Q_from] and 'td' [...,2]=[dP,dQ] -> divided by baseMVA (both power).
        Returns a float32 copy so on-disk data is never mutated."""
        if self.units != "pu":
            return arr
        b = self.baseMVA
        a = np.array(arr, dtype=np.float32, copy=True)
        if kind == "node":
            a[..., 1] /= b
            a[..., 2] /= b  # P_inj, Q_inj  MW/MVAr -> p.u.
            a[..., 3] = np.deg2rad(a[..., 3])  # theta  deg -> rad
        else:
            a /= b  # branch flows / temporal delta: power -> p.u.
        return a

    def __getitem__(self, i: int) -> Union[Dict[str, Any], "Data"]:
        """One record as a dict of tensors (or a PyG Data with format="pyg"): the measurements and
        masks, the labels and provenance, and whichever optional layers the file carries."""
        d, j = self._record_source(i)
        item = self._base_item(d, j)
        self._add_optional_layers(item, d, j)
        return self._to_pyg(item) if self.format == "pyg" else item

    def _record_source(self, i: int) -> Tuple[Any, int]:
        """Where record i is read from: the preloaded arrays (position-aligned with the view) or the
        file's data group at the real file row self.idx[i]."""
        if self._mem is not None:
            return self._mem, i
        return self._h()["data"], int(self.idx[i])

    def _base_item(self, d: Any, j: int) -> Dict[str, Any]:
        """The fields every record has: the static graph (shared tensors, not copies), the measurements
        in self.units, the masks, the label and the scalar provenance."""
        torch = _torch()
        # edge_attr needs the v0.5.0+ physics schema, so attach it only when present.
        if not hasattr(self, "_ei_t"):
            self._ei_t = self.edge_index
            self._ea_t = self.edge_attr if self.has_physics else None
        item = dict(
            edge_index=self._ei_t,  # [2,E] static connectivity (same tensor every record)
            node_x=torch.as_tensor(self._to_units(d["node_x"][j], "node"), dtype=torch.float32),  # [N,4]
            node_m=torch.as_tensor(d["node_m"][j], dtype=torch.float32),  # [N,4] availability mask
            edge_x=torch.as_tensor(self._to_units(d["edge_x"][j], "edge"), dtype=torch.float32),  # [E,2]
            edge_m=torch.as_tensor(d["edge_m"][j], dtype=torch.float32),  # [E,2] flow availability mask
            y=torch.as_tensor(d["y"][j], dtype=torch.float32),  # [N] per-bus attack label
            family=int(d["family"][j]),
            stealthy=int(d["stealthy"][j]),
            seq_id=int(d["seq_id"][j]),
            timestep=int(d["timestep"][j]),
        )
        if self._ea_t is not None:
            item["edge_attr"] = self._ea_t  # [E,8] per-unit line features; v0.5.0+ shards only
        return item

    def _add_optional_layers(self, item: Dict[str, Any], d: Any, j: int) -> None:
        """The layers a file may carry: the temporal features and the noiseless truth at the record's
        pool timestep (node, metered flows, flows on every branch)."""
        torch = _torch()
        if self.has_temporal:  # [N,2] current-minus-previous-scan injection (v0.3+)
            item["temporal_delta"] = torch.as_tensor(
                self._to_units(d["temporal_delta"][j], "td"), dtype=torch.float32
            )
        if self.has_swing:  # [N,2] windowed relative swing (recent-window z-score)
            item["swing"] = torch.as_tensor(d["swing"][j], dtype=torch.float32)
        if self._clean_np is None:
            return
        t = item["timestep"]
        # torch.tensor COPIES: with units="physical" _to_units is a no-op view into the shared cached
        # table, and an aliased tensor would let one record's in-place edit corrupt every other record.
        item["clean"] = torch.tensor(self._to_units(self._clean_np[t], "node"), dtype=torch.float32)
        if self._eclean_np is not None:  # [E,2] exact true flows (unmetered zeroed)
            item["edge_clean"] = torch.tensor(self._to_units(self._eclean_np[t], "edge"), dtype=torch.float32)
        ecf = self._clean_flows_full()
        if ecf is not None:  # [E,2] exact true flows on EVERY branch, metered or not
            item["edge_clean_full"] = torch.tensor(self._to_units(ecf[t], "edge"), dtype=torch.float32)

    # Dict-record key -> PyG attribute name. Everything not listed keeps its dict name, so a field
    # added to __getitem__ shows up in PyG records and batches without touching this method. The
    # static [E,8] branch physics travel as edge_phys because PyG's edge_attr slot holds the per-record
    # branch flows, the same contract torch_data.pyg_stream follows.
    _PYG_RENAME = {"node_x": "x", "node_m": "node_mask", "edge_m": "edge_mask", "edge_attr": "edge_phys"}

    def _to_pyg(self, item: Dict[str, Any]) -> "Data":
        # Repackage the dict record as a torch_geometric Data object, mechanically. PyG batches every
        # per-node/per-edge tensor along dim 0 and every scalar into a [B] tensor, so masks, temporal
        # features, clean layers and metadata all ride along.
        try:
            from torch_geometric.data import Data
        except ImportError as e:
            raise ImportError("PyG is required for format='pyg': pip install 'fdia-graph[pyg]'") from e
        fields = {self._PYG_RENAME.get(k, k): v for k, v in item.items()}
        fields["edge_attr"] = item["edge_x"]  # same tensor as fields["edge_x"], no copy until batching
        if self.slack is not None:
            fields["slack"] = self.slack  # reference-bus index, batched to [B] like family
        return Data(**fields)

    @staticmethod
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Dict-format collate: every record shares N and E, so per-record tensors stack into a batch
        dimension, the static graph rides along once, and scalar metadata becomes one long tensor.
        (PyG has its own loader.)"""
        torch = _torch()
        out = {k: torch.stack([b[k] for b in batch]) for k in _BATCH_STACKED if k in batch[0]}
        out["edge_index"] = batch[0]["edge_index"]  # same topology for every sample
        if "edge_attr" in batch[0]:  # absent on older no-physics shards
            out["edge_attr"] = batch[0]["edge_attr"]
        for k in _BATCH_SCALARS:
            out[k] = torch.as_tensor([b[k] for b in batch], dtype=torch.long)  # [B] per record
        return out

    def loader(
        self, batch_size: int = 64, shuffle: Optional[bool] = None, num_workers: int = 0, **kw: Any
    ) -> "DataLoader":
        """Return a ready DataLoader. Shuffle defaults on for train-like use; masks/labels included per batch."""
        torch = _torch()
        from torch.utils.data import DataLoader

        if shuffle is None:
            shuffle = True  # default on (mostly used for training)
        if self.format == "pyg":
            # PyG's own DataLoader does graph-aware batching (offsets edge_index); no custom collate needed.
            from torch_geometric.loader import DataLoader as PyGLoader

            return PyGLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kw)
        # Plain-dict path: standard torch DataLoader wired to our stacking collate above.
        # self is a structural Dataset (__len__/__getitem__) but can't inherit torch's Dataset,
        # since torch is an optional lazy import; pyright can't see the duck-typed conformance.
        return DataLoader(
            self,  # type: ignore[reportArgumentType]
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate,
            **kw,
        )

    def summary(self) -> Dict[str, Any]:
        # Cheap overview: read only the family column for this view's rows, tally per family.
        with h5py.File(self.path, "r") as f:
            fam = f["data/family"][:][self.idx]  # family codes for just the kept rows
        return dict(
            system=self.system,
            N=self.N,
            E=self.E,
            n=len(self),
            families={FAMILIES[k]: int((fam == k).sum()) for k in FAMILIES if (fam == k).any()},
        )

    # ------------------------------------------------------------------ #
    #  Whole-split exporters: unlike __getitem__/.loader() (stream one record), these pull the ENTIRE
    #  filtered split into memory. All share to_numpy() as the single HDF5 read, so views are identical.
    # ------------------------------------------------------------------ #
    def _default_fields(self) -> List[str]:
        """Every per-record array the file carries, in the order to_numpy returns them."""
        return (
            ["node_x", "node_m", "edge_x", "edge_m", "y"]
            + (["temporal_delta"] if self.has_temporal else [])
            + (["swing"] if self.has_swing else [])
            + (["clean", "edge_clean"] if self.has_clean else [])
            + (["edge_clean_full"] if self.has_clean_full else [])
            + ["family", "stealthy", "seq_id", "timestep"]
        )

    def _clean_layers(self, want: Sequence[str], ts: np.ndarray) -> Dict[str, np.ndarray]:
        """The requested clean layers gathered per record through the pool timestep."""
        out: Dict[str, np.ndarray] = {}
        if "clean" in want and self._clean_np is not None:
            out["clean"] = self._clean_np[ts]
        if "edge_clean" in want and self._eclean_np is not None:
            out["edge_clean"] = self._eclean_np[ts]
        if "edge_clean_full" in want:
            ecf = self._clean_flows_full()
            if ecf is not None:
                out["edge_clean_full"] = ecf[ts]
        return out

    def to_numpy(self, fields: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        """Return the whole selected split as a dict of numpy arrays (batched over records).

        Keys: node_x [n,N,4], node_m, edge_x [n,E,2], edge_m, y [n,N], family/stealthy/seq_id/timestep [n],
        plus the static graph: edge_index [2,E], edge_reactance [E]. `fields` optionally limits the per-record
        arrays read (the graph arrays are always included since they're tiny and needed to interpret edges).
        """
        want = list(fields) if fields else self._default_fields()
        per_record = [k for k in want if k not in _CLEAN_LAYERS]
        clean_want = [k for k in want if k in _CLEAN_LAYERS]
        # Static graph arrays always included (tiny, and needed to interpret edges).
        out = {"edge_index": self.edge_index_np, "edge_reactance": self.edge_reactance_np}
        with h5py.File(self.path, "r") as f:
            d = f["data"]
            for k in per_record:
                # self.idx is sorted-unique by construction, as h5py fancy-indexing requires
                out[k] = d[k][self.idx]  # one bulk gather per field -> [n, ...] numpy array
            if clean_want:
                out.update(self._clean_layers(clean_want, d["timestep"][self.idx]))
        return self._in_units(out)

    def _in_units(self, out: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """The returned arrays in self.units: power and angle arrays converted to per unit when asked,
        masks, labels and the swing z-score untouched."""
        if self.units == "pu":
            for k, kind in _UNIT_KIND.items():
                if k in out:
                    out[k] = self._to_units(out[k], kind)
        return out

    def to_torch(
        self, fields: Optional[Sequence[str]] = None, device: Optional[Union[str, "torch.device"]] = None
    ) -> Dict[str, torch.Tensor]:
        """Same data as to_numpy(), but as torch tensors (floats stay float32, label ids stay int64).
        Handy when you want the full split resident as tensors rather than streamed via a DataLoader."""
        torch = _torch()
        np_ = self.to_numpy(fields)  # single source-of-truth HDF5 read
        int_keys = {
            "family",
            "stealthy",
            "seq_id",
            "timestep",
            "edge_index",
        }  # -> int64; measurements -> float32
        out = {}
        for k, v in np_.items():
            t = torch.as_tensor(v)
            t = t.long() if k in int_keys else t.float()
            out[k] = t.to(device) if device else t  # optionally move onto the target device
        return out

    def to_tf(self, fields: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Same data as to_numpy(), but as TensorFlow tensors (requires tensorflow installed).
        Returns a dict of tf.Tensors; wrap in tf.data.Dataset.from_tensor_slices(...) if you want a pipeline."""
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError("TensorFlow is required for to_tf(): pip install tensorflow") from e
        # Same single to_numpy() read, wrapped as tf.Tensors.
        return {k: tf.convert_to_tensor(v) for k, v in self.to_numpy(fields).items()}

    def to_pandas(self, flatten_features: bool = True) -> "pd.DataFrame":
        """Return a pandas DataFrame — one row per record — for tabular analysis / filtering.

        Always includes the metadata columns (family name, stealthy flag, seq_id, timestep, and n_attacked
        buses). Because the measurement graph is 3-D (records × buses × channels), `flatten_features=True`
        additionally spreads the per-bus/branch measurements into flat columns (V_b{n}, Pinj_b{n}, Pflow_e{n},
        …) so the whole split is a plain table; set it False for just the metadata (much narrower)."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for to_pandas(): pip install pandas") from e
        a = self.to_numpy()  # same single HDF5 read backing every exporter
        # Metadata frame, one row per record; n_attacked_buses = row-sum of the [n,N] label matrix.
        df = pd.DataFrame(
            {
                "family": [FAMILIES[int(k)] for k in a["family"]],  # readable family name
                "family_id": a["family"],
                "stealthy": a["stealthy"].astype(bool),
                "seq_id": a["seq_id"],
                "timestep": a["timestep"],
                "n_attacked_buses": a["y"].sum(axis=1).astype(int),
            }
        )
        if flatten_features:
            N, E = self.N, self.E
            # node_x[:, :, ci] is [n, N] for channel ci -> N columns V_b0.., Pinj_b0.., etc.
            for ci, nm in enumerate(["V", "Pinj", "Qinj", "theta"]):
                cols = pd.DataFrame(a["node_x"][:, :, ci], columns=[f"{nm}_b{b}" for b in range(N)])
                df = pd.concat([df, cols], axis=1)
            # edge_x[:, :, ci] is [n, E] for channel ci -> E columns Pflow_e0.., Qflow_e0..
            for ci, nm in enumerate(["Pflow", "Qflow"]):
                cols = pd.DataFrame(a["edge_x"][:, :, ci], columns=[f"{nm}_e{e}" for e in range(E)])
                df = pd.concat([df, cols], axis=1)
            # per-bus binary attack labels -> N columns attacked_b0..attacked_b{N-1}
            label_cols = pd.DataFrame(a["y"].astype(int), columns=[f"attacked_b{b}" for b in range(N)])
            df = pd.concat([df, label_cols], axis=1)
        return df
