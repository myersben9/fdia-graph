"""FdiaGraph: a PyTorch-ready dataset over one HDF5 file, a timeline (one row per frame in time order,
`kind="timeline"`).

The static graph is read once; records are sliced lazily, so the whole file is never loaded. An item
holds node_x [N,4] = [|V|, P_inj, Q_inj, theta], edge_x [E,2] = [P_from, Q_from], their masks node_m
and edge_m (a masked reading is already zero), the per-bus label y [N] and family / stealthy /
seq_id / timestep (docs/reference/DATA_DICTIONARY.md). The class is one mixin per file: `records.py`
(one record: `ds[i]`, `loader`), `export.py` (the whole view), `sequence.py` (windows and episodes),
`graph.py` (the static graph), `physics.py` (admittances and clean flows), over `base.py`.
`order="time"` keeps the file order (chronological; `windows` and `episodes` need it); `order="random"`
is the same frames in a permutation fixed by `seed`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, Optional, Union

import h5py

# numpy + h5py are the import-time deps (both required); torch/pandas are lazy (optional extras, see base._torch).
import numpy as np

from .. import schema
from ..constants.channels import NodeChannel
from ..constants.families import HELDOUT_CODES
from ..errors import NotATimeline
from ..models.choices import Split
from ..models.config import LoadOptions
from ..schema import SPLIT_CODE as _SPLIT
from ..schema import STATIC_PHYSICS as _STATIC_PHYSICS
from ..schema import Attr
from .base import family_ids
from .export import ExportMixin
from .graph import GraphMixin
from .physics import AdmittanceMixin
from .records import RecordsMixin
from .sequence import SequenceMixin, read_episodes


class _RecordFilter(NamedTuple):
    """Which records a FdiaGraph view keeps: the partition, the families, the
    unseen-attack protocol (As/Ar held out of train and val)."""

    split: Optional[str]
    families: Optional[Sequence[Union[str, int]]]
    heldout: bool


def _record_mask(fam: np.ndarray, sp: Optional[np.ndarray], filt: _RecordFilter) -> np.ndarray:
    """The kept record positions (sorted, unique): restrict to the requested partition, apply the
    unseen-attack protocol, and keep only the requested families."""
    keep = np.ones(len(fam), bool)
    if filt.split is not None:
        code = _SPLIT[filt.split]
        keep &= (sp == code) if sp is not None else False  # a file without the column is refused first
        if filt.heldout and filt.split != Split.TEST:  # test keeps As/Ar
            keep &= ~np.isin(fam, list(HELDOUT_CODES))
    if filt.families is not None:
        keep &= np.isin(fam, family_ids(filt.families))
    return np.nonzero(keep)[0]


class FdiaGraph(GraphMixin, AdmittanceMixin, RecordsMixin, ExportMixin, SequenceMixin):
    """torch.utils.data.Dataset over one .h5 file. Use `.loader(...)` for a ready DataLoader, or index items."""

    def __init__(
        self,
        path: str,
        split: Optional[str] = None,
        families: Optional[Sequence[Union[str, int]]] = None,
        heldout: bool = False,
        format: str = "torch",
        units: str = "physical",
        preload: bool = False,
        order: str = "time",
        seed: int = 0,
    ) -> None:
        # One pass over the small metadata arrays (family, split) picks the kept rows; the big
        # measurement arrays are read only in __getitem__.
        self.path, self.format = path, format  # file + output flavor ("torch"/"pyg")
        # The file stores physical units (V pu, P/Q MW/MVAr, theta degrees); units="pu" converts the
        # returned arrays on the fly (power over baseMVA, theta to radians). The swing z-score is
        # dimensionless and never rescaled.
        options = LoadOptions(split, units, order, format)  # checked before the file is opened
        split, units, order = options.split, options.units, options.order
        format = options.format
        if families is not None:
            family_ids(families)  # an unknown family fails here, not after the read
        self.units = units
        self._f = None  # lazy per-worker h5py handle, opened on first __getitem__
        with h5py.File(path, "r") as f:  # read-only; metadata copied out before block exit
            self._read_header(f)
            self._read_static_graph(f)
            self._read_layers(f)
            self.slack = self._reference_bus()
            fam = f[schema.FAMILY][:]
            sp = f[schema.SPLIT][:] if schema.SPLIT in f else None  # split code, or None on unsplit files
            self._episodes = read_episodes(f, schema.Group.EPISODES in f)
        self.has_split = sp is not None
        if split is not None:
            self.require("split", by=f"split={split!r} (run the split step first)")
        # Kept row positions; SORTED+UNIQUE by construction, which lets to_numpy() use h5py fancy-indexing.
        self.idx = _record_mask(fam, sp, _RecordFilter(split, families, heldout))
        # order="random": the same rows in a permutation fixed by the seed, applied as a view index.
        self._perm = np.random.default_rng(seed).permutation(len(self.idx)) if order == "random" else None
        self._mem: Optional[dict[str, np.ndarray]] = None
        if preload and len(self.idx):
            self._preload(path)

    def _read_header(self, f: h5py.File) -> None:
        """Dims and the power base, of a timeline file: any other file is refused."""
        a = f.attrs
        if str(a.get(Attr.KIND, "")) != schema.KIND_TIMELINE:
            raise NotATimeline(
                f"{self.path} is not a timeline file; a record shard of a data release before v0.8.0 reads "
                "with fdia-graph 0.21 (pip install fdia-graph==0.21.0)"
            )
        self.system = int(a.get(Attr.SYSTEM, a.get(Attr.N, 0)))
        self.N = int(a[Attr.N])
        self.E = int(a[Attr.E])  # fixed graph size: bus count N, branch count E
        self.baseMVA = float(a.get(Attr.BASEMVA, 100.0))  # p.u. base; v0.4.1+, default 100 MVA
        # the flow layers' columns
        self.edge_feat = str(a.get(Attr.EDGE_FEAT, schema.EDGE_FEAT_DEFAULT)).split(",")
        # how pmu_i holds a current phasor: real and imaginary part unless the file says otherwise
        self.current_form = str(a.get(Attr.CURRENT_FORM, schema.CURRENT_RECTANGULAR))

    def _read_static_graph(self, f: h5py.File) -> None:
        """The static graph, read once and cached as numpy: the edge index and the per-unit branch
        physics, bus shunts and per-bus attributes, each None when the file lacks it."""
        self.edge_index_np = f[schema.EDGE_INDEX][:].astype(np.int64)
        self._phys = {}
        for _k in _STATIC_PHYSICS:
            p = schema.path(schema.Group.GRAPH, _k)
            self._phys[_k] = f[p][:].astype(np.float64) if p in f else None
        self.has_physics = self._phys["edge_x"] is not None
        # Reserved: a per-record data/edge_status is read (None when absent) but not yet applied; Ybus
        # and the flows use the static graph/edge_status.
        self.edge_status_per_record = f[schema.EDGE_STATUS][:] if schema.EDGE_STATUS in f else None

    def _read_layers(self, f: h5py.File) -> None:
        """Which optional layers the file carries. The clean layer (the noiseless attack-free truth)
        is stored once per pool timestep and resolved per record through data/timestep; it is small
        ([Tpool,N,4] and [Tpool,E,2]), so it lives in RAM."""
        self.has_temporal = schema.TEMPORAL_DELTA in f
        self.has_swing = schema.SWING in f
        self._clean_np = f[schema.NODE_CLEAN][:] if schema.NODE_CLEAN in f else None
        self._eclean_np = f[schema.EDGE_CLEAN][:] if schema.EDGE_CLEAN in f else None
        self.has_clean = self._clean_np is not None
        self.has_benign = schema.NODE_BENIGN in f  # timeline files: the attack-removed layer per frame
        self.has_currents = schema.PMU_I in f  # hybrid-meter timelines: the PMU branch-current phasors
        # edge_clean_full ([Tpool,E,2], the same flows on EVERY branch) is derived from the clean
        # state through Yf on first use, so it needs the clean layer and the branch physics.
        self._eclean_full_np: Optional[np.ndarray] = None
        physics = (self._phys.get(k) for k in schema.BRANCH_MODEL_PHYSICS.values())
        self.has_clean_full = self.has_clean and all(p is not None for p in physics)

    def _reference_bus(self) -> Optional[int]:
        """The slack bus: the one bus of type 3 (REF) when the file carries bus types, otherwise the
        one bus whose clean angle is pinned across the pool (needs the clean layer and more than one
        pool timestep). None when neither applies."""
        bt = self._phys.get("bus_type")
        ref = np.where(bt == 3)[0] if bt is not None else np.empty(0, int)
        if len(ref) == 1:
            return int(ref[0])
        if self._clean_np is not None and len(self._clean_np) > 1:
            # Tolerance rather than exact zero: a pinned angle can carry float32 round-off.
            pinned = np.where(self._clean_np[:, :, NodeChannel.THETA].std(axis=0) < 1e-6)[0]
            if len(pinned) == 1:
                return int(pinned[0])
        return None

    def _preload(self, path: str) -> None:
        """Optional RAM cache: the kept rows read once, so __getitem__ skips the per-record h5py
        overhead. About 350 MB for an ieee118 split. One contiguous slice plus a numpy subset is
        faster than h5py point reads."""
        with h5py.File(path, "r") as f:
            dg = f[schema.Group.DATA]
            keys = (
                ["node_x", "node_m", "edge_x", "edge_m", "y", "family", "stealthy", "seq_id", "timestep"]
                + (["temporal_delta"] if self.has_temporal else [])
                + (["swing"] if self.has_swing else [])
            )
            lo, hi = int(self.idx[0]), int(self.idx[-1]) + 1
            rel = self.idx - lo
            self._mem = {k: dg[k][lo:hi][rel] for k in keys if k in dg}
            self._mem.update({k: f[p][lo:hi][rel] for k, p in self._frame_layers().items()})

    def __len__(self) -> int:
        return len(self.idx)  # number of records this filtered view exposes
