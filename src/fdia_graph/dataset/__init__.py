"""FdiaGraph: a PyTorch-ready dataset over one HDF5 file, a timeline (one row per frame in time order,
`kind="timeline"`).

Static graph read once; per-record tensors sliced lazily so the whole file is never loaded. Each item:
  node_x [N,4]=[|V|,P_inj,Q_inj,theta]  node_m [N,4] availability mask
  edge_x [E,2]=[P_from,Q_from]          edge_m [E,2] availability mask
  y [N] per-bus attack label  + family / stealthy / seq_id / timestep
  and, on a timeline, benign / edge_benign (the attack removed, the noise kept).
Masked measurements (mask==0) are already zeroed; the model consumes the masks. `order="random"`
is the same frames in a permutation fixed by `seed`; `order="time"` keeps the file order, which on
a timeline is chronological, so `ds.windows` and `ds.episodes` work on it.

The class is six mixins, one file each, each owning one kind of question:
  records.py   RecordsMixin    one record at a time: `ds[i]`, `len(ds)`, `collate`, `loader`
  export.py    ExportMixin     the whole view at once: `export`, `summary`
  sequence.py  SequenceMixin   the view as a time series: `windows`, `episodes`
  graph.py     GraphMixin      the static graph: `edge_index`, `branch_*`, `bus_shunt_*`, `edge_attr`
  physics.py   AdmittanceMixin the admittances built from it: `ybus`, `yf`, `yt`, the clean flows
  base.py      DatasetBase     the attributes `__init__` sets and the mixins read, the tables
`__init__` itself opens the file, reads the header and the static graph, and selects the rows.
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
        # ONE pass over the small metadata arrays (family/gap/split) picks the kept rows;
        # the big measurement arrays are read only in __getitem__.
        self.path, self.format = path, format  # file + output flavor ("torch"/"pyg")
        # Unit system for RETURNED measurements. File stores engineering units (V p.u., P/Q MW/MVAr, theta deg);
        # units="pu" converts losslessly on the fly (P/Q + branch flows / baseMVA, theta deg->rad, V already p.u.),
        # so one shard serves both physical and normalized views. temporal_delta scales with power (->p.u.);
        # swing is a dimensionless z-score, never rescaled.
        # the arguments are converted before the file is opened, so a wrong one never reads the file
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
        self.edge_feat = str(a.get(Attr.EDGE_FEAT, schema.EDGE_FEAT_DEFAULT)).split(
            ","
        )  # the flow layers' columns
        # how pmu_i holds a current phasor: real and imaginary part unless the file says otherwise
        self.current_form = str(a.get(Attr.CURRENT_FORM, schema.CURRENT_RECTANGULAR))

    def _read_static_graph(self, f: h5py.File) -> None:
        """The static graph, read ONCE and cached as numpy (same for every record): edge index and the
        per-unit branch physics, bus shunts and per-bus
        attributes, each None when the file predates the schema."""
        self.edge_index_np = f[schema.EDGE_INDEX][:].astype(np.int64)
        self._phys = {}
        for _k in _STATIC_PHYSICS:
            p = schema.path(schema.Group.GRAPH, _k)
            self._phys[_k] = f[p][:].astype(np.float64) if p in f else None
        self.has_physics = self._phys["edge_x"] is not None
        # Forward-compat: v0.6.0 PER-RECORD data/edge_status will override static graph/edge_status;
        # None on v0.5.0 shards, so v0.5.0 loaders already read v0.6.0 shards correctly.
        self.edge_status_per_record = f[schema.EDGE_STATUS][:] if schema.EDGE_STATUS in f else None

    def _read_layers(self, f: h5py.File) -> None:
        """Which optional layers the file carries: the temporal features (v0.3+, v0.4.1+) and the
        noiseless attack-free truth (v0.7.2+), stored ONCE per pool timestep and resolved per record
        via data/timestep. The clean layer is small ([Tpool,N,4] / [Tpool,E,2]) so it lives in RAM."""
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
            pinned = np.where(self._clean_np[:, :, NodeChannel.THETA].std(axis=0) < 1e-6)[0]
            if len(pinned) == 1:
                return int(pinned[0])
        return None

    def _preload(self, path: str) -> None:
        """Optional RAM cache: bulk-read the kept rows once so __getitem__ skips per-record h5py
        overhead (the real .loader() bottleneck). About 350 MB for an ieee118 split. One contiguous
        slice plus a numpy subset beats h5py point reads (splits are near-contiguous)."""
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
