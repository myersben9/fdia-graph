"""The one writer of a timeline file, for every grid: `TimelineWriter` owns the open HDF5 file.

The per-frame layers are created empty at full length and filled in batches as the walk stores its
frames. After the walk come the meter masks, the per-frame columns, the episode tables, the
attributes and the temporal features. A source holds a writer and calls its methods.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Optional, Union

import h5py
import numpy as np

from .. import schema
from ..constants.families import BENIGN_CODE, FAMILIES, FAMILY_CODE, STEALTHY_FAMILIES
from ..formulas.temporal import recent_change_scale
from ..models.choices import Split
from ..models.frames import AttackVector, Frame, LayerSet, MinimizerResult
from ..models.plan import Episode, EpisodeRecord, Placement
from ..schema import Attr, Group

_BATCH = 256  # frames staged in memory between two flushes to the file
CleanSlice = Callable[[int, int], tuple[np.ndarray, np.ndarray]]  # the noiseless truth of frames a..b-1


class TimelineWriter:
    """One timeline file, open for the whole walk, and the frames staged for it.

    Frames are stored in time order (`store`): node_x / edge_x are the observed scan, the benign layers
    the same scan with the attack removed and the noise kept, the clean layers the noiseless truth.
    The layers reach the file in batches of _BATCH frames, so memory is bounded whatever the length.
    `dims` are the file's frames, buses and branches."""

    # ---- the layout this writer creates: the datasets, their shapes and their types ------------------
    CHUNK_ROWS = 128  # frames per chunk: a per-frame dataset is chunked along the frame axis
    FLOW_WIDTH = -1  # in the layer tables: as many columns as a flow layer has (`LayerSet.edge_width`)
    # the per-frame layers: path -> (one row per bus "C" or per branch "E", columns (0: none), dtype)
    FRAME_LAYERS = {
        schema.NODE_X: ("C", 4, np.float32),
        schema.EDGE_X: ("E", FLOW_WIDTH, np.float32),
        schema.Y: ("C", 0, np.uint8),
        schema.TEMPORAL_DELTA: ("C", 2, np.float32),
        schema.SWING: ("C", 2, np.float32),
        schema.NODE_BENIGN: ("C", 4, np.float32),
        schema.EDGE_BENIGN: ("E", FLOW_WIDTH, np.float32),
        schema.NODE_CLEAN: ("C", 4, np.float32),
        schema.EDGE_CLEAN: ("E", FLOW_WIDTH, np.float32),
        schema.NODE_TAMPER: ("C", 4, np.uint8),
        schema.EDGE_TAMPER: ("E", FLOW_WIDTH, np.uint8),
        schema.PMU_I: ("E", 4, np.float32),  # the PMU branch-current layers
        schema.PMU_I_BENIGN: ("E", 4, np.float32),
        schema.PMU_I_TAMPER: ("E", 4, np.uint8),
    }
    # filled after the walk; a `LayerSet` may leave them out
    TEMPORAL_LAYERS = (schema.TEMPORAL_DELTA, schema.SWING)
    # the columns of the episode tables under episodes/, in the order of a row's values: (name, dtype)
    EPISODE_COLUMNS = (
        (schema.EPISODE_ONSET, np.int32),
        (schema.EPISODE_LENGTH, np.int32),
        (schema.EPISODE_FAMILY, np.int8),
    )
    MIN_COLUMNS = (
        (schema.EPISODE_MIN_EPISODE, np.int32),
        (schema.EPISODE_MIN_DEVICES, np.int32),
        (schema.EPISODE_MIN_CHANNELS, np.int32),
        (schema.EPISODE_MIN_EVALUATED, np.int32),
    )
    AM_COLUMNS = (
        (schema.EPISODE_AM_EPISODE, np.int32),
        (schema.EPISODE_AM_LINE, np.int32),
        (schema.EPISODE_AM_RATING, np.float32),
        (schema.EPISODE_AM_TARGET, np.float32),
        (schema.EPISODE_AM_REACHED, np.float32),
        (schema.EPISODE_AM_EMITTED, np.float32),
    )

    # ---- 1. stage frames: the file is opened, its layers created, the walk's frames stored ------------

    def __init__(
        self, path: str, dims: tuple[int, int, int], clean: CleanSlice, spec: LayerSet = LayerSet()
    ) -> None:
        T, C, E = dims
        self.dims, self.spec, self._n, self._base = dims, spec, min(_BATCH, T), 0
        self._layers: dict[str, np.ndarray] = {}  # the staged batch of each per-frame layer
        # their datasets, held open for the whole walk: HDF5 then lays the chunks out in one order
        self._sink: dict[str, h5py.Dataset] = {}
        self.family = np.zeros(T, np.int16)
        self.seq_id = np.full(T, -1, np.int32)
        self.mag_bus: list[np.ndarray] = [np.zeros(0, np.int32)] * T
        self.mag: list[np.ndarray] = [np.zeros(0, np.float32)] * T
        # the meter plan and the mask of the PMU currents, from the first stored frame
        self.node_m: Optional[np.ndarray] = None
        self.edge_m: Optional[np.ndarray] = None
        self.i_m: Optional[np.ndarray] = None
        self.episodes: list[Episode] = []  # the rows of episodes/, onsets counted in the file's frames
        self.min_rows: list[tuple[int, MinimizerResult]] = []  # (episode, fewest-tamper result), knob on
        self.am_rows: list[tuple[int, int, float, float, float, float]] = []  # (episode, *a row of AmRows)
        # the attack vector of the last stored frame, observed minus its benign twin (zero when benign):
        # an adjacent episode's stealth bound starts from it
        self.last_attack = AttackVector(np.zeros((C, 4)), np.zeros((E, spec.edge_width)), np.zeros((E, 4)))
        self.attacked = 0  # frames stored so far with at least one attacked bus
        self._clean = clean
        self.f = h5py.File(path, "w")
        try:
            self._create_layers()
            self._clean_batch = clean(0, self._n)
        except BaseException:  # no open handle is left behind: the caller removes the unfinished file
            self.f.close()
            raise

    def _create_layers(self) -> None:
        """Create the groups and the per-frame datasets the layer set asks for, each with a batch of
        frames staged for it."""
        for group in (Group.DATA, Group.BENIGN, Group.CLEAN, Group.ATTACK):
            self.f.create_group(group)
        spec, rows = self.spec, {"C": self.dims[1], "E": self.dims[2]}
        table = {n: v for n, v in self.FRAME_LAYERS.items() if spec.temporal or n not in self.TEMPORAL_LAYERS}
        for name, (axis, width, dtype) in table.items():
            width = spec.edge_width if width == self.FLOW_WIDTH else width
            shape = (rows[axis], width) if width else (rows[axis],)
            self._sink[name] = self._create(name, shape, dtype)
            self._layers[name] = np.zeros((self._n, *shape), dtype)

    def _create(self, name: str, shape: tuple[int, ...], dtype: type) -> h5py.Dataset:
        """A per-frame dataset at full length, chunked along the frame axis and gzipped."""
        full, chunks = (self.dims[0], *shape), (min(self.CHUNK_ROWS, self.dims[0]), *shape)
        return self.f.create_dataset(name, shape=full, dtype=dtype, chunks=chunks, compression="gzip")

    def header(self, base_mva: float, seed: int) -> dict[str, Union[int, float, str]]:
        """The attributes every file carries: its size, the feature legends and their units. Values are
        physical (node_x = [V pu, P_inj MW, Q_inj MVAr, theta deg], flows MW/MVAr); baseMVA lets the
        loader also serve a per-unit view."""
        T, N, E = self.dims
        flows = schema.FLOW_NAMES[: self.spec.edge_width]
        return {
            Attr.SYSTEM: N,
            Attr.N: N,
            Attr.E: E,
            Attr.N_RECORDS: T,
            Attr.NODE_FEAT: schema.NODE_FEAT_LEGEND,
            Attr.EDGE_FEAT: ",".join(flows),
            Attr.NODE_UNITS: schema.NODE_UNITS_LEGEND,
            Attr.EDGE_UNITS: ",".join(f"{name}:{unit}" for name, unit in zip(flows, schema.FLOW_UNITS)),
            Attr.BASEMVA: base_mva,
            Attr.SEED: seed,
        }

    def write_graph(
        self,
        edges: np.ndarray,
        values: Mapping[str, Optional[np.ndarray]],
        legend: Mapping[str, str] = schema.GRAPH_LEGEND,
    ) -> None:
        """Write graph/: the static topology (`edges [2, E]`) with the per-unit branch physics and bus
        shunts that rebuild Ybus exactly (against makeYbus: 7e-15, 3e-14 and 5e-13 on IEEE 14, 118 and
        300). `values` are the grid's datasets by their `Static` name; a
        name the reader's table does not list is not written."""
        gg = self.f.create_group(Group.GRAPH)
        gg.create_dataset(schema.EDGE_INDEX.split("/")[1], data=edges)
        for name in schema.STATIC_PHYSICS:  # the reader's table, in its order; what this grid has of it
            if name in values:
                gg.create_dataset(name, data=values[name])
        gg.attrs.update(legend)

    def store(self, t: int, fid: int, frame: Frame, sid: int = -1) -> None:
        """Store frame t (frames arrive in order): an attacked frame of family `fid` and episode `sid`
        with its un-attacked twin, or a benign one."""
        if t >= self._base + self._n:
            self.flush()
        r = t - self._base
        L = self._layers
        bnx = frame.node_x if frame.benign_node_x is None else frame.benign_node_x
        bex = frame.edge_x if frame.benign_edge_x is None else frame.benign_edge_x
        L[schema.NODE_X][r] = frame.node_x
        L[schema.NODE_BENIGN][r] = bnx
        L[schema.Y][r] = frame.y
        L[schema.EDGE_X][r] = frame.edge_x
        L[schema.EDGE_BENIGN][r] = bex
        L[schema.NODE_CLEAN][r], L[schema.EDGE_CLEAN][r] = self._clean_batch[0][r], self._clean_batch[1][r]
        self.family[t] = fid
        self.seq_id[t] = sid if fid else -1
        self.attacked += int(frame.y.any())
        if self.node_m is None:
            self.node_m, self.edge_m, self.i_m = frame.node_m, frame.edge_m, frame.i_m
        self.last_attack = AttackVector(
            np.asarray(frame.node_x, np.float64) - np.asarray(bnx, np.float64),
            np.asarray(frame.edge_x, np.float64) - np.asarray(bex, np.float64),
            self._store_currents(r, fid, frame),
        )
        self._store_attack(t, fid, frame, bnx, bex)

    def _store_currents(self, r: int, fid: int, frame: Frame) -> np.ndarray:
        """Stage a frame's PMU branch-current layers and return its attack vector on them (observed
        minus un-attacked, zero on a benign frame)."""
        bix = frame.i_x if frame.benign_i_x is None else frame.benign_i_x
        L = self._layers
        L[schema.PMU_I][r], L[schema.PMU_I_BENIGN][r] = frame.i_x, bix
        tamper = frame.i_tamper if fid != BENIGN_CODE and frame.i_tamper is not None else frame.i_x != bix
        L[schema.PMU_I_TAMPER][r] = tamper if fid != BENIGN_CODE else 0
        return np.asarray(frame.i_x, np.float64) - np.asarray(bix, np.float64)

    def _store_attack(self, t: int, fid: int, frame: Frame, bnx: np.ndarray, bex: np.ndarray) -> None:
        """Stage the designed magnitudes and the tamper masks of a frame. A benign frame zeroes its
        masks, because the staged batch rows are reused between flushes."""
        r = t - self._base
        if fid == BENIGN_CODE:
            self._layers[schema.NODE_TAMPER][r] = 0
            self._layers[schema.EDGE_TAMPER][r] = 0
            return
        self.mag_bus[t] = np.asarray(frame.mag_bus, np.int32)
        self.mag[t] = np.asarray(frame.mag, np.float32)
        if frame.tamper is not None:  # a stealthy family: the meters its local false state moves
            node, edge = frame.tamper
        else:  # in-place corruption: exactly the meters that changed
            node, edge = frame.node_x != bnx, frame.edge_x != bex
        self._layers[schema.NODE_TAMPER][r] = node
        self._layers[schema.EDGE_TAMPER][r] = edge

    def flush(self) -> None:
        """Write the staged frames to the file and stage the next batch."""
        m = min(self._n, self.dims[0] - self._base)  # frames staged in this batch
        for name, arr in self._layers.items():
            self._sink[name][self._base : self._base + m] = arr[:m]
        self._base += m
        if self._base < self.dims[0]:
            self._clean_batch = self._clean(self._base, self._base + self._n)

    def add_episode(self, record: EpisodeRecord) -> None:
        """Add the next row of episodes/; its index is the `sid` its frames were stored under. The
        record also holds what the fewest-tamper search chose, and a row per target line of an
        overload Am."""
        sid = len(self.episodes)
        self.episodes.append(record.episode)
        if record.tamper is not None:
            self.min_rows.append((sid, record.tamper))
        self.am_rows += [(sid, *row) for row in record.lines]

    # ---- 2. finish: after the walk, everything the file carries besides its frames -----------------

    def finish(
        self,
        columns: dict[str, np.ndarray],
        header: Mapping[str, object],
        knobs: Mapping[str, object],
        record: dict[str, object],
        rec: Placement,
        frac: Sequence[float],
        tamper_rule: str = schema.IEEE_TAMPER_RULE,
    ) -> None:
        """After the walk, write the last batch, the meter masks, the per-frame columns (the grid's own
        `columns` by dataset path), the episodes, the attributes and the temporal features."""
        self.flush()
        self._write_masks()
        self._write_columns(columns)
        self._write_episodes(tamper_rule)
        self._write_attrs(header, knobs, record, rec, columns[schema.SPLIT], frac)
        if self.spec.temporal:
            self.write_temporal_layers(self.f)

    def _write_masks(self) -> None:
        """Write data/node_m and data/edge_m per frame: the static plan repeated, in bounded slabs. Per
        frame, so a later series can change topology mid-file without a layout change."""
        assert self.node_m is not None and self.edge_m is not None and self.i_m is not None
        T = self.dims[0]
        masks = [(schema.NODE_M, self.node_m), (schema.EDGE_M, self.edge_m), (schema.PMU_I_M, self.i_m)]
        for name, m in masks:
            ds = self._create(name, m.shape, np.uint8)
            for a in range(0, T, 4096):
                b = min(a + 4096, T)
                ds[a:b] = np.broadcast_to(m, (b - a, *m.shape))

    def _write_columns(self, columns: Mapping[str, np.ndarray]) -> None:
        """Write the per-frame columns: family, stealthy, seq_id, timestep, and the grid's own `columns`
        by dataset path (the split; on New York also the time and each episode's target branch)."""
        own = {
            schema.FAMILY: self.family.astype(np.int8),
            schema.STEALTHY: np.isin(self.family, sorted(STEALTHY_FAMILIES)).astype(np.uint8),
            schema.SEQ_ID: self.seq_id,
            schema.TIMESTEP: np.arange(self.dims[0], dtype=np.int32),
        }
        for name, arr in {**own, **columns}.items():
            self.f.create_dataset(name, data=arr)

    def _write_episodes(self, tamper_rule: str) -> None:
        """Write episodes/ (one row per episode, buses ragged), the ragged magnitudes of attack/, what
        the fewest-tamper search chose per episode, and one row per target line of an overload Am."""
        eg = self.f.require_group(Group.EPISODES)  # a grid's own column may have made it
        ep, found = self.episodes, self.min_rows
        self._write_table(self.EPISODE_COLUMNS, [(e.onset, e.length, FAMILY_CODE[e.family]) for e in ep])
        self._write_ragged(eg, (schema.EPISODE_BUS_PTR, schema.EPISODE_BUS_IDX), [e.buses for e in ep])
        self._write_ragged(self.f, (schema.MAG_PTR, schema.MAG_BUS), self.mag_bus)
        self.f.create_dataset(schema.MAG, data=np.concatenate(self.mag))  # float32, one list per frame
        if found:  # the fewest-tamper knob is on
            self._write_table(self.MIN_COLUMNS, [(s, r.devices, r.channels, r.evaluated) for s, r in found])
            names = (schema.EPISODE_MIN_SUPPORT_PTR, schema.EPISODE_MIN_SUPPORT_IDX)
            self._write_ragged(eg, names, [r.support for _, r in found])
        if self.am_rows:  # the overload attack
            self._write_table(self.AM_COLUMNS, self.am_rows)
        self.f[Group.ATTACK].attrs[Attr.TAMPER] = tamper_rule

    def _write_table(self, columns: Sequence[tuple[str, type]], rows: Sequence[Iterable[float]]) -> None:
        """Write one dataset of episodes/ per column of `columns` (name, dtype) from `rows`, each in
        the columns' order. A table with no row has empty columns."""
        values = list(zip(*rows)) or [()] * len(columns)
        for (name, dtype), column in zip(columns, values):
            self.f[Group.EPISODES].create_dataset(name, data=np.array(column, dtype))

    @staticmethod
    def _write_ragged(group: h5py.Group, names: tuple[str, str], rows: Sequence[np.ndarray]) -> None:
        """Write variable-length int rows as two datasets of `group`: the pointers [n + 1] into the
        flat values, and the flat values."""
        ptr = np.zeros(len(rows) + 1, np.int64)
        ptr[1:] = np.cumsum([len(r) for r in rows])
        flat = [np.asarray(r, np.int32) for r in rows]
        group.create_dataset(names[0], data=ptr)
        group.create_dataset(names[1], data=np.concatenate(flat) if flat else np.zeros(0, np.int32))

    def _write_attrs(
        self,
        header: Mapping[str, object],
        knobs: Mapping[str, object],
        record: dict[str, object],
        rec: Placement,
        split: np.ndarray,
        frac: Sequence[float],
    ) -> None:
        """Write the attributes in this order: the grid's `header`, the timeline's own, the `knobs`
        (None as -1), `record` as JSON with its hash, and what the placement `rec` asked for and got.
        The placement counts are per split (rows) and family (columns, `placed_families`)."""
        T = self.dims[0]
        sizes = np.bincount(split, minlength=len(Split)).astype(int)
        attacked = np.bincount(split[self.seq_id >= 0], minlength=len(Split)).astype(float)
        in_episodes = sum(e.length for e in self.episodes)
        out = {
            **header,
            Attr.KIND: schema.KIND_TIMELINE,
            Attr.T: T,
            Attr.FAMILIES: ",".join(f"{k}{v}" for k, v in FAMILIES.items()),
            Attr.ATTACKED_FRAC: float(self.attacked / max(1, T)),
            Attr.N_EPISODES: len(self.episodes),
            # frames inside a built episode whose own scan could not be built, stored benign
            Attr.FALLBACK_BENIGN: int(in_episodes) - int((self.seq_id >= 0).sum()),
            **{k: (-1 if v is None else v) for k, v in knobs.items()},
            Attr.SETTINGS: json.dumps(record, sort_keys=True),
            Attr.SETTINGS_HASH: schema.config_hash(record),
            Attr.SPLIT_FRAC: np.asarray(frac, float),
            Attr.SPLIT_SIZES: sizes,
            Attr.SPLIT_ATTACKED_FRAC: attacked / np.maximum(1, sizes),
            Attr.PLACED_FAMILIES: ",".join(rec.families),
            Attr.EPISODES_REQUESTED: rec.requested,
            Attr.EPISODES_BUILT: rec.built,
            Attr.EPISODE_REDRAWS: rec.redraws,
            Attr.EPISODES_DROPPED: rec.dropped,
            Attr.EPISODE_SHORTFALL: rec.requested - rec.built,
        }
        self.f.attrs.update(out)  # an attribute is created where its key first stands in `out`

    def close(self) -> None:
        """Close the file."""
        self.f.close()

    # ---- 3. temporal features: from the observed frames of any timeline file -----------------------

    @staticmethod
    def write_temporal_layers(f: h5py.File, block: int = 2000) -> None:
        """Write `temporal_delta` and `swing` of every frame from the observed injections alone: each
        frame against the previous emitted frame, scaled by `formulas.temporal.recent_change_scale`
        over `schema.SWING_WINDOW_SCANS` frames. Runs in blocks of frames, so memory is bounded."""
        nx = f[schema.NODE_X]
        T = nx.shape[0]
        delta, swing = f[schema.TEMPORAL_DELTA], f[schema.SWING]
        prev = None
        for a in range(0, T, block):
            b = min(a + block, T)
            scale = TimelineWriter._block_scale(nx, a, b)
            rows = np.asarray(nx[a:b], np.float32)
            # each frame's previous scan: the one before it in the block, the first frame's from the
            # last block (the very first frame is its own previous, so its change is zero)
            before = np.concatenate([rows[:1] if prev is None else prev[None], rows[:-1]])
            change = rows[:, :, 1:3] - before[:, :, 1:3]
            delta[a:b] = change.astype(np.float32)
            swing[a:b] = (change / scale).astype(np.float32)
            prev = rows[-1]

    @staticmethod
    def _block_scale(nx: h5py.Dataset, a: int, b: int) -> np.ndarray:
        """The swing scale of frames a..b-1 [b - a, N, 2]. The kernel runs from window + 1 frames
        before a, so every frame's window lies inside the slice."""
        window = schema.SWING_WINDOW_SCANS
        g0 = max(0, a - window - 1)
        pq = np.zeros((b - g0, nx.shape[1], 4), np.float64)  # the kernel reads columns 1:3
        pq[:, :, 1:3] = nx[g0:b, :, 1:3]
        return recent_change_scale(pq, window, nx.shape[1])[a - g0 :]
