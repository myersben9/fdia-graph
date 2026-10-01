"""The write stage of a timeline: its datasets, columns, episodes and attributes in the HDF5 file.

The per-frame layers are created empty at full length before the walk and filled in batches as it
passes (`generation.emit.TimelineBuffers`); after the walk come the meter masks, the small per-frame
columns, the episode tables, the ragged magnitudes, the attributes and the temporal features.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import h5py
import numpy as np

from .. import schema
from ..dataset.base import FAMILIES, STEALTHY_FAMILIES
from ..engine import FdiaGenerator
from ..formulas.temporal import SWING_WINDOW, recent_change_scale, swing_zscore, temporal_delta
from ..models.config import OverloadSettings, TimelineSettings
from ..models.frames import MinimizerResult
from ..schema import Attr
from . import _CHUNK_ROWS, _base_attrs
from .emit import TimelineBuffers, layers
from .plan import _Placement, split_column

KIND = schema.KIND_TIMELINE  # the file attribute that tells a timeline from a shard


def ragged(rows: Sequence[np.ndarray], dtype) -> tuple[np.ndarray, np.ndarray]:
    """A list of variable-length rows as (ptr [n+1], flat values)."""
    ptr = np.zeros(len(rows) + 1, np.int64)
    ptr[1:] = np.cumsum([len(r) for r in rows])
    flat = np.concatenate([np.asarray(r, dtype) for r in rows]) if rows else np.zeros(0, dtype)
    return ptr, flat


def create_layers(f: h5py.File, T: int, C: int, E: int, currents: bool = False) -> dict[str, h5py.Dataset]:
    """The per-frame datasets at full length, chunked along the frame axis and gzipped, empty
    until the walk flushes into them (the PMU current layers only when the plan reads currents)."""
    for group in (schema.Group.DATA, schema.Group.BENIGN, schema.Group.CLEAN, schema.Group.ATTACK):
        f.create_group(group)
    sink = {}
    for name, (shape, dtype) in layers(currents).items():
        trailing = shape(C, E)
        sink[name] = f.create_dataset(
            name,
            shape=(T, *trailing),
            dtype=dtype,
            chunks=(min(_CHUNK_ROWS, T), *trailing),
            compression="gzip",
            compression_opts=4,
        )
    return sink


def _write_masks(f: h5py.File, buf: TimelineBuffers, T: int) -> None:
    """data/node_m and data/edge_m per frame (the static plan repeated, written in bounded slabs
    so a future N-1 series can switch topology mid-file without a layout change)."""
    assert buf.node_m is not None and buf.edge_m is not None
    masks = [(schema.NODE_M, buf.node_m), (schema.EDGE_M, buf.edge_m)]
    if buf.i_m is not None:  # hybrid meters: the PMU current channels read at each branch end
        masks.append((schema.PMU_I_M, buf.i_m))
    for name, m in masks:
        ds = f.create_dataset(
            name,
            shape=(T, *m.shape),
            dtype=np.uint8,
            chunks=(min(_CHUNK_ROWS, T), *m.shape),
            compression="gzip",
        )
        for a in range(0, T, 4096):
            ds[a : min(a + 4096, T)] = np.broadcast_to(m, (min(a + 4096, T) - a, *m.shape))


def _write_episodes(f: h5py.File, buf: TimelineBuffers) -> None:
    """episodes/ (one row per episode, buses ragged) and the ragged magnitudes of attack/."""
    eg = f.create_group(schema.Group.EPISODES)
    ep = buf.episodes
    eg.create_dataset(schema.EPISODE_ONSET, data=np.array([e["onset"] for e in ep], np.int32))
    eg.create_dataset(schema.EPISODE_LENGTH, data=np.array([e["length"] for e in ep], np.int32))
    eg.create_dataset(schema.EPISODE_FAMILY, data=np.array([e["family"] for e in ep], np.int8))
    ptr, idx = ragged([np.asarray(e["buses"]) for e in ep], np.int32)
    eg.create_dataset(schema.EPISODE_BUS_PTR, data=ptr)
    eg.create_dataset(schema.EPISODE_BUS_IDX, data=idx)
    ptr, bus = ragged(buf.mag_bus, np.int32)
    _, mag = ragged(buf.mag, np.float32)
    f.create_dataset(schema.MAG_PTR, data=ptr)
    f.create_dataset(schema.MAG_BUS, data=bus)
    f.create_dataset(schema.MAG, data=mag)
    if buf.min_rows:  # the fewest-tamper knob: what the search chose per episode and whether it is proven
        _write_min_rows(eg, buf.min_rows)
    if buf.am_rows:  # the overload attack: its target branch, rating and the flow it reached
        _write_am_rows(eg, buf.am_rows)
    f[schema.Group.ATTACK].attrs[schema.Attr.TAMPER] = (
        "1 where the attacker wrote the meter: the meters the local false state moves for the "
        "stealthy families, the changed channels for Ad/As/Ar"
    )


def _write_min_rows(eg: h5py.Group, min_rows: list[tuple[int, MinimizerResult]]) -> None:
    """episodes/min_*: one row per episode the fewest-tamper search ran on."""
    rows = [r for _, r in min_rows]
    eg.create_dataset(schema.EPISODE_MIN_EPISODE, data=np.array([s for s, _ in min_rows], np.int32))
    eg.create_dataset(schema.EPISODE_MIN_DEVICES, data=np.array([r.devices for r in rows], np.int32))
    eg.create_dataset(schema.EPISODE_MIN_CHANNELS, data=np.array([r.channels for r in rows], np.int32))
    eg.create_dataset(schema.EPISODE_MIN_PROVEN, data=np.array([r.proven for r in rows], np.uint8))
    eg.create_dataset(schema.EPISODE_MIN_EVALUATED, data=np.array([r.evaluated for r in rows], np.int32))
    eg.create_dataset(schema.EPISODE_MIN_LOWER, data=np.array([r.lower_bound for r in rows], np.int32))
    eg.create_dataset(schema.EPISODE_MIN_UNSOLVED, data=np.array([r.unsolved for r in rows], np.int32))
    ptr, idx = ragged([np.asarray(r.support) for r in rows], np.int32)
    eg.create_dataset(schema.EPISODE_MIN_SUPPORT_PTR, data=ptr)
    eg.create_dataset(schema.EPISODE_MIN_SUPPORT_IDX, data=idx)


def _write_am_rows(eg: h5py.Group, am_rows: list[tuple[int, int, float, float, float, float]]) -> None:
    """episodes/am_*: one row per target line of each overload Am episode (two rows for a two-line
    episode, D17), the episode, the target branch, its rating (MVA), the goal at the window's end,
    the noiseless apparent flow the last frame reached and the emitted (noisy) one."""
    names = (
        (schema.EPISODE_AM_EPISODE, np.int32),
        (schema.EPISODE_AM_LINE, np.int32),
        (schema.EPISODE_AM_RATING, np.float32),
        (schema.EPISODE_AM_TARGET, np.float32),
        (schema.EPISODE_AM_REACHED, np.float32),
        (schema.EPISODE_AM_EMITTED, np.float32),
    )
    for (name, dtype), column in zip(names, zip(*am_rows)):
        eg.create_dataset(name, data=np.array(column, dtype))


def placement_attrs(
    rec: _Placement, bounds: list[tuple[int, int]], frac: Sequence[float], seq_id: np.ndarray
) -> dict[str, object]:
    """What the split-first placement asked for and got: the fractions and sizes of the splits, the
    attacked fraction each reached, and per split (rows) and family (columns, `placed_families`) the
    episodes requested, built, moved for want of a feasible design, dropped because they did not
    fit, and short of the request (dropped or given up)."""
    sizes = np.array([b - a for a, b in bounds], int)
    attacked = np.array([int((seq_id[a:b] >= 0).sum()) for a, b in bounds], float)
    return {
        Attr.SPLIT_FRAC: np.asarray(frac, float),
        Attr.SPLIT_SIZES: sizes,
        Attr.SPLIT_ATTACKED_FRAC: attacked / np.maximum(1, sizes),
        Attr.PLACED_FAMILIES: ",".join(FAMILIES[f] for f in rec.families),
        Attr.EPISODES_REQUESTED: rec.requested,
        Attr.EPISODES_BUILT: rec.built,
        Attr.EPISODE_REDRAWS: rec.redraws,
        Attr.EPISODES_DROPPED: rec.dropped,
        Attr.EPISODE_SHORTFALL: rec.requested - rec.built,
        Attr.JITTER_KEYED: 1,
    }


def _timeline_attrs(
    g: FdiaGenerator,
    T: int,
    seed: int,
    buf: TimelineBuffers,
    knobs: dict[str, Any],  # the recorded knobs, each its own type, written as file attributes
) -> dict[str, object]:
    """The attributes every file carries (dims, units, provenance) plus what makes this one a timeline."""
    attrs: dict[str, object] = dict(_base_attrs(g, T, seed))
    in_episodes = sum(e["length"] for e in buf.episodes)
    attrs.update(
        {
            Attr.KIND: KIND,
            Attr.T: T,
            Attr.FAMILIES: ",".join(f"{k}{v}" for k, v in FAMILIES.items()),
            Attr.ATTACKED_FRAC: float(buf.attacked / max(1, T)),
            Attr.N_EPISODES: len(buf.episodes),
            # frames inside a built episode whose own scan could not be built, stored benign
            Attr.FALLBACK_BENIGN: int(in_episodes) - int((buf.seq_id >= 0).sum()),
        }
    )
    attrs.update({k: (-1 if v is None else v) for k, v in knobs.items()})
    return attrs


def finish_timeline(
    f: h5py.File,
    g: FdiaGenerator,
    buf: TimelineBuffers,
    placed: tuple[_Placement, list[tuple[int, int]], Sequence[float]],
    run: tuple[int, dict],
) -> None:
    """After the walk: the last batch, the masks, the small per-frame columns (family, stealthy,
    seq_id, timestep, the split), the episodes, the ragged magnitudes and the attributes (`run` =
    the seed and the recorded knobs)."""
    seed, knobs = run
    buf.flush()
    T = buf.T
    _write_masks(f, buf, T)
    for name, arr in (
        (schema.FAMILY, buf.family.astype(np.int8)),
        (schema.STEALTHY, np.isin(buf.family, sorted(STEALTHY_FAMILIES)).astype(np.uint8)),
        (schema.SEQ_ID, buf.seq_id),
        (schema.TIMESTEP, np.arange(T, dtype=np.int32)),
        (schema.SPLIT, split_column(placed[1], T)),
    ):
        f.create_dataset(name, data=arr)
    _write_episodes(f, buf)
    f.attrs.update(_timeline_attrs(g, T, seed, buf, knobs))
    f.attrs.update(placement_attrs(*placed, buf.seq_id))


def write_temporal_layers(f: h5py.File, block: int = 2000) -> None:
    """`temporal_delta` and `swing` of every frame, from the observed injections alone: each frame
    against the previous emitted frame, the swing scale from the observed changes of the frames
    before it (formulas.temporal.recent_change_scale over SWING_WINDOW frames). Nothing but the
    measurements enters, so a detector reading these features at test time sees only what an operator
    sees. Called after the walk and by trust.secured_copy after it pins meters. Runs in blocks of
    frames with bounded memory: each block's scale comes from the kernel over the block and the
    SWING_WINDOW + 1 frames before it, which covers every window the block's frames use."""
    nx = f[schema.NODE_X]
    T, N = nx.shape[0], nx.shape[1]
    every = np.ones(N, bool)
    delta, swing = f[schema.TEMPORAL_DELTA], f[schema.SWING]
    prev = None
    for a in range(0, T, block):
        b = min(a + block, T)
        scale = _block_scale(nx, a, b)
        rows = np.asarray(nx[a:b], np.float32)
        d = np.zeros(rows.shape[:2] + (2,), np.float32)
        s = np.zeros_like(d)
        for j in range(len(rows)):
            before = rows[j] if prev is None else prev
            d[j] = temporal_delta(rows[j], before, every)
            s[j] = swing_zscore(rows[j], before, scale[j], every)
            prev = rows[j]
        delta[a:b], swing[a:b] = d, s


def _block_scale(nx: h5py.Dataset, a: int, b: int) -> np.ndarray:
    """The swing scale of frames a..b-1 [b - a, N, 2]: the kernel run over the frames from
    SWING_WINDOW + 1 before a to b-1, so every frame's window lies inside the slice."""
    g0 = max(0, a - SWING_WINDOW - 1)
    pq = np.zeros((b - g0, nx.shape[1], 4), np.float64)  # the kernel reads columns 1:3
    pq[:, :, 1:3] = nx[g0:b, :, 1:3]
    return recent_change_scale(pq, SWING_WINDOW, nx.shape[1])[a - g0 :]


def search_attrs(
    s: TimelineSettings, overload: Optional[OverloadSettings], currents: bool
) -> dict[str, object]:
    """The attributes of the searches a walk ran and of the meters it read, written only when they
    apply: the fewest-tamper knobs, the Am attack, the stealth scale of At's search (Am has no stealth
    bound, [D11]; an Am-only file records the scale it was given, which nothing used), and on
    a hybrid-meter file its meter model and the legend of its current layers; the overload attack's
    rating source and margin [D15]."""
    out: dict[str, object] = {}
    if s.search.min_tamper:
        out.update({Attr.MIN_TAMPER: 1, Attr.MIN_BUDGET: s.search.min_budget})
    if overload is not None:
        out[Attr.AM_ATTACK] = s.am_attack
        out.update(
            {
                Attr.RATING_SOURCE: overload.rating_source,
                Attr.RATING_MARGIN: overload.rating_margin,
                Attr.LOAD_CAP: overload.load_cap,
                Attr.N_LINES: overload.n_lines,
                Attr.SUPPORT_METHOD: overload.support_method,
            }
        )
    if s.search.min_tamper or overload is not None:
        out[Attr.STEALTH_SCALE] = s.ramp.stealth_scale
    if currents:  # a hybrid-meter file, which reads PMU currents
        out.update(
            {
                Attr.METER_MODEL: s.meters.meter_model,
                Attr.CURRENT_FEAT: "Re_I_from,Im_I_from,Re_I_to,Im_I_to",
                Attr.CURRENT_UNITS: "pu on the base current",
            }
        )
    return out
