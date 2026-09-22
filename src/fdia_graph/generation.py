"""generate(system, name, **knobs) — build a custom dataset and register it as `name`.

`generate` writes one timeline file (`fdia_graph.timeline`): every frame of the operating-point
pool scanned in time order with attack episodes of every family, which `fg.load(name)` reads as a
record table (`order="random"`) or as the timeline (`order="time"`). The knobs are those of
`timeline.generate_timeline` plus `frames` (how many pool timesteps to walk, default the whole
pool). This module also holds what the writer shares with the engine: the operating-point pool
(`_load_states`, any column order, npz or HDF5), the per-run frame context, the noise floor and
the swing window, and the file attributes and static graph group every file carries.

  states            source of operating points: path to a pool .npz or .h5 (key 'X' [T,N,4]) or an init dir of
                    X_*.npy; if None, uses $FDIA_GRAPH_INIT or downloads the system's operating-point pool.
  out               output .h5 path (default under the cache dir)
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import h5py
import numpy as np

from . import schema
from .engine import FdiaGenerator
from .engine.records import FrameKnobs
from .formulas.temporal import SWING_WINDOW, recent_change_scale
from .models.grid import NODE
from .registry import CACHE_DIR, register_local
from .schema import KIND_TIMELINE, Attr, Group, Static

# Swing-feature lookback (scans). Tuned: rate-of-change catch-rate plateaus ~60 scans; ramp At stays near
# the benign floor at every window, so At remains the ML-only family.
SWING_W = SWING_WINDOW

# Lower edge of the plausibility band: a realized change below this fraction of the meter reading sits inside
# the noise floor (accuracy-class sigma ~1.7%) and resolves to noise, so we reject such draws for spike/meter
# families. Ramp At is DELIBERATELY exempt so its per-scan step can stay sub-floor while the deviation
# accumulates. Upper edge of the band is `attack_intensity` (default 0.20).
NOISE_FLOOR = 0.02


def as_v_first(X: np.ndarray) -> np.ndarray:
    """Return a state pool [T,N,4] in the SDK's one column order [|V|, Pinj, Qinj, theta].

    Pools written before 0.12 (the downloadable pool assets, older `states=` files, the original
    pipeline's X_*.npy) are [Pinj, Qinj, |V|, theta]. The order is detected from the data: the voltage
    column is the only one whose every value sits in a per-unit band around 1.0, so a pool is
    converted when column 2 looks like |V| and column 0 does not. Anything else is an error rather
    than a guess.
    """
    X = np.asarray(X, np.float64)
    if X.ndim != 3 or X.shape[2] != 4:
        raise ValueError(f"a state pool is [T, N, 4], got shape {X.shape}")

    def looks_like_v(col: np.ndarray) -> bool:
        return bool(np.all((col > 0.5) & (col < 1.5)))

    v0, v2 = looks_like_v(X[:, :, NODE.v]), looks_like_v(X[:, :, NODE.q_inj])
    if v0 and not v2:
        return X
    if v2 and not v0:
        return X[:, :, [2, 0, 1, 3]]  # [P, Q, V, th] -> [V, P, Q, th]
    raise ValueError("cannot tell the pool's column order; expected [|V|, Pinj, Qinj, theta]")


def _load_states(
    system: Union[int, str], states: Optional[Union[str, np.ndarray]], pool_cap: int = 8000
) -> np.ndarray:
    """Return an operating-point pool [T,N,4] = [|V|, Pinj, Qinj, theta] to inject attacks onto.

    Priority: explicit `states` -> $FDIA_GRAPH_INIT -> downloadable pool asset. A local init DIRECTORY can
    hold ~86k per-timestep files, so we stride-sample at most `pool_cap` evenly across the timeline (keeps
    daily/seasonal variety) rather than reading every file, which made a naive load take minutes. Every
    source goes through as_v_first, so pools saved in the older [P, Q, V, theta] order still load.
    """
    return as_v_first(_read_states(system, states, pool_cap))


def _read_states(
    system: Union[int, str], states: Optional[Union[str, np.ndarray]], pool_cap: int
) -> np.ndarray:
    # In-memory pool accepted directly (no disk). Checked first because a numpy array has no truth value for `or`.
    if isinstance(states, np.ndarray):
        return states.astype(np.float64)
    # Source: caller arg wins, else FDIA_GRAPH_INIT, else None (downloaded below).
    src = states or os.environ.get("FDIA_GRAPH_INIT")
    if src and os.path.isdir(src):
        # Init directory: all X_*.npy sorted by integer timestep (name "X_<t>.npy").
        xs = sorted(glob.glob(os.path.join(src, "X_*.npy")), key=lambda p: int(os.path.basename(p)[2:-4]))
        stride = max(1, len(xs) // pool_cap)  # even stride -> diverse sample, bounded cost
        # float64 for the WLS/AC solve downstream.
        return np.stack([np.load(f) for f in xs[::stride][:pool_cap]]).astype(np.float64)
    if src and src.endswith(".npz"):
        return np.load(src)["X"].astype(np.float64)  # precomputed compact pool (the SDK default)
    if src and src.endswith((".h5", ".hdf5")):
        with h5py.File(src, "r") as f:  # the pool as HDF5, dataset "X" [T, N, 4]
            return np.asarray(f["X"], np.float64)
    # fall back to the downloadable operating-point pool for this system, at the pinned data release
    # (which carries all eight ladder pools; a hardcoded old tag here once 404'd newer systems)
    from .download import ensure_local
    from .registry import pool_spec

    return _read_states(system, ensure_local(pool_spec(system)), pool_cap)


def _swing_scale(X: np.ndarray, C: int) -> np.ndarray:
    """The swing feature's scale for a pool: formulas.temporal.recent_change_scale over SWING_W scans."""
    return recent_change_scale(X, SWING_W, C)


@dataclass
class _FrameContext:
    """What every record of one generation run shares, passed explicitly to the record functions."""

    g: FdiaGenerator
    X: np.ndarray  # operating-point pool [T, N, 4] in [|V|, Pinj, Qinj, theta] order
    scale: np.ndarray  # swing scale per timestep [T, N, 2], from _swing_scale
    knobs: FrameKnobs  # the attack settings every scan shares
    mag_log: list[tuple[int, np.ndarray, np.ndarray]]  # (family, designed magnitude, swing) per attacked bus


def generate(
    system: Union[int, str],
    name: str,
    frames: Optional[int] = None,
    states: Optional[Union[str, np.ndarray]] = None,
    out: Optional[str] = None,
    seed: int = 123,
    **knobs: Any,
) -> str:
    """Walk one attacked timeline over the operating-point pool of `system`, write it as one HDF5
    file and register it as `name`; returns the path. `frames` caps the pool timesteps walked;
    every other knob is `timeline.generate_timeline`'s (families, attacked_frac, attack_intensity,
    ramp_rate, ramp_len, am_len, am_rate, am_direction, hops, max_load_mw, corrupt_len, replay_tau,
    redundancy, split)."""
    from .timeline import generate_timeline

    if frames is not None and (isinstance(frames, bool) or not isinstance(frames, int) or frames < 1):
        raise ValueError(
            f"frames caps the pool timesteps walked and must be a positive integer, got {frames!r}"
        )
    X = _load_states(system, states)
    if frames is not None:
        X = X[:frames]
    out = out or os.path.join(CACHE_DIR, f"{name}.h5")
    path = generate_timeline(system, states=X, seed=seed, out=out, **knobs)
    register_local(
        name, path, meta=dict(system=system, kind=KIND_TIMELINE, frames=len(X), seed=seed, **knobs)
    )
    return path


_CHUNK_ROWS = 128  # per-frame datasets are chunked along the frame axis for efficient partial reads


def _base_attrs(g: FdiaGenerator, n_records: int, seed: int) -> dict[str, Any]:
    """File attributes every file carries: dims, feature legends, units, topology provenance.

    Units are ENGINEERING quantities (node_x = [V pu, P_inj MW, Q_inj MVAr, theta deg], edge flows
    MW/MVAr); baseMVA lets the loader also serve a per-unit view. Topology: "base" = intact,
    "n1_line" = one line out for every record, with the contingency's size (outage_base_flow_mw) so
    a file is self-describing.
    """
    return {
        Attr.SYSTEM: g.C,
        Attr.N: g.C,
        Attr.E: g.E,
        Attr.N_RECORDS: n_records,
        Attr.NODE_FEAT: "V,P_inj,Q_inj,theta",
        Attr.EDGE_FEAT: "P_from,Q_from",
        Attr.NODE_UNITS: "V:pu,P_inj:MW,Q_inj:MVAr,theta:deg",
        Attr.EDGE_UNITS: "P_from:MW,Q_from:MVAr",
        Attr.BASEMVA: float(g.base.sn_mva),
        Attr.LRA_TARGET_LINE: g._primary_target_line,
        Attr.SEED: seed,
        Attr.TOPOLOGY: ("base" if g.contingency.line is None else "n1_line"),
        Attr.OUTAGE_LINE: (-1 if g.contingency.line is None else int(g.contingency.line)),
        Attr.OUTAGE_BRANCH_POS: int(g.contingency.pos),
        Attr.OUTAGE_LINE_NAME: g.contingency.name,
        Attr.OUTAGE_FROM_BUS: int(g.contingency.from_bus),
        Attr.OUTAGE_TO_BUS: int(g.contingency.to_bus),
        Attr.OUTAGE_BASE_FLOW_MW: float(g.contingency.base_flow_mw),
    }


def _static_physics(g: FdiaGenerator) -> dict[str, Any]:
    """The graph/ datasets this writer fills, keyed by the schema's names (`Static`), so a file
    never carries what the reader's table (`STATIC_PHYSICS`) does not name."""
    br = g.branch
    return {
        Static.EDGE_R: br.r,
        Static.EDGE_X: br.x,
        Static.EDGE_B: br.b,
        Static.EDGE_G: br.g,
        Static.EDGE_GS: g.edge_gs,
        Static.EDGE_BS: g.edge_bs,
        Static.EDGE_TAP: br.tap,
        Static.EDGE_SHIFT: br.shift_deg,
        Static.EDGE_STATUS: br.status,
        Static.EDGE_IS_TRAFO: g.edge_is_trafo,
        Static.BUS_SHUNT_G: g.bus_shunt_g,
        Static.BUS_SHUNT_B: g.bus_shunt_b,
    }


def _write_graph(f: Any, g: FdiaGenerator) -> None:
    """graph/ group: the static topology shared by all frames, including the full per-unit branch
    physics and bus shunts that reconstruct Ybus exactly (verified against makeYbus to 7e-15, 3e-14
    and 5e-13 on IEEE 14, 118 and 300), so a model reads exactly the estimator's physics."""
    gg = f.create_group(Group.GRAPH)
    gg.create_dataset(schema.EDGE_INDEX.split("/")[1], data=g.ei)
    # DEPRECATED, unit-inconsistent (ohms for lines, vk percent for trafos). Kept for v0.4.x readers.
    gg.create_dataset(schema.EDGE_REACTANCE.split("/")[1], data=g.x_react)
    values = _static_physics(g)
    for name in schema.STATIC_PHYSICS:  # the reader's table, in its order; what this writer has of it
        if name in values:
            gg.create_dataset(name, data=values[name])
    gg.attrs.update(
        {
            Attr.EDGE_FEAT_STATIC: "r,x,b,g,tap,shift,status,is_trafo (per unit, ppc order = lines then trafos)",
            Attr.BUS_FEAT_STATIC: "shunt_g,shunt_b (MW/MVAr at 1.0 pu, ppc bus order)",
            Attr.EDGE_REACTANCE_DEPRECATED: "mixes ohms (lines) with vk_percent (trafos); use edge_x",
            Attr.YBUS_RECONSTRUCTIBLE: "yes, see fdia_graph tests: Y = f(edge_r,x,b,g,tap,shift,status)+bus shunts",
        }
    )
