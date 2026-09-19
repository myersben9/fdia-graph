"""generate(system, name, **knobs) — build a custom dataset and register it as `name`.

Since 0.18 `generate` writes one timeline file (`fdia_graph.timeline`): every frame of the
operating-point pool scanned in time order with attack episodes of every family, which
`fg.load(name)` reads as a record table (`order="random"`) or as the timeline (`order="time"`).
The knobs are those of `timeline.generate_timeline` plus `frames` (how many pool timesteps to
walk, default the whole pool). `generate_shard` is the pre-0.18 record-shard writer, kept for one
minor version; its knobs are listed on the function.

Shared by both:
  states            source of operating points: path to a pool .npz or .h5 (key 'X' [T,N,4]) or an init dir of
                    X_*.npy; if None, uses $FDIA_GRAPH_INIT or downloads the system's operating-point pool.
  out               output .h5 path (default under the cache dir)
"""

from __future__ import annotations

import glob
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import h5py
import numpy as np

# FdiaGenerator = physics/attack math; FAM_ID = family name -> integer id; attack_frame = one scan.
from .engine import FAM_ID, FdiaGenerator
from .engine.records import AM_FAMILY, RAMP_FAMILY, SINGLE_SHOT_ORDER, FrameKnobs, attack_frame
from .formulas.attacks import ramp_profile
from .formulas.temporal import recent_change_scale, swing_zscore, temporal_delta
from .models.data import ShardArrays  # noqa: F401  re-exported: defined here before the models package

# CACHE_DIR = on-disk shard home; register_local makes the new dataset findable by load(name).
from .models.frames import Record  # noqa: F401  re-exported: defined here before the models package
from .models.grid import NODE
from .registry import CACHE_DIR, register_local

# Single-shot family name -> id (Aq=1, Ad=2, As=3, Ar=4, Al/LRA=6).
_SINGLE = {"Aq": 1, "Ad": 2, "As": 3, "Ar": 4, "Al": 6}
# Corrupt-in-place families: id -> letter code passed to g.corrupt().
_FAMK = {2: "Ad", 3: "As", 4: "Ar"}
# Swing-feature lookback (scans). Tuned: rate-of-change catch-rate plateaus ~60 scans; ramp At stays near
# the benign floor at every window, so At remains the ML-only family.
SWING_W = 60

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
    # fall back to the downloadable operating-point pool for this system
    from .download import ensure_local
    from .registry import _RELEASE, AssetSpec, system_id

    # Built-in release asset spec (pool_ieee{C}.npz); follows the pinned shard release, which carries
    # all 8 ladder pools (a hardcoded old tag here 404'd generate() on systems added after that tag).
    C = system_id(system)
    spec = AssetSpec(
        "builtin", f"pool{C}", file=f"pool_ieee{C}.npz", release=_RELEASE, repo="myersben9/fdia-graph"
    )
    # ensure_local downloads/caches and returns the local path.
    return np.load(ensure_local(spec))["X"].astype(np.float64)


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


def _record_features(nx: np.ndarray, nm: np.ndarray, prev: np.ndarray, scale_t: np.ndarray, C: int):
    """The two temporal features of a record (formulas.temporal), at injection-metered buses."""
    metered = nm[:, NODE.p_inj] > 0
    return temporal_delta(nx, prev, metered), swing_zscore(nx, prev, scale_t, metered)


def _make_record(ctx: _FrameContext, t: int, family: int, sid: int, atk: Any) -> Optional[Record]:
    """One record of `family` on pool timestep t through the shared per-frame physics
    (engine.records), finished with its temporal features. sid = ramp sequence id (-1 otherwise),
    atk = (load-table positions, multiplier). None when the scan was rejected."""
    targets, mult = atk if atk is not None else (None, None)
    frame = attack_frame(ctx.g, ctx.X[t], family, targets, mult, ctx.knobs)
    if frame is None:
        return None
    prev = ctx.X[t - 1] if t > 0 else ctx.X[t]  # same [|V|, Pinj, Qinj, theta] columns as node_x
    td, sw = _record_features(frame.node_x, frame.node_m, prev, ctx.scale[t], ctx.g.C)
    if len(frame.mag):
        # Designed per-bus magnitude next to the realized swing, for the plausibility-band sidecar.
        mf = np.zeros(ctx.g.C)
        mf[frame.mag_bus] = frame.mag
        yb = frame.y.astype(bool)
        ctx.mag_log.append((family, mf[yb], np.abs(sw[yb]).max(1)))
    return Record(
        frame.node_x,
        frame.node_m,
        frame.edge_x,
        frame.edge_m,
        frame.y,
        family,
        sid,
        t,
        0,
        frame.stealthy,
        td,
        sw,
    )


def _draw_targets(
    rng: np.random.Generator, apos: np.ndarray, fam: int, intensity: float, p: Optional[np.ndarray]
):
    """The attacked load-table positions and load multipliers for one single-shot attempt.
    Aq: a footprint of 1 to 6 buses, each with its own multiplier in 1.05 .. 1 + intensity.
    Ad, As, Ar, Al: up to 4 buses; their frames ignore the multiplier (corrupt in place / LRA delta)."""
    nab = len(apos)
    if fam == 1:
        k = int(rng.integers(1, min(6, nab) + 1))
        a = rng.choice(apos, k, replace=False, p=p)
        return a, 1 + rng.uniform(0.05, intensity, size=k)
    a = rng.choice(apos, min(4, nab), replace=False, p=p)
    return a, 1 + rng.uniform(0.05, intensity)


def _draw_benign(ctx: _FrameContext, rng: np.random.Generator, n_benign: int) -> list[Record]:
    """n_benign records on distinct pool timesteps, in draw order."""
    nT = len(ctx.X)
    recs = []
    for t in rng.choice(nT, min(n_benign, nT), replace=False):
        r = _make_record(ctx, int(t), 0, -1, None)
        if r is not None:
            recs.append(r)
    return recs


def _draw_single_shot(
    ctx: _FrameContext, rng: np.random.Generator, fam: int, per_family: int, p: Optional[np.ndarray]
):
    """Draw until per_family records of `fam` succeed, capped at per_family * 25 attempts so an
    infeasible configuration cannot loop forever. Returns (records, (attempts, accepted)): the
    attempts count is the honest measure of how hard the topology is to attack."""
    apos = ctx.g.attackable_pos  # targets come only from attackable positions (real active load)
    nT = len(ctx.X)
    recs: list[Record] = []
    got = tries = 0
    while got < per_family and tries < per_family * 25:
        tries += 1
        t = int(rng.integers(nT))
        r = _make_record(ctx, t, fam, -1, _draw_targets(rng, apos, fam, ctx.knobs.intensity, p))
        if r is not None:
            recs.append(r)
            got += 1  # count only converged records
    return recs, (tries, got)


def _ramp_sequence(
    ctx: _FrameContext, rng: np.random.Generator, sid: int, ramp_len: int, ramp_rate: float, p
):
    """One ramp sequence: a fixed bus set ramps up or down at rate_up to a peak, optionally holds,
    then returns at an independent rate_down; each sequence is self-contained. Stops at the first
    non-converging step. Returns (records, steps solved)."""
    nT = len(ctx.X)
    apos = ctx.g.attackable_pos
    t0 = int(rng.integers(nT - ramp_len))  # start leaving room for the full sequence
    atk = rng.choice(apos, min(5, len(apos)), replace=False, p=p)  # fixed bus set
    direction = 1.0 if rng.random() < 0.5 else -1.0  # +1 = surge first, -1 = dip first
    rate_up = ramp_rate * rng.uniform(0.7, 1.3)
    rate_down = ramp_rate * rng.uniform(0.7, 1.3)  # independent slopes
    rise_len = max(1, int(rng.uniform(0.20, 0.45) * ramp_len))  # steps ramping to the peak/trough
    hold_len = int(rng.uniform(0.0, 0.25) * ramp_len)  # steps held at the peak (0 = no plateau)
    seq: list[Record] = []
    steps = 0
    for i in range(ramp_len):
        steps += 1
        dev = ramp_profile(i, rise_len, hold_len, rate_up, rate_down)
        r = _make_record(ctx, t0 + i, RAMP_FAMILY, sid, (atk, 1 + direction * dev))  # multiplier = 1 +/- dev
        if r is None:
            break  # abort the sequence on the first non-converging step
        seq.append(r)
    return seq, steps


def _draw_ramps(
    ctx: _FrameContext, rng: np.random.Generator, per_family: int, ramp_len: int, ramp_rate: float, p
):
    """Ramp sequences until about per_family records; a sequence counts only if at least 10 steps
    solved. Returns (records, (steps solved, accepted)): attempts minus accepted is what was thrown away."""
    recs: list[Record] = []
    got = steps = sid = 0
    while got < per_family:
        seq, solved = _ramp_sequence(ctx, rng, sid, ramp_len, ramp_rate, p)
        steps += solved
        if len(seq) >= 10:
            recs.extend(seq)
            got += len(seq)
            sid += 1  # the next accepted sequence gets the next id
    return recs, (steps, got)


def _draw_families(
    ctx: _FrameContext,
    rng: np.random.Generator,
    fam_ids: list[int],
    per_family: int,
    ramp: tuple[int, float],
    p,
):
    """Every attacked family in the fixed draw order (single-shot families 1, 2, 3, 4, 6, then the
    ramp), so the RNG sequence, and with it the shard, stays identical release to release.
    Returns (records, {family: (attempts, accepted)})."""
    recs: list[Record] = []
    yield_: dict[int, tuple[int, int]] = {}
    for fam in [k for k in SINGLE_SHOT_ORDER if k in fam_ids]:
        got, yield_[fam] = _draw_single_shot(ctx, rng, fam, per_family, p)
        recs += got
    if RAMP_FAMILY in fam_ids:
        got, yield_[RAMP_FAMILY] = _draw_ramps(ctx, rng, per_family, ramp[0], ramp[1], p)
        recs += got
    return recs, yield_


def _write_magnitude_sidecar(
    out: str, mag_log: list[tuple[int, np.ndarray, np.ndarray]], floor: float, cap: float
) -> None:
    """<out>.mag.npz: (family, designed magnitude, realized swing) per attacked bus, so the
    plausibility band can be verified against what the gate enforced."""
    if not mag_log:
        return
    fam_col = np.concatenate([np.full(len(m), fid, np.int8) for fid, m, s in mag_log])
    mag_col = np.concatenate([np.asarray(m, float) for _, m, s in mag_log])
    sw_col = np.concatenate([np.asarray(s, float) for _, m, s in mag_log])
    np.savez(out + ".mag.npz", family=fam_col, mag=mag_col, swing=sw_col, floor=floor, cap=cap)


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
    ramp_rate, ramp_len, am_len, am_rate, am_sigma, am_direction, corrupt_len, replay_tau,
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
    register_local(name, path, meta=dict(system=system, kind="timeline", frames=len(X), seed=seed, **knobs))
    return path


def generate_shard(
    system: Union[int, str],
    name: str,
    per_family: int = 3000,
    families: Sequence[str] = ("Aq", "Ad", "As", "Ar", "At", "Al"),
    attack_intensity: float = 0.20,
    ramp_rate: float = 0.002,
    ramp_len: int = 60,
    replay_tau: Optional[int] = None,
    n_benign: int = 20000,
    lra_targets: int = 15,
    redundancy: Optional[dict] = None,
    split: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 123,
    states: Optional[Union[str, np.ndarray]] = None,
    out: Optional[str] = None,
    outage: Optional[Union[int, str]] = None,
    targeting: str = "uniform",
    targeting_strength: float = 1.5,
) -> str:
    """Build a pre-0.18 record shard and register it as `name`; returns the .h5 path. Retires in 0.19.

    per_family        attacked records per family (default 3000); n_benign benign records (default 20000)
    families          which attacks to include (default all six; Am is timeline-only)
    attack_intensity  per-bus load shift bound of Aq / Al and the upper plausibility cap (default 0.20)
    ramp_rate, ramp_len   the At ramp; replay_tau the Ar/As replay depth (None = random lag >= 20)
    redundancy        meter coverage {vbus_frac, pmu_frac, flow_frac} (default 0.6/0.2/0.9)
    split             chronological train/val/test fractions by record (default (0.6, 0.2, 0.2))
    targeting: how attacked-bus SETS are drawn. "uniform" (default) = uniform over attackable load buses,
    byte-identical to prior releases. "centrality" tilts toward structurally critical buses (fused
    degree/closeness/betweenness; Doostinia et al., IEEE TIA 2025), more realistic and more damaging;
    targeting_strength is the exponential tilt (0 == uniform) and applies to every family. Physics unchanged.
    lra_targets: size of the LRA target-line pool (each attack picks one, so bus sets vary).
    outage: line to take OUT OF SERVICE for the whole shard (None = intact). One shard per topology, so the
    graph/ group always describes its own topology; `states` must then be a pool re-solved under that same
    topology (FdiaGenerator.resolve_states), or benign records carry intact-network voltages.
    """
    # Meter coverage defaults (60% V buses, 20% PMU, 90% flows) with caller overrides merged on top.
    red = {"vbus_frac": 0.6, "pmu_frac": 0.2, "flow_frac": 0.9, **(redundancy or {})}
    # The outage is applied in the constructor before Ybus/base state and consumes no randomness, so the
    # meter plan and per-meter biases are identical across topologies at a given seed.
    g = FdiaGenerator(system, seed=seed, outage=outage, **red)
    # Candidate LRA target lines (bound by intensity, up to min(6, #load) buses, from lra_targets lines).
    lra_k = min(6, len(g.load_bus))
    g._pick_lra_target(attack_intensity, lra_k, n_targets=lra_targets)
    rng = g.rng  # every draw below comes from the generator's seeded RNG, in this order
    cent_p = g.centrality_probs(targeting_strength) if targeting == "centrality" else None
    X = _load_states(system, states)  # operating-point pool [T, N, 4], order-normalized
    knobs = FrameKnobs(
        attack_intensity, NOISE_FLOOR, lra_k, replay_tau, reject_below_floor=True, with_benign=False
    )
    ctx = _FrameContext(g, X, _swing_scale(X, g.C), knobs, [])

    fam_ids = [FAM_ID[f] for f in families]
    if AM_FAMILY in fam_ids:
        raise ValueError("Am is a timeline family: fdia_graph.timeline.generate_timeline builds it")
    recs = _draw_benign(ctx, rng, n_benign)
    attacked, yield_ = _draw_families(ctx, rng, fam_ids, per_family, (ramp_len, ramp_rate), cent_p)
    recs += attacked

    out = out or os.path.join(CACHE_DIR, f"{name}.h5")
    tried = {k: v[0] for k, v in yield_.items()}
    taken = {k: v[1] for k, v in yield_.items()}
    _write(g, recs, out, split, seed, solve_stats=(tried, taken), pool=X)
    _write_magnitude_sidecar(out, ctx.mag_log, NOISE_FLOOR, attack_intensity)
    meta = dict(
        system=system,
        per_family=per_family,
        families=list(families),
        attack_intensity=attack_intensity,
        ramp_rate=ramp_rate,
        seed=seed,
        outage_line=g.contingency.line if g.contingency.line is not None else -1,
    )
    register_local(name, out, meta=meta)
    return out


_CHUNK_ROWS = 128  # per-record datasets are chunked along the record axis for efficient partial reads


_RECORD_ARRAYS = ("node_x", "node_m", "edge_x", "edge_m", "y", "temporal_delta", "swing")
_RECORD_SCALARS = (
    ("family", np.int8),
    ("seq_id", np.int32),
    ("timestep", np.int32),
    ("gap", np.uint8),
    ("stealthy", np.uint8),
)


def _stack_records(recs: list[Record]) -> ShardArrays:
    """The record tuples as [T, ...] arrays, scalar fields with dtypes sized to their range."""
    stacked = {name: np.stack([getattr(r, name) for r in recs]) for name in _RECORD_ARRAYS}
    scalars = {name: np.array([getattr(r, name) for r in recs], dtype) for name, dtype in _RECORD_SCALARS}
    return ShardArrays(**stacked, **scalars)


def _shard_attrs(
    g: FdiaGenerator, n_records: int, seed: int, solve_stats: Optional[tuple[dict, dict]]
) -> dict[str, Any]:
    """File attributes: dims, feature legends, units, the family legend, topology provenance, yield.

    Units are ENGINEERING quantities (node_x = [V pu, P_inj MW, Q_inj MVAr, theta deg], edge flows
    MW/MVAr); baseMVA lets the loader also serve a per-unit view. Topology: "base" = intact,
    "n1_line" = one line out for every record, with the contingency's size (outage_base_flow_mw) so
    a shard is self-describing. solve_yield = "famid:attempts/accepted" per family, the drop rate.
    """
    attrs = dict(
        system=g.C,
        N=g.C,
        E=g.E,
        n_records=n_records,
        node_feat="V,P_inj,Q_inj,theta",
        edge_feat="P_from,Q_from",
        node_units="V:pu,P_inj:MW,Q_inj:MVAr,theta:deg",
        edge_units="P_from:MW,Q_from:MVAr",
        baseMVA=float(g.base.sn_mva),
        families="0benign,1Aq,2Ad,3As,4Ar,5At,6Al",
        lra_target_line=g._Ltgt,
        seed=seed,
        topology=("base" if g.contingency.line is None else "n1_line"),
        outage_line=(-1 if g.contingency.line is None else int(g.contingency.line)),
        outage_branch_pos=int(g.contingency.pos),
        outage_line_name=g.contingency.name,
        outage_from_bus=int(g.contingency.from_bus),
        outage_to_bus=int(g.contingency.to_bus),
        outage_base_flow_mw=float(g.contingency.base_flow_mw),
    )
    if solve_stats is not None:
        tried, taken = solve_stats
        attrs["solve_yield"] = ",".join(f"{k}:{tried[k]}/{taken[k]}" for k in sorted(tried))
    return attrs


def _write_graph(f: Any, g: FdiaGenerator) -> None:
    """graph/ group: the static topology shared by all records, including the full per-unit branch
    physics and bus shunts that reconstruct Ybus exactly (verified against makeYbus to 7e-15, 3e-14
    and 5e-13 on IEEE 14, 118 and 300), so a model reads exactly the estimator's physics."""
    gg = f.create_group("graph")
    gg.create_dataset("edge_index", data=g.ei)
    # DEPRECATED, unit-inconsistent (ohms for lines, vk percent for trafos). Kept for v0.4.x readers.
    gg.create_dataset("edge_reactance", data=g.x_react)
    br = g.branch
    for name, data in (
        ("edge_r", br.r),
        ("edge_x", br.x),
        ("edge_b", br.b),
        ("edge_g", br.g),
        ("edge_gs", g.edge_gs),
        ("edge_bs", g.edge_bs),
        ("edge_tap", br.tap),
        ("edge_shift", br.shift_deg),
        ("edge_status", br.status),
        ("edge_is_trafo", g.edge_is_trafo),
        ("bus_shunt_g", g.bus_shunt_g),
        ("bus_shunt_b", g.bus_shunt_b),
    ):
        gg.create_dataset(name, data=data)
    gg.attrs.update(
        dict(
            edge_feat_static="r,x,b,g,tap,shift,status,is_trafo (per unit, ppc order = lines then trafos)",
            bus_feat_static="shunt_g,shunt_b (MW/MVAr at 1.0 pu, ppc bus order)",
            edge_reactance_deprecated="mixes ohms (lines) with vk_percent (trafos); use edge_x",
            ybus_reconstructible="yes, see fdia_graph tests: Y = f(edge_r,x,b,g,tap,shift,status)+bus shunts",
        )
    )


def _chunked(group: Any, name: str, data: np.ndarray) -> None:
    """A per-record dataset chunked along the record axis (at most _CHUNK_ROWS rows) and gzipped."""
    ch = (min(_CHUNK_ROWS, len(data)),) + data.shape[1:]
    group.create_dataset(name, data=data, chunks=ch, compression="gzip", compression_opts=4)


def _write_data(f: Any, arrays: ShardArrays, split_code: np.ndarray) -> None:
    """data/ group: the per-record tensors and the scalar fields (including the split)."""
    d = f.create_group("data")
    for name in ("node_x", "edge_x", "temporal_delta", "swing"):
        _chunked(d, name, arrays[name])
    for name in ("node_m", "edge_m", "y"):
        d.create_dataset(name, data=arrays[name], compression="gzip")
    for name in ("family", "seq_id", "timestep", "gap", "stealthy"):
        d.create_dataset(name, data=arrays[name])
    d.create_dataset("split", data=split_code)


def _write_clean(f: Any, g: FdiaGenerator, pool: np.ndarray, timestep: np.ndarray) -> None:
    """clean/ group (v0.7.2+): the NOISELESS attack-free truth per POOL timestep (the SE target),
    resolved per record via data/timestep; the same layer the streams ship. node_clean is the pool
    itself (already in node_x column order); edge_clean = exact Ybus flows, unmetered zeroed."""
    Xp = np.asarray(pool, np.float64)[: int(timestep.max()) + 1]
    cg = f.create_group("clean")
    _chunked(cg, "node_clean", Xp.astype(np.float32))  # [Tpool, N, 4]
    _chunked(cg, "edge_clean", g.clean_flows_from_states(Xp))  # [Tpool, E, 2]


def _write(
    g: FdiaGenerator,
    recs: list[Record],
    out: str,
    split: tuple[float, float, float],
    seed: int,
    solve_stats: Optional[tuple[dict, dict]] = None,
    pool: Optional[np.ndarray] = None,
) -> None:
    """Serialize the records to one HDF5 file: attributes, graph/, data/ and (with a pool) clean/.

    pool is the operating-point pool as _load_states returned it, an in-memory [T, N, 4] array
    already in [|V|, Pinj, Qinj, theta] order, never the caller's raw `states` argument.
    """
    arrays = _stack_records(recs)
    split_code = _chrono_split(arrays.timestep, arrays.seq_id, split)  # train/val/test = 0/1/2
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with h5py.File(out, "w") as f:
        f.attrs.update(_shard_attrs(g, len(recs), seed, solve_stats))
        _write_graph(f, g)
        _write_data(f, arrays, split_code)
        if pool is not None:
            _write_clean(f, g, pool, arrays.timestep)


def _chrono_split(tstep: np.ndarray, seq: np.ndarray, frac: tuple[float, float, float]) -> np.ndarray:
    # Assign train(0)/val(1)/test(2) by chronological order, keeping ramp sequences intact (no future leak).
    T = len(tstep)
    # Groups: one per ramp sequence id (seq>=0), plus a singleton per non-sequence record (seq<0).
    groups = [np.where(seq == s)[0] for s in np.unique(seq[seq >= 0])] + [
        np.array([i]) for i in np.where(seq < 0)[0]
    ]
    # Order groups by earliest timestep so the split is chronological.
    groups.sort(key=lambda gp: int(tstep[gp].min()))
    sp = np.empty(T, np.int8)
    c = 0  # sp = split labels; c = records already placed
    for gp in groups:
        # Cumulative fraction c/T picks the bucket: <frac[0] train, <frac[0]+frac[1] val, else test.
        f = c / T
        sp[gp] = 0 if f < frac[0] else (1 if f < frac[0] + frac[1] else 2)
        c += len(gp)
    return sp
