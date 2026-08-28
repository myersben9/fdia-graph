"""generate(system, name, **knobs) — build a custom dataset and register it as `name`.

Research knobs (all optional, sensible defaults matching the published shards):
  per_family        int   attacked records per family (default 3000)
  families          list  which attacks to include (default all: Aq,Ad,As,Ar,At,Al)
  attack_intensity  float per-bus load shift magnitude for Aq / Al(LRA) bound; also the upper plausibility cap (default 0.20 = 20%)
  ramp_rate         float ramp perturbation growth per step (default 0.002)
  ramp_len          int   ramp sequence length (default 60)
  replay_tau        int   Ar/As replay depth in frames back (default None = random lag >=20; set for a fixed lag)
  n_benign          int   benign records (default 20000)
  redundancy        dict  meter coverage: {vbus_frac,pmu_frac,flow_frac} (default 0.6/0.2/0.9)
  split             tuple chronological train/val/test fractions (default (0.6,0.2,0.2))
  seed              int   (default 123)
  states            source of operating points: path to a pool .npz (key 'X' [T,N,4]) or an init dir of
                    X_*.npy; if None, uses $FDIA_GRAPH_INIT or downloads the system's operating-point pool.
  out               output .h5 path (default under the cache dir)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import glob
import os
import numpy as np
import h5py

# FdiaGenerator = physics/attack math; FAM_ID = family name -> integer id.
from ._core import FdiaGenerator, FAM_ID

# CACHE_DIR = on-disk shard home; register_local makes the new dataset findable by load(name).
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


def _load_states(
    system: Union[int, str], states: Optional[Union[str, np.ndarray]], pool_cap: int = 8000
) -> np.ndarray:
    """Return an operating-point pool [T,N,4] to inject attacks onto.

    Priority: explicit `states` -> $FDIA_GRAPH_INIT -> downloadable pool asset. A local init DIRECTORY can
    hold ~86k per-timestep files, so we stride-sample at most `pool_cap` evenly across the timeline (keeps
    daily/seasonal variety) rather than reading every file, which made a naive load take minutes.
    """
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
    # fall back to the downloadable operating-point pool for this system
    from .download import ensure_local
    from .registry import _RELEASE

    # Built-in release asset spec (pool_ieee{system}.npz); follows the pinned shard release, which carries
    # all 8 ladder pools (a hardcoded old tag here 404'd generate() on systems added after that tag).
    spec = {
        "kind": "builtin",
        "name": f"pool{system}",
        "file": f"pool_ieee{system}.npz",
        "release": _RELEASE,
        "repo": "myersben9/fdia-graph",
        "sha256": None,
    }
    # ensure_local downloads/caches and returns the local path.
    return np.load(ensure_local(spec))["X"].astype(np.float64)


def generate(
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
    redundancy: Optional[Dict] = None,
    split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 123,
    states: Optional[Union[str, np.ndarray]] = None,
    out: Optional[str] = None,
    outage: Optional[Union[int, str]] = None,
    targeting: str = "uniform",
    targeting_strength: float = 1.5,
) -> str:
    # targeting: how attacked-bus SETS are drawn. "uniform" (default) = uniform over attackable load buses,
    # byte-identical to prior releases. "centrality" tilts toward structurally critical buses (fused
    # degree/closeness/betweenness; Doostinia et al., IEEE TIA 2025) — more realistic and more damaging.
    # targeting_strength = exponential tilt (0 == uniform); applies to every family. Physics/stealth unchanged.
    # lra_targets: size of the LRA target-line pool (each attack picks one -> diverse bus sets).
    # outage: line to take OUT OF SERVICE for the whole shard (None = intact). One shard per topology, so the
    # graph/ group always describes its own topology; `states` must then be a pool re-solved under that same
    # topology (FdiaGenerator.resolve_states), or benign records carry intact-network voltages.
    # Meter coverage defaults (60% V buses, 20% PMU, 90% flows) with caller overrides merged on top.
    red = {"vbus_frac": 0.6, "pmu_frac": 0.2, "flow_frac": 0.9, **(redundancy or {})}
    # Outage is applied in the constructor before Ybus/base state and consumes no randomness, so the meter
    # plan and per-meter biases are identical across topologies at a given seed.
    g = FdiaGenerator(system, seed=seed, outage=outage, **red)
    # Pre-pick candidate LRA target lines (bound by intensity, up to min(6,#load) buses, from lra_targets lines).
    g._pick_lra_target(attack_intensity, min(6, len(g.load_bus)), n_targets=lra_targets)
    # K = max buses an LRA delta may touch; reuse g's seeded RNG for reproducibility.
    K = min(6, len(g.load_bus))
    rng = g.rng
    # Target-selection weights over attackable positions: None -> uniform, else centrality-biased vector.
    cent_p = g.centrality_probs(targeting_strength) if targeting == "centrality" else None
    # DESIGNED per-bus magnitude log (family_id, |delta|/|base|), written to <out>.mag.npz so the plausibility
    # band can be verified against what the gate enforces (designed fraction, no benign-baseline contamination).
    mag_log = []
    # Operating-point pool [T,N,4]; nT = #timesteps, C = #nodes/classes (label width).
    X = _load_states(system, states)
    nT = len(X)
    C = g.C
    # Precompute the swing feature's per-timestep "recent typical change" scale ONCE via prefix sums (a
    # per-record windowed std would be ~72k slow Python iterations).
    # SCALE[t,b,:] = std over [t-SWING_W, t) of the per-bus scan-to-scan |change| in P/Q.
    _D = np.abs(np.diff(X[:, :, :2], axis=0))  # [nT-1, N, 2] scan-to-scan |change|
    _c1 = np.concatenate([np.zeros((1,) + _D.shape[1:]), np.cumsum(_D, 0)], 0)  # prefix sum, [nT,N,2]
    _c2 = np.concatenate([np.zeros((1,) + _D.shape[1:]), np.cumsum(_D**2, 0)], 0)  # prefix sum of squares
    SCALE = np.full((nT, C, 2), 1e-3, np.float32)
    for t in range(2, nT):  # window changes D[max(0,t-W) .. t-2]
        s = max(0, t - SWING_W)
        e = t - 1  # sum over D[s..e-1] via c[e]-c[s]
        n = e - s
        if n >= 3:
            su = _c1[e] - _c1[s]
            sq = _c2[e] - _c2[s]
            SCALE[t] = np.sqrt(np.maximum(sq / n - (su / n) ** 2, 0.0)) + 1e-3
    # Translate the requested family names into their integer ids for membership tests below.
    fam_ids = [FAM_ID[f] for f in families]

    def _fin(
        nx: np.ndarray,
        nm: np.ndarray,
        ex: np.ndarray,
        em: np.ndarray,
        y: np.ndarray,
        family: int,
        sid: int,
        t: int,
        gap: int,
        stealthy: int,
    ) -> Tuple:
        # Finalize a record: attach two temporal features at injection-metered buses (0 elsewhere).
        # temporal_delta: current injection minus previous scan (single-step change).
        # swing: that change as a z-score of the bus's typical recent jump (SCALE[t]); single-shot spikes
        # (Aq/Al) read large, ramp At/benign stay ~1. This is the abrupt-change signal for localizing spikes.
        mP = nm[:, 1] > 0
        prev = X[t - 1] if t > 0 else X[t]
        td = np.zeros((C, 2), np.float32)
        td[mP, 0] = nx[mP, 1] - prev[mP, 0]
        td[mP, 1] = nx[mP, 2] - prev[mP, 1]
        sw = np.zeros((C, 2), np.float32)
        sc = SCALE[t]
        sw[mP, 0] = (nx[mP, 1] - prev[mP, 0]) / sc[mP, 0]
        sw[mP, 1] = (nx[mP, 2] - prev[mP, 1]) / sc[mP, 1]
        return (nx, nm, ex, em, y, family, sid, t, gap, stealthy, td, sw)

    def make(t: int, family: int, sid: int, atk: Any) -> Optional[Tuple]:
        # Build ONE record from X[t] for `family`. sid = sequence id (ramp only), atk = (bus-indices, multiplier).
        # Returns an 11-tuple, or None if the AC solve failed.
        Xt = X[t]
        if family in (1, 5):
            # Aq (1) / ramp (5): re-solve with scaled load, emit REAL (stealthy) measurements.
            # Base active load = stored P at load buses + generator P there; copy reactive load.
            Lp = Xt[g.load_bus, 0] + g.load_genP
            Lq = Xt[g.load_bus, 1].copy()
            Lp_true = Lp.copy()  # unattacked load -> pins generation dispatch in solve()
            dev = np.abs(np.asarray(atk[1]) - 1.0)  # per-bus designed load-shift fraction
            # Floor gate is Aq-only; ramp At is exempt so its per-scan step may stay sub-floor.
            if family == 1 and np.max(dev) < NOISE_FLOOR:
                return None  # within-noise no-op -> reject, loop redraws
            Lp = Lp.copy()
            Lp[atk[0]] *= atk[1]  # scale targeted load buses by the attack multiplier
            net = g.solve(Lp, Lq, Xt=Xt, Lp_true=Lp_true)  # re-solve, generation pinned to TRUE dispatch
            if net is None:
                return None  # non-convergence: skip (expected occasionally)
            nx, nm, ex, em = g.emit(net)
            y = np.zeros(C, np.uint8)
            y[g.load_bus[atk[0]]] = 1  # label attacked buses
            r = _fin(nx, nm, ex, em, y, family, sid, t, 0, 1)  # stealthy=1
            mf = np.zeros(C)
            mf[g.load_bus[atk[0]]] = dev  # per-bus DESIGNED magnitude (fraction of load)
            yb = y.astype(bool)
            mag_log.append((family, mf[yb], np.abs(r[-1][yb]).max(1)))  # (mag, swing) per bus
            return r
        if family == 6:
            # LRA (6): load-redistribution — zero-sum-ish delta d over K buses, then re-solve.
            Lp = Xt[g.load_bus, 0] + g.load_genP
            Lq = Xt[g.load_bus, 1].copy()
            d, a = g.lra_delta(
                Lp, attack_intensity, K, floor=NOISE_FLOOR
            )  # d = per-bus delta, a = attacked indices
            if len(a) == 0:
                return None  # no feasible redistribution -> skip
            dev = np.abs(d[a]) / (np.abs(Lp[a]) + 1e-6)  # per-bus designed redistribution fraction
            if np.min(dev) < NOISE_FLOOR:
                return None  # any bus inside the noise floor -> reject
            net = g.solve(
                Lp + d, Lq, Xt=Xt, Lp_true=Lp
            )  # redistributed load; generation pinned to TRUE dispatch
            if net is None:
                return None  # non-convergence -> skip
            nx, nm, ex, em = g.emit(net)
            y = np.zeros(C, np.uint8)
            y[g.load_bus[a]] = 1  # label the LRA bus set
            r = _fin(nx, nm, ex, em, y, 6, -1, t, 0, 1)  # LRA single-shot, stealthy=1
            mf = np.zeros(C)
            mf[g.load_bus[a]] = dev
            yb = y.astype(bool)
            mag_log.append((6, mf[yb], np.abs(r[-1][yb]).max(1)))
            return r
        # Remaining families (0 benign, 2/3/4 corrupt-in-place): emit from stored state, no re-solve.
        nx, nm, ex, em = g.emit_from_state(Xt)
        if family == 0:
            # Benign: keep a rolling buffer of clean frames feeding the Ar/As replay.
            g.benign_buf.append(nx.copy())
            if len(g.benign_buf) > 300:
                g.benign_buf.pop(0)  # cap buffer at 300 frames (FIFO)
            return _fin(nx, nm, ex, em, np.zeros(C, np.uint8), 0, -1, t, 0, 0)  # benign
        a = atk[0]  # indices into the LOAD-BUS ARRAY (0..#load-1) for a corrupt-in-place family (Ad/As/Ar)
        abus = g.load_bus[
            a
        ]  # -> actual BUS indices. corrupt()/emit index by bus, so map here: passing raw `a`
        # tampered buses 0..nlb while the label pointed at load_bus[a], corrupting DIFFERENT buses than labeled.
        # Ar/As replay a benign frame from the buffer for temporal contrast. replay_tau fixes the lag (exactly
        # tau frames back, clamped to the buffer); default is a random lag >=20 frames. Oldest/None if too small.
        if replay_tau is not None and g.benign_buf:
            replay = g.benign_buf[-min(replay_tau, len(g.benign_buf))]
        elif len(g.benign_buf) > 20:
            replay = g.benign_buf[int(rng.integers(0, len(g.benign_buf) - 20))]
        else:
            replay = g.benign_buf[0] if g.benign_buf else None
        # corrupt() tampers measurements in place per family code, keeping each realized change inside the band.
        nx, ex, weak, mags = g.corrupt(
            nx, ex, abus, _FAMK[family], replay, floor=NOISE_FLOOR, cap=attack_intensity
        )
        nx[nm == 0] = 0.0
        ex[em == 0] = 0.0  # corrupt() can write unmetered channels; re-assert mask==0 -> value==0
        if weak:
            return None  # replayed change fell inside the noise floor -> reject, loop redraws
        y = np.zeros(C, np.uint8)
        y[abus] = 1  # label the SAME buses that were corrupted (now consistent)
        r = _fin(nx, nm, ex, em, y, family, -1, t, 0, 0)  # corrupt-in-place single-shot, stealthy=0
        if len(mags):
            mf = np.zeros(C)
            mf[abus] = mags
            yb = y.astype(bool)
            mag_log.append((family, mf[yb], np.abs(r[-1][yb]).max(1)))
        return r

    recs = []
    sid = 0  # accumulate all records; sid = next ramp sequence id
    # Per-family (attempts, accepted). The loops retry on non-convergence until the quota is met, so meeting
    # quota hides how many operating points couldn't be attacked. Recorded to shard attrs because the drop rate
    # is the honest measure of how much harder a topology (e.g. N-1) is to attack.
    tried = {}
    taken = {}
    for t in rng.choice(nT, min(n_benign, nT), replace=False):  # benign
        recs.append(make(int(t), 0, -1, None))
    for fam in [k for k in (1, 2, 3, 4, 6) if k in fam_ids]:  # single-shot families (retry to target)
        # Draw until per_family succeed, capped at per_family*25 attempts so infeasible configs can't loop forever.
        # Targets come ONLY from attackable positions (real active load); a indexes load_bus and Lp alike.
        apos = g.attackable_pos
        nab = len(apos)
        got = tries = 0
        while got < per_family and tries < per_family * 25:
            tries += 1
            t = int(rng.integers(nT))
            if fam == 1:
                # Aq: variable footprint (1..6 buses), each with its OWN multiplier in 1.05..1+intensity.
                k = int(rng.integers(1, min(6, nab) + 1))
                a = rng.choice(apos, k, replace=False, p=cent_p)
                mult = 1 + rng.uniform(0.05, attack_intensity, size=k)
            else:
                # Ad/As/Ar and LRA: up to 4 buses; their make() branches ignore mult (corrupt in place / LRA delta).
                a = rng.choice(apos, min(4, nab), replace=False, p=cent_p)
                mult = 1 + rng.uniform(0.05, attack_intensity)
            r = make(t, fam, -1, (a, mult))
            if r is not None:
                recs.append(r)
                got += 1  # count only converged records
        tried[fam] = tries
        taken[fam] = got
    if 5 in fam_ids:  # ramp sequences to ~per_family records
        # Ramp = temporal surge/dip with an ASYMMETRIC shape: a fixed bus set ramps up or down at rate_up to a
        # peak/trough, optionally holds, then returns at a different rate_down. Direction/turn/slopes vary per
        # sequence; each is self-contained (ends near baseline, no jump).
        ramp_got = 0
        ramp_steps = 0
        ramp_seqs = 0
        while ramp_got < per_family:
            ramp_seqs += 1
            t0 = int(rng.integers(nT - ramp_len))  # start leaving room for the full sequence
            atk = rng.choice(
                g.attackable_pos, min(5, len(g.attackable_pos)), replace=False, p=cent_p
            )  # fixed bus set
            direction = 1.0 if rng.random() < 0.5 else -1.0  # +1 = surge first, -1 = dip first
            rate_up = ramp_rate * rng.uniform(0.7, 1.3)
            rate_down = ramp_rate * rng.uniform(0.7, 1.3)  # independent slopes
            rise_len = max(1, int(rng.uniform(0.20, 0.45) * ramp_len))  # steps ramping to the peak/trough
            hold_len = int(rng.uniform(0.0, 0.25) * ramp_len)  # steps held at the peak (0 = no plateau)
            peak_dev = rate_up * rise_len  # deviation magnitude at the turn
            seq = []
            for i in range(ramp_len):
                ramp_steps += 1
                if i < rise_len:
                    dev = rate_up * i  # ramp toward peak
                elif i < rise_len + hold_len:
                    dev = peak_dev  # hold
                else:
                    dev = max(0.0, peak_dev - rate_down * (i - rise_len - hold_len))  # ramp back
                r = make(t0 + i, 5, sid, (atk, 1 + direction * dev))  # multiplier = 1 +/- deviation
                if r is None:
                    break  # abort sequence on first non-converging step
                seq.append(r)
            # Keep only if >=10 steps solved, then advance the sequence id.
            if len(seq) >= 10:
                recs.extend(seq)
                ramp_got += len(seq)
                sid += 1
        # Ramp "attempts" = timesteps SOLVED, so attempts-minus-accepted is the steps thrown away.
        tried[5] = ramp_steps
        taken[5] = ramp_got

    # Output path: <name>.h5 under the SDK cache dir unless `out` overrode it.
    out = out or os.path.join(CACHE_DIR, f"{name}.h5")
    _write(g, recs, out, split, seed, solve_stats=(tried, taken))
    # Sidecar: flatten per-bus designed magnitudes into (family_id, magnitude) rows for band verification.
    if mag_log:
        fam_col = np.concatenate([np.full(len(m), fid, np.int8) for fid, m, s in mag_log])
        mag_col = np.concatenate([np.asarray(m, float) for _, m, s in mag_log])
        sw_col = np.concatenate([np.asarray(s, float) for _, m, s in mag_log])
        np.savez(
            out + ".mag.npz",
            family=fam_col,
            mag=mag_col,
            swing=sw_col,
            floor=NOISE_FLOOR,
            cap=attack_intensity,
        )
    # Register locally under `name` with reproducibility metadata so load(name) finds it.
    register_local(
        name,
        out,
        meta=dict(
            system=system,
            per_family=per_family,
            families=list(families),
            attack_intensity=attack_intensity,
            ramp_rate=ramp_rate,
            seed=seed,
            outage_line=g.outage if g.outage is not None else -1,
        ),
    )
    return out


def _write(
    g: "FdiaGenerator",
    recs: List,
    out: str,
    split: Tuple[float, float, float],
    seed: int,
    solve_stats: Optional[Tuple[Dict, Dict]] = None,
) -> None:
    # Serialize the record-tuples to one HDF5 file: graph structure + stacked per-record arrays + split.
    T = len(recs)
    C, E = g.C, g.E  # T records, C nodes, E edges

    def arr(i, dt):
        return np.array([r[i] for r in recs], dt)  # pull tuple field i across all records as dtype dt

    # Array-valued fields -> [T, ...] tensors.
    node_x = np.stack([r[0] for r in recs])
    node_m = np.stack([r[1] for r in recs])
    edge_x = np.stack([r[2] for r in recs])
    edge_m = np.stack([r[3] for r in recs])
    y = np.stack([r[4] for r in recs])
    temporal_delta = np.stack([r[10] for r in recs])  # [T,N,2] current-minus-previous-scan injection
    swing = np.stack([r[11] for r in recs])  # [T,N,2] windowed relative-swing (z-score)
    # Scalar-per-record fields (dtypes sized to range).
    fam = arr(5, np.int8)
    seq = arr(6, np.int32)
    tstep = arr(7, np.int32)
    gap = arr(8, np.uint8)
    st = arr(9, np.uint8)
    # train/val/test (0/1/2) per record, chronological and sequence-aware.
    sp = _chrono_split(tstep, seq, split)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with h5py.File(out, "w") as f:
        # File attrs: dims, feature legends, family legend, LRA target line, seed. Units are ENGINEERING
        # quantities (node_x = [V pu, P_inj MW, Q_inj MVAr, theta deg], edge flows MW/MVAr); baseMVA is the
        # power base letting the loader also serve a per-unit view.
        f.attrs.update(
            dict(
                system=C,
                N=C,
                E=E,
                n_records=T,
                node_feat="V,P_inj,Q_inj,theta",
                edge_feat="P_from,Q_from",
                node_units="V:pu,P_inj:MW,Q_inj:MVAr,theta:deg",
                edge_units="P_from:MW,Q_from:MVAr",
                baseMVA=float(g.base.sn_mva),
                families="0benign,1Aq,2Ad,3As,4Ar,5At,6Al",
                lra_target_line=g._Ltgt,
                seed=seed,
            )
        )
        # TOPOLOGY provenance: "base" = intact, "n1_line" = one line out for every record. base_flow gives the
        # contingency size (the ranking the scenario was selected by). Recorded so a shard is self-describing.
        f.attrs.update(
            dict(
                topology=("base" if g.outage is None else "n1_line"),
                outage_line=(-1 if g.outage is None else int(g.outage)),
                outage_branch_pos=int(g.outage_pos),
                outage_line_name=g.outage_name,
                outage_from_bus=int(g.outage_from_bus),
                outage_to_bus=int(g.outage_to_bus),
                outage_base_flow_mw=float(g.outage_base_flow_mw),
            )
        )
        # Per-family yield "famid:attempts/accepted" so the drop rate is readable from the file.
        if solve_stats is not None:
            _tr, _tk = solve_stats
            f.attrs["solve_yield"] = ",".join(f"{k}:{_tr[k]}/{_tk[k]}" for k in sorted(_tr))
        # graph/ group: static topology shared by all records.
        gg = f.create_group("graph")
        gg.create_dataset("edge_index", data=g.ei)
        # DEPRECATED, unit-inconsistent (ohms for lines, vk percent for trafos). Kept for v0.4.x readers.
        gg.create_dataset("edge_reactance", data=g.x_react)
        # Full per-unit branch physics + bus shunts: reconstruct Ybus EXACTLY (verified vs makeYbus to
        # 7e-15/3e-14/5e-13 on IEEE 14/118/300), so a model reads exactly the estimator's physics.
        for _n, _v in (
            ("edge_r", g.edge_r),
            ("edge_x", g.edge_x),
            ("edge_b", g.edge_b),
            ("edge_g", g.edge_g),
            ("edge_gs", g.edge_gs),
            ("edge_bs", g.edge_bs),
            ("edge_tap", g.edge_tap),
            ("edge_shift", g.edge_shift),
            ("edge_status", g.edge_status),
            ("edge_is_trafo", g.edge_is_trafo),
            ("bus_shunt_g", g.bus_shunt_g),
            ("bus_shunt_b", g.bus_shunt_b),
        ):
            gg.create_dataset(_n, data=_v)
        gg.attrs.update(
            dict(
                edge_feat_static="r,x,b,g,tap,shift,status,is_trafo (per unit, ppc order = lines then trafos)",
                bus_feat_static="shunt_g,shunt_b (MW/MVAr at 1.0 pu, ppc bus order)",
                edge_reactance_deprecated="mixes ohms (lines) with vk_percent (trafos); use edge_x",
                ybus_reconstructible="yes, see fdia_graph tests: Y = f(edge_r,x,b,g,tap,shift,status)+bus shunts",
            )
        )
        # data/ group: per-record tensors, chunked along the record axis (<=128) for efficient partial reads.
        d = f.create_group("data")
        ch = (min(128, T),)
        d.create_dataset(
            "node_x", data=node_x, chunks=ch + node_x.shape[1:], compression="gzip", compression_opts=4
        )
        d.create_dataset("node_m", data=node_m, compression="gzip")
        d.create_dataset(
            "edge_x", data=edge_x, chunks=ch + edge_x.shape[1:], compression="gzip", compression_opts=4
        )
        d.create_dataset("edge_m", data=edge_m, compression="gzip")
        d.create_dataset("y", data=y, compression="gzip")
        d.create_dataset(
            "temporal_delta",
            data=temporal_delta,
            chunks=ch + temporal_delta.shape[1:],
            compression="gzip",
            compression_opts=4,
        )
        d.create_dataset(
            "swing", data=swing, chunks=ch + swing.shape[1:], compression="gzip", compression_opts=4
        )
        # Each scalar field (including the split) as its own dataset.
        for nm_, a in [
            ("family", fam),
            ("seq_id", seq),
            ("timestep", tstep),
            ("gap", gap),
            ("stealthy", st),
            ("split", sp),
        ]:
            d.create_dataset(nm_, data=a)


def _chrono_split(tstep: np.ndarray, seq: np.ndarray, frac: Tuple[float, float, float]) -> np.ndarray:
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
