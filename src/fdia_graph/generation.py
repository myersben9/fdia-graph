"""generate(system, name, **knobs) — build a custom dataset and register it as `name`.

Research knobs (all optional, sensible defaults matching the published shards):
  per_family        int   attacked records per family (default 3000)
  families          list  which attacks to include (default all: Aq,Ad,As,Ar,At,Al)
  attack_intensity  float per-bus load shift magnitude for Aq / Al(LRA) bound (default 0.15 = ±15%)
  ramp_rate         float ramp perturbation growth per step (default 0.002)
  ramp_len          int   ramp sequence length (default 60)
  n_benign          int   benign records (default 20000)
  redundancy        dict  meter coverage: {vbus_frac,pmu_frac,flow_frac} (default 0.6/0.2/0.9)
  split             tuple chronological train/val/test fractions (default (0.6,0.2,0.2))
  seed              int   (default 123)
  states            source of operating points: path to a pool .npz (key 'X' [T,N,4]) or an init dir of
                    X_*.npy; if None, uses $FDIA_GRAPH_INIT or downloads the system's operating-point pool.
  out               output .h5 path (default under the cache dir)
"""
# stdlib + numeric deps: glob/os for filesystem discovery, numpy for arrays, h5py to write the shard.
import glob, os, numpy as np, h5py
# FdiaGenerator does the actual physics/attack math; FAM_ID maps a family name -> its integer id.
from ._core import FdiaGenerator, FAM_ID
# CACHE_DIR is the SDK's on-disk home for shards; register_local records the new dataset so load(name) finds it.
from .registry import CACHE_DIR, register_local

# Map of single-shot family name -> integer id, used for reference/lookups (Aq=1, Ad=2, As=3, Ar=4, Al/LRA=6).
_SINGLE = {"Aq": 1, "Ad": 2, "As": 3, "Ar": 4, "Al": 6}
# Reverse lookup for the "corrupt-in-place" families (Ad/As/Ar): family id -> letter code passed to g.corrupt().
_FAMK = {2: "Ad", 3: "As", 4: "Ar"}
# Window (in scans) for the per-bus "swing" feature — how far back to measure each bus's recent-normal range.
SWING_W = 20


def _load_states(system, states, pool_cap=8000):
    """Return an operating-point pool [T,N,4] to inject attacks onto.

    Priority: explicit `states` arg -> $FDIA_GRAPH_INIT -> downloadable pool asset. A local init DIRECTORY
    can hold ~86k per-timestep files, so we stride-sample at most `pool_cap` of them (evenly across the
    timeline, preserving daily/seasonal load variety) rather than reading every file — that read is what
    made a naive load take minutes.
    """
    # In-memory pool: profiles.generate_states(...) returns a [T,N,4] array; accept it directly so the whole
    # pipeline (load_profile -> generate_states -> generate) can run without touching disk. (Checked before the
    # `or` below because a numpy array has no unambiguous truth value.)
    if isinstance(states, np.ndarray):
        return states.astype(np.float64)
    # Resolve the source: caller-supplied `states` wins, else the FDIA_GRAPH_INIT env var, else None (downloaded later).
    src = states or os.environ.get("FDIA_GRAPH_INIT")
    if src and os.path.isdir(src):
        # Init DIRECTORY case: gather all per-timestep X_*.npy, sorted by their integer timestep (name is "X_<t>.npy").
        xs = sorted(glob.glob(os.path.join(src, "X_*.npy")), key=lambda p: int(os.path.basename(p)[2:-4]))
        stride = max(1, len(xs) // pool_cap)                    # even stride -> diverse sample, bounded cost
        # Read only every `stride`-th file, cap at pool_cap, stack to [T,N,4]; float64 for the WLS/AC solve downstream.
        return np.stack([np.load(f) for f in xs[::stride][:pool_cap]]).astype(np.float64)
    if src and src.endswith(".npz"):
        return np.load(src)["X"].astype(np.float64)             # precomputed compact pool (the SDK default)
    # fall back to the downloadable operating-point pool for this system
    from .download import ensure_local
    # Describe the built-in release asset (pool_ieee{system}.npz) to fetch; release/repo overridable via env.
    spec = {"kind": "builtin", "name": f"pool{system}", "file": f"pool_ieee{system}.npz",
            "release": os.environ.get("FDIA_GRAPH_RELEASE", "v0.1.0"), "repo": "myersben9/fdia-graph", "sha256": None}
    # ensure_local downloads+caches the asset and returns its local path; load its 'X' key as the pool.
    return np.load(ensure_local(spec))["X"].astype(np.float64)


def generate(system, name, per_family=3000, families=("Aq", "Ad", "As", "Ar", "At", "Al"),
             attack_intensity=0.15, ramp_rate=0.002, ramp_len=60, n_benign=20000, lra_targets=15,
             redundancy=None, split=(0.6, 0.2, 0.2), seed=123, states=None, out=None):
    # lra_targets: size of the LRA target-line pool (each LRA attack picks one at random -> diverse bus sets)
    # Merge caller's redundancy overrides onto the default meter coverage (60% V buses, 20% PMU, 90% flows).
    red = {"vbus_frac": 0.6, "pmu_frac": 0.2, "flow_frac": 0.9, **(redundancy or {})}
    # Build the generator for this IEEE system: fixes the seed and wires in the meter-coverage fractions.
    g = FdiaGenerator(system, seed=seed, **red)
    # Pre-compute a pool of candidate LRA target lines (load-redistribution attack): bound by intensity, up to
    # min(6, #load buses) buses per attack, drawing from `lra_targets` distinct target lines for variety.
    g._pick_lra_target(attack_intensity, min(6, len(g.load_bus)), n_targets=lra_targets)
    # K = max buses an LRA delta may touch; reuse g's seeded RNG so the whole build is reproducible.
    K = min(6, len(g.load_bus)); rng = g.rng
    # Load the operating-point pool [T,N,4]; nT = #timesteps available, C = #nodes/classes (label width).
    X = _load_states(system, states); nT = len(X); C = g.C
    # Translate the requested family names into their integer ids for membership tests below.
    fam_ids = [FAM_ID[f] for f in families]

    def _fin(nx, nm, ex, em, y, family, sid, t, gap, stealthy):
        # Finalize a record: attach two temporal features at injection-metered buses (0 elsewhere).
        #  temporal_delta [N,2] = current injection - PREVIOUS scan's injection (single-step change).
        #  swing [N,2] = windowed relative swing = (current reading - recent-window mean) / (recent-window std):
        #    a per-bus z-score of how far the current injection deviates from its OWN recent normal over the past
        #    SWING_W scans. A single-shot attack (Aq/Al) spikes far outside its recent window -> large swing; a
        #    gradual ramp (At) or benign drift stays inside it -> small. This hands the model the abrupt-change
        #    signal directly, so it can localize the spike-and-revert attacks a rate-of-change detector would flag.
        mP = nm[:, 1] > 0
        prev = X[t - 1] if t > 0 else X[t]
        td = np.zeros((C, 2), np.float32)
        td[mP, 0] = nx[mP, 1] - prev[mP, 0]; td[mP, 1] = nx[mP, 2] - prev[mP, 1]
        sw = np.zeros((C, 2), np.float32)
        w0 = max(0, t - SWING_W)                            # recent window of TRUE states [t-SWING_W, t)
        if t - w0 >= 4:
            dwin = np.abs(np.diff(X[w0:t, :, :2], axis=0))  # [w-1,N,2] per-bus scan-to-scan |change| over the window
            scale = dwin.std(0) + 1e-3                       # this bus's TYPICAL recent change magnitude
            # rate-of-change z-score: how big is THIS scan's jump vs the bus's normal recent jumps? A single-shot
            # attack (Aq/Al) is a huge abrupt jump -> large; a gradual ramp (At) or benign drift stays ~1.
            sw[mP, 0] = (nx[mP, 1] - prev[mP, 0]) / scale[mP, 0]
            sw[mP, 1] = (nx[mP, 2] - prev[mP, 1]) / scale[mP, 1]
        return (nx, nm, ex, em, y, family, sid, t, gap, stealthy, td, sw)

    def make(t, family, sid, atk):
        # Build ONE record from operating point X[t] for the given family. `sid` = sequence id (ramp only),
        # `atk` = (bus-indices, multiplier) payload. Returns an 11-tuple record, or None if the AC solve failed.
        Xt = X[t]
        if family in (1, 5):
            # Aq (1) and ramp (5): re-solve power flow with a scaled load, then emit REAL (stealthy) measurements.
            # Base active load = stored P injection at load buses + generator P at those buses; copy reactive load.
            Lp = Xt[g.load_bus, 0] + g.load_genP; Lq = Xt[g.load_bus, 1].copy()
            Lp_true = Lp.copy()                    # unattacked load -> pins the generation dispatch in solve()
            Lp = Lp.copy(); Lp[atk[0]] *= atk[1]   # scale the targeted load buses by the attack multiplier atk[1]
            net = g.solve(Lp, Lq, Xt=Xt, Lp_true=Lp_true)   # re-solve, generation pinned to the TRUE dispatch
            if net is None: return None            # non-convergence: skip this timestep (expected occasionally)
            # emit() -> node feats, node mask, edge feats, edge mask for the solved (attacked) network.
            nx, nm, ex, em = g.emit(net); y = np.zeros(C, np.uint8); y[g.load_bus[atk[0]]] = 1  # label attacked buses
            return _fin(nx, nm, ex, em, y, family, sid, t, 0, 1)   # stealthy=1
        if family == 6:
            # LRA (6): load-redistribution — compute a zero-sum-ish delta d over K buses, then re-solve.
            Lp = Xt[g.load_bus, 0] + g.load_genP; Lq = Xt[g.load_bus, 1].copy()
            d, a = g.lra_delta(Lp, attack_intensity, K)   # d = per-bus load delta, a = indices actually attacked
            if len(a) == 0: return None                   # no feasible redistribution found -> skip
            net = g.solve(Lp + d, Lq, Xt=Xt, Lp_true=Lp)  # redistributed load; generation pinned to TRUE dispatch
            if net is None: return None                   # non-convergence -> skip
            nx, nm, ex, em = g.emit(net); y = np.zeros(C, np.uint8); y[g.load_bus[a]] = 1  # label the LRA bus set
            return _fin(nx, nm, ex, em, y, 6, -1, t, 0, 1)   # LRA single-shot, stealthy=1
        # Remaining families (0 benign, 2/3/4 corrupt-in-place): emit directly from the stored state, no re-solve.
        nx, nm, ex, em = g.emit_from_state(Xt)
        if family == 0:
            # Benign record: keep a rolling buffer of recent clean node-feature frames (used as replay for Ar/As).
            g.benign_buf.append(nx.copy())
            if len(g.benign_buf) > 300: g.benign_buf.pop(0)   # cap buffer at 300 frames (FIFO)
            return _fin(nx, nm, ex, em, np.zeros(C, np.uint8), 0, -1, t, 0, 0)   # benign
        a = atk[0]   # the attacked bus indices for a corrupt-in-place family (Ad/As/Ar)
        # Ar/As may replay an earlier benign frame: pick one at least ~20 frames back for temporal contrast,
        # falling back to the oldest (or None) when the buffer is too small.
        replay = g.benign_buf[int(rng.integers(0, len(g.benign_buf)-20))] if len(g.benign_buf) > 20 else (g.benign_buf[0] if g.benign_buf else None)
        # corrupt() tampers the measurements in place per the family code (Ad/As/Ar) using the replay frame.
        nx, ex = g.corrupt(nx, ex, a, _FAMK[family], replay)
        y = np.zeros(C, np.uint8); y[g.load_bus[a]] = 1   # label the corrupted buses
        return _fin(nx, nm, ex, em, y, family, -1, t, 0, 0)   # corrupt-in-place single-shot, stealthy=0

    recs = []; sid = 0   # accumulate all records; sid = next ramp sequence id
    for t in rng.choice(nT, min(n_benign, nT), replace=False):        # benign
        # Sample n_benign distinct timesteps (capped at nT) and emit a benign record for each.
        recs.append(make(int(t), 0, -1, None))
    for fam in [k for k in (1, 2, 3, 4, 6) if k in fam_ids]:           # single-shot families (retry to target)
        # For each requested single-shot family, keep drawing random timesteps until `per_family` records
        # succeed, with a hard cap of per_family*25 attempts so infeasible configs can't loop forever.
        nlb = len(g.load_bus)
        got = tries = 0
        while got < per_family and tries < per_family*25:
            tries += 1; t = int(rng.integers(nT))
            if fam == 1:
                # Aq: a VARIABLE number of attacked load buses (1..6), each rescaled by its OWN multiplier
                # (per-bus magnitude, independently drawn in 1.05..1+intensity) — so the attacked footprint
                # varies in both size and per-bus strength rather than a fixed 4 buses at one shared factor.
                k = int(rng.integers(1, min(6, nlb) + 1))
                a = rng.choice(nlb, k, replace=False)
                mult = 1 + rng.uniform(0.05, attack_intensity, size=k)
            else:
                # Meter-level families (Ad/As/Ar) and LRA: up to 4 buses; the multiplier is unused by their
                # make() branches (they corrupt in place / compute an LRA delta), so a scalar is fine.
                a = rng.choice(nlb, min(4, nlb), replace=False)
                mult = 1 + rng.uniform(0.05, attack_intensity)
            r = make(t, fam, -1, (a, mult))
            if r is not None: recs.append(r); got += 1   # count only successful (converged) records
    if 5 in fam_ids:                                                   # ramp sequences to ~per_family records
        # Ramp attacks are temporal load surges/dips with a randomized, ASYMMETRIC shape: a fixed bus set is
        # ramped in one direction (up = surge, or down = dip) at rate_up to a peak/trough, optionally HELD there
        # for a while, then returned toward baseline at a DIFFERENT rate_down. The turning point falls at a
        # random time, the two slopes differ, and the direction is chosen per sequence — so the family covers
        # up-then-down and down-then-up ramps. Each sequence is self-contained (ends back near baseline, no jump).
        ramp_got = 0
        while ramp_got < per_family:
            t0 = int(rng.integers(nT - ramp_len))   # start timestep leaving room for the full sequence
            atk = rng.choice(len(g.load_bus), min(5, len(g.load_bus)), replace=False)  # fixed bus set for the sequence
            direction = 1.0 if rng.random() < 0.5 else -1.0            # +1 = surge up first, -1 = dip down first
            rate_up = ramp_rate*rng.uniform(0.7, 1.3); rate_down = ramp_rate*rng.uniform(0.7, 1.3)  # independent slopes
            rise_len = max(1, int(rng.uniform(0.20, 0.45)*ramp_len))  # steps spent ramping to the peak/trough
            hold_len = int(rng.uniform(0.0, 0.25)*ramp_len)           # steps held at the peak/trough (0 = no plateau)
            peak_dev = rate_up*rise_len                               # deviation magnitude at the turn (signed below)
            seq = []
            for i in range(ramp_len):
                if i < rise_len:                dev = rate_up*i                                   # ramp toward the peak
                elif i < rise_len+hold_len:     dev = peak_dev                                    # hold at the peak
                else:                           dev = max(0.0, peak_dev - rate_down*(i-rise_len-hold_len))  # ramp back
                r = make(t0+i, 5, sid, (atk, 1 + direction*dev))      # multiplier = 1 +/- deviation (2 directions)
                if r is None: break                 # abort this sequence on first non-converging step
                seq.append(r)
            # Keep the sequence only if at least 10 steps solved; then advance the sequence id.
            if len(seq) >= 10: recs.extend(seq); ramp_got += len(seq); sid += 1

    # Default output path lives under the SDK cache dir as <name>.h5 unless the caller overrode `out`.
    out = out or os.path.join(CACHE_DIR, f"{name}.h5")
    # Serialize every record + graph structure + chronological split into the HDF5 shard.
    _write(g, recs, out, split, seed)
    # Register the shard locally under `name` (with reproducibility metadata) so load(name) can retrieve it later.
    register_local(name, out, meta=dict(system=system, per_family=per_family, families=list(families),
                                        attack_intensity=attack_intensity, ramp_rate=ramp_rate, seed=seed))
    return out   # hand back the path to the written shard


def _write(g, recs, out, split, seed):
    # Serialize the list of record-tuples into one HDF5 file: graph structure + stacked per-record arrays + split.
    T = len(recs); C, E = g.C, g.E   # T records, C nodes, E edges
    arr = lambda i, dt: np.array([r[i] for r in recs], dt)   # helper: pull tuple field i across all records as dtype dt
    # Stack the array-valued fields into [T, ...] tensors: node feats/mask, edge feats/mask, and labels y.
    node_x = np.stack([r[0] for r in recs]); node_m = np.stack([r[1] for r in recs])
    edge_x = np.stack([r[2] for r in recs]); edge_m = np.stack([r[3] for r in recs]); y = np.stack([r[4] for r in recs])
    temporal_delta = np.stack([r[10] for r in recs])          # [T,N,2] current-minus-previous-scan injection feature
    swing = np.stack([r[11] for r in recs])                   # [T,N,2] windowed relative-swing (recent-window z-score)
    # Scalar-per-record fields: family id, sequence id, timestep, gap flag, stealthy flag (dtypes sized to range).
    fam = arr(5, np.int8); seq = arr(6, np.int32); tstep = arr(7, np.int32); gap = arr(8, np.uint8); st = arr(9, np.uint8)
    # Compute the train/val/test assignment (0/1/2) per record, chronologically and sequence-aware.
    sp = _chrono_split(tstep, seq, split)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)   # ensure the output directory exists
    with h5py.File(out, "w") as f:
        # File-level attributes: dimensions, feature-column legends, family legend, LRA target line, seed.
        f.attrs.update(dict(system=C, N=C, E=E, n_records=T, node_feat="V,P_inj,Q_inj,theta", edge_feat="P_from,Q_from",
                            families="0benign,1Aq,2Ad,3As,4Ar,5At,6Al", lra_target_line=g._Ltgt, seed=seed))
        # graph/ group: static topology shared by all records — edge_index and per-edge reactance.
        gg = f.create_group("graph"); gg.create_dataset("edge_index", data=g.ei); gg.create_dataset("edge_reactance", data=g.x_react)
        # data/ group: the per-record tensors. Chunk along the record axis (<=128) for efficient partial reads.
        d = f.create_group("data"); ch = (min(128, T),)
        d.create_dataset("node_x", data=node_x, chunks=ch+node_x.shape[1:], compression="gzip", compression_opts=4)
        d.create_dataset("node_m", data=node_m, compression="gzip")
        d.create_dataset("edge_x", data=edge_x, chunks=ch+edge_x.shape[1:], compression="gzip", compression_opts=4)
        d.create_dataset("edge_m", data=edge_m, compression="gzip"); d.create_dataset("y", data=y, compression="gzip")
        d.create_dataset("temporal_delta", data=temporal_delta, chunks=ch+temporal_delta.shape[1:], compression="gzip", compression_opts=4)
        d.create_dataset("swing", data=swing, chunks=ch+swing.shape[1:], compression="gzip", compression_opts=4)
        # Store each scalar field (including the computed split) as its own dataset alongside the tensors.
        for nm_, a in [("family", fam), ("seq_id", seq), ("timestep", tstep), ("gap", gap), ("stealthy", st), ("split", sp)]:
            d.create_dataset(nm_, data=a)


def _chrono_split(tstep, seq, frac):
    # Assign each record to train(0)/val(1)/test(2) by chronological order, keeping ramp sequences intact.
    T = len(tstep)
    # Build groups: one group per ramp sequence id (seq>=0), plus one singleton group per non-sequence record (seq<0).
    groups = [np.where(seq == s)[0] for s in np.unique(seq[seq >= 0])] + [np.array([i]) for i in np.where(seq < 0)[0]]
    # Order the groups by their earliest timestep so the split is chronological (no future leaking into train).
    groups.sort(key=lambda gp: int(tstep[gp].min()))
    sp = np.empty(T, np.int8); c = 0   # sp = per-record split labels; c = running count already placed
    for gp in groups:
        # Cumulative fraction c/T decides the bucket: <frac[0] -> train(0), <frac[0]+frac[1] -> val(1), else test(2).
        f = c/T; sp[gp] = 0 if f < frac[0] else (1 if f < frac[0]+frac[1] else 2); c += len(gp)
    return sp
