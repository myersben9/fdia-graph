"""generate(system, name, **knobs) — build a custom dataset and register it as `name`.

Research knobs (all optional, sensible defaults matching the published shards):
  per_family        int   attacked records per family (default 3000)
  families          list  which attacks to include (default all: Ao,Ad,As,Ar,ramp,LRA)
  attack_intensity  float per-bus load shift magnitude for Ao / LRA bound (default 0.15 = ±15%)
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
import glob, os, numpy as np, h5py
from ._core import FdiaGenerator, FAM_ID
from .registry import CACHE_DIR, register_local

_SINGLE = {"Ao": 1, "Ad": 2, "As": 3, "Ar": 4, "LRA": 6}
_FAMK = {2: "Ad", 3: "As", 4: "Ar"}


def _load_states(system, states, pool_cap=8000):
    """Return an operating-point pool [T,N,4] to inject attacks onto.

    Priority: explicit `states` arg -> $FDIA_GRAPH_INIT -> downloadable pool asset. A local init DIRECTORY
    can hold ~86k per-timestep files, so we stride-sample at most `pool_cap` of them (evenly across the
    timeline, preserving daily/seasonal load variety) rather than reading every file — that read is what
    made a naive load take minutes.
    """
    src = states or os.environ.get("FDIA_GRAPH_INIT")
    if src and os.path.isdir(src):
        xs = sorted(glob.glob(os.path.join(src, "X_*.npy")), key=lambda p: int(os.path.basename(p)[2:-4]))
        stride = max(1, len(xs) // pool_cap)                    # even stride -> diverse sample, bounded cost
        return np.stack([np.load(f) for f in xs[::stride][:pool_cap]]).astype(np.float64)
    if src and src.endswith(".npz"):
        return np.load(src)["X"].astype(np.float64)             # precomputed compact pool (the SDK default)
    # fall back to the downloadable operating-point pool for this system
    from .download import ensure_local
    spec = {"kind": "builtin", "name": f"pool{system}", "file": f"pool_ieee{system}.npz",
            "release": os.environ.get("FDIA_GRAPH_RELEASE", "v0.1.0"), "repo": "myersben9/fdia-graph", "sha256": None}
    return np.load(ensure_local(spec))["X"].astype(np.float64)


def generate(system, name, per_family=3000, families=("Ao", "Ad", "As", "Ar", "ramp", "LRA"),
             attack_intensity=0.15, ramp_rate=0.002, ramp_len=60, n_benign=20000, lra_targets=15,
             redundancy=None, split=(0.6, 0.2, 0.2), seed=123, states=None, out=None):
    # lra_targets: size of the LRA target-line pool (each LRA attack picks one at random -> diverse bus sets)
    red = {"vbus_frac": 0.6, "pmu_frac": 0.2, "flow_frac": 0.9, **(redundancy or {})}
    g = FdiaGenerator(system, seed=seed, **red)
    g._pick_lra_target(attack_intensity, min(6, len(g.load_bus)), n_targets=lra_targets)
    K = min(6, len(g.load_bus)); rng = g.rng
    X = _load_states(system, states); nT = len(X); C = g.C
    fam_ids = [FAM_ID[f] for f in families]

    def make(t, family, sid, atk):
        Xt = X[t]
        if family in (1, 5):
            Lp = Xt[g.load_bus, 0] + g.load_genP; Lq = Xt[g.load_bus, 1].copy()
            Lp = Lp.copy(); Lp[atk[0]] *= atk[1]
            net = g.solve(Lp, Lq)
            if net is None: return None
            nx, nm, ex, em = g.emit(net); y = np.zeros(C, np.uint8); y[g.load_bus[atk[0]]] = 1
            return (nx, nm, ex, em, y, family, sid, t, 0, 1)
        if family == 6:
            Lp = Xt[g.load_bus, 0] + g.load_genP; Lq = Xt[g.load_bus, 1].copy()
            d, a = g.lra_delta(Lp, attack_intensity, K)
            if len(a) == 0: return None
            net = g.solve(Lp + d, Lq)
            if net is None: return None
            nx, nm, ex, em = g.emit(net); y = np.zeros(C, np.uint8); y[g.load_bus[a]] = 1
            return (nx, nm, ex, em, y, 6, -1, t, 0, 1)
        nx, nm, ex, em = g.emit_from_state(Xt)
        if family == 0:
            g.benign_buf.append(nx.copy())
            if len(g.benign_buf) > 300: g.benign_buf.pop(0)
            return (nx, nm, ex, em, np.zeros(C, np.uint8), 0, -1, t, 0, 0)
        a = atk[0]
        replay = g.benign_buf[int(rng.integers(0, len(g.benign_buf)-20))] if len(g.benign_buf) > 20 else (g.benign_buf[0] if g.benign_buf else None)
        nx, ex = g.corrupt(nx, ex, a, _FAMK[family], replay)
        y = np.zeros(C, np.uint8); y[g.load_bus[a]] = 1
        return (nx, nm, ex, em, y, family, -1, t, 0, 0)

    recs = []; sid = 0
    for t in rng.choice(nT, min(n_benign, nT), replace=False):        # benign
        recs.append(make(int(t), 0, -1, None))
    for fam in [k for k in (1, 2, 3, 4, 6) if k in fam_ids]:           # single-shot families (retry to target)
        got = tries = 0
        while got < per_family and tries < per_family*25:
            tries += 1; t = int(rng.integers(nT))
            a = rng.choice(len(g.load_bus), min(4, len(g.load_bus)), replace=False)
            r = make(t, fam, -1, (a, 1 + rng.uniform(0.05, attack_intensity)))
            if r is not None: recs.append(r); got += 1
    if 5 in fam_ids:                                                   # ramp sequences to ~per_family records
        ramp_got = 0
        while ramp_got < per_family:
            t0 = int(rng.integers(nT - ramp_len))
            atk = rng.choice(len(g.load_bus), min(5, len(g.load_bus)), replace=False); rate = ramp_rate*rng.uniform(0.7, 1.0)
            seq = []
            for i in range(ramp_len):
                r = make(t0+i, 5, sid, (atk, 1 + rate*i))
                if r is None: break
                seq.append(r)
            if len(seq) >= 10: recs.extend(seq); ramp_got += len(seq); sid += 1

    out = out or os.path.join(CACHE_DIR, f"{name}.h5")
    _write(g, recs, out, split, seed)
    register_local(name, out, meta=dict(system=system, per_family=per_family, families=list(families),
                                        attack_intensity=attack_intensity, ramp_rate=ramp_rate, seed=seed))
    return out


def _write(g, recs, out, split, seed):
    T = len(recs); C, E = g.C, g.E
    arr = lambda i, dt: np.array([r[i] for r in recs], dt)
    node_x = np.stack([r[0] for r in recs]); node_m = np.stack([r[1] for r in recs])
    edge_x = np.stack([r[2] for r in recs]); edge_m = np.stack([r[3] for r in recs]); y = np.stack([r[4] for r in recs])
    fam = arr(5, np.int8); seq = arr(6, np.int32); tstep = arr(7, np.int32); gap = arr(8, np.uint8); st = arr(9, np.uint8)
    sp = _chrono_split(tstep, seq, split)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with h5py.File(out, "w") as f:
        f.attrs.update(dict(system=C, N=C, E=E, n_records=T, node_feat="V,P_inj,Q_inj,theta", edge_feat="P_from,Q_from",
                            families="0benign,1Ao,2Ad,3As,4Ar,5ramp,6LRA", lra_target_line=g._Ltgt, seed=seed))
        gg = f.create_group("graph"); gg.create_dataset("edge_index", data=g.ei); gg.create_dataset("edge_reactance", data=g.x_react)
        d = f.create_group("data"); ch = (min(128, T),)
        d.create_dataset("node_x", data=node_x, chunks=ch+node_x.shape[1:], compression="gzip", compression_opts=4)
        d.create_dataset("node_m", data=node_m, compression="gzip")
        d.create_dataset("edge_x", data=edge_x, chunks=ch+edge_x.shape[1:], compression="gzip", compression_opts=4)
        d.create_dataset("edge_m", data=edge_m, compression="gzip"); d.create_dataset("y", data=y, compression="gzip")
        for nm_, a in [("family", fam), ("seq_id", seq), ("timestep", tstep), ("gap", gap), ("stealthy", st), ("split", sp)]:
            d.create_dataset(nm_, data=a)


def _chrono_split(tstep, seq, frac):
    T = len(tstep)
    groups = [np.where(seq == s)[0] for s in np.unique(seq[seq >= 0])] + [np.array([i]) for i in np.where(seq < 0)[0]]
    groups.sort(key=lambda gp: int(tstep[gp].min()))
    sp = np.empty(T, np.int8); c = 0
    for gp in groups:
        f = c/T; sp[gp] = 0 if f < frac[0] else (1 if f < frac[0]+frac[1] else 2); c += len(gp)
    return sp
