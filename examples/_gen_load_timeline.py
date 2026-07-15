#!/usr/bin/env python
"""Stage load-over-time traces for the report's attack-progression figure — from REAL records only.

NO synthetic/hand-drawn data. For one real load bus we pull its benign injection time series straight from
the benign records (one per timestep), then for each family we splice in the ACTUAL attacked reading from a
real record of that family that targets the same bus (for At, the real ramp sequence's trajectory). Every
point plotted is a value that exists in the shard.

Runs for ALL THREE systems so the report can show an attack-progression page per grid; keys in the output npz
are prefixed ieee{C}_ (e.g. ieee118_Aq_atk). Saves results/load_timeline.npz (+ one CSV sidecar per system).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, h5py
import pandapower.networks as pn
CASES = {14: pn.case14, 118: pn.case118, 300: pn.case300}
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
SHDIR = os.environ.get("FDIA_LOCAL_SHARDS", os.path.join(HERE, "release_v0.4.0"))
os.makedirs(os.path.join(RES, "sidecars"), exist_ok=True)

W = 80
FAMc = {1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 6: "Al"}     # single-shot families (At/ramp handled separately)


def build_system(C):
    """Return a dict of real load traces for IEEE-C, keyed 'benign_ben','Aq_atk',... plus 'bus','W'.
    Returns None if the shard is missing."""
    shard = os.path.join(SHDIR, f"ml_only_ieee{C}.h5")
    if not os.path.exists(shard):
        print(f"ieee{C}: no shard, skip", flush=True); return None
    with h5py.File(shard, "r") as f:
        d = f["data"]
        nx = d["node_x"][:, :, 1]        # P_inj per record per bus (MW)
        fam = d["family"][:]; ts = d["timestep"][:]; seq = d["seq_id"][:]; y = d["y"][:]

    # benign timeline: one benign record per timestep -> map timestep -> record index
    ben = np.where(fam == 0)[0]
    order = np.argsort(ts[ben]); ben_sorted = ben[order]; ben_ts_sorted = ts[ben][order]
    ts_to_ben = {int(t): int(r) for t, r in zip(ben_ts_sorted, ben_sorted)}

    # Pick the most-attacked PURE-LOAD bus (has load, but NO co-located generator/slack). This keeps the plotted
    # injection a clean, positive load trace: on a bus that also has generation the net injection is dominated by
    # (and can be sign-flipped by) the generator, which buries the load attack and confuses the axis. Falls back to
    # the most-attacked bus overall if a case somehow has no pure-load bus that is attacked.
    net = CASES[C]()
    gen_buses = set(net.gen.bus.values) | set(net.ext_grid.bus.values)
    if len(net.sgen): gen_buses |= set(net.sgen.bus.values)
    N = y.shape[1]
    att = y.sum(0)
    pure_load = np.array([bb for bb in range(N) if bb not in gen_buses and att[bb] > 0], dtype=int)
    b = int(pure_load[np.argmax(att[pure_load])]) if len(pure_load) else int(np.argmax(att))

    def benign_series(t0, T):
        """Real benign P_inj for bus b over timesteps [t0, t0+T)."""
        return np.array([nx[ts_to_ben[t], b] if t in ts_to_ben else np.nan for t in range(t0, t0 + T)])

    out = {}
    t_start = int(np.median(ben_ts_sorted)) - W // 2
    out["benign_ben"] = benign_series(t_start, W); out["benign_atk"] = out["benign_ben"].copy(); out["benign_hits"] = np.array([], int)

    # single-shot families: real attacked reading spliced onto bus b's real benign window at its real timestep
    for code, name in FAMc.items():
        recs = np.where((fam == code) & (y[:, b] > 0))[0]
        if len(recs) == 0: continue
        rr = recs[np.argsort(ts[recs])]
        t0 = int(ts[rr[len(rr) // 2]]) - W // 2
        benw = benign_series(t0, W); atkw = benw.copy(); hit_idx = []
        for r in rr:
            i = int(ts[r]) - t0
            if 0 <= i < W: atkw[i] = nx[r, b]; hit_idx.append(i)      # REAL attacked value at its REAL timestep
        out[f"{name}_ben"] = benw; out[f"{name}_atk"] = atkw; out[f"{name}_hits"] = np.array(hit_idx, int)

    # At (ramp, family 5): the real trajectory of one sequence that targets bus b
    at_seqs = np.unique(seq[(fam == 5) & (y[:, b] > 0) & (seq >= 0)])
    if len(at_seqs):
        s = int(at_seqs[len(at_seqs) // 2]); rs = np.where(seq == s)[0]; rs = rs[np.argsort(ts[rs])]
        t0 = int(ts[rs[0]]) - 10; benw = benign_series(t0, W); atkw = benw.copy(); hit_idx = []
        for r in rs:
            i = int(ts[r]) - t0
            if 0 <= i < W: atkw[i] = nx[r, b]; hit_idx.append(i)      # REAL ramp reading at each REAL scan
        out["At_ben"] = benw; out["At_atk"] = atkw; out["At_hits"] = np.array(hit_idx, int)

    out["bus"] = b; out["W"] = W
    print(f"ieee{C}: bus {b} (attacked {int(y.sum(0)[b])}x), families {[k[:-4] for k in out if k.endswith('_atk')]}", flush=True)
    return out


packed = {}
for C in (14, 118, 300):
    sysout = build_system(C)
    if sysout is None: continue
    for k, v in sysout.items():
        packed[f"ieee{C}_{k}"] = v
    # per-system CSV sidecar (restyle-without-rerun)
    fams = [k[:-4] for k in sysout if k.endswith("_atk")]
    rows = ["scan," + ",".join(f"{fm}_benign,{fm}_attacked" for fm in fams)] + \
           ["\n".join([f"{i}," + ",".join(f"{sysout[f'{fm}_ben'][i]:.3f},{sysout[f'{fm}_atk'][i]:.3f}" for fm in fams) for i in range(W)])]
    open(os.path.join(RES, "sidecars", f"load_timeline_ieee{C}.csv"), "w").write("\n".join(rows))

np.savez(os.path.join(RES, "load_timeline.npz"), **packed)
print(f"[done] load_timeline.npz  ({len([k for k in packed if k.endswith('_bus')])} systems)", flush=True)
