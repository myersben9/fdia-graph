#!/usr/bin/env python
"""Build an AC operating-state pool for a SMALL IEEE case (30 or 57) for the cadence-control SE study.

The SDK's fdia_graph.profiles.generate_states hardcodes the supported cases to (14, 118, 300). Rather than
edit owner SDK code, this standalone builder reuses the EXACT same operating-state recipe (imported constants
K_DEFAULT / SIGMA_DEFAULT / CLIP_DEFAULT / JITTER_RHO and the AR(1) scale builder _ar1_scale) so the pools for
case30/case57 are byte-for-byte the same construction as the 14/118/300 pools: real NYISO scaling vector ->
clip(1 + k*S_t + AR(1) jitter) per bus -> AC power flow -> SE-consistent (without-shunt) injection state.

Writes release_v0.4.1/pool_ieee{C}.npz with key X = [T, C, 4] = [P_inj MW, Q_inj MVAr, |V| pu, theta deg].
Env: SE_CASE (30|57), POOL_N (default 36000, matches the reference pool size), FDIA_SEED (default 123).
"""
import os, sys, time
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandapower as pp
import pandapower.networks as pn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
# import the EXACT recipe pieces from the owner SDK so the small-case pool matches the 14/118/300 construction
from fdia_graph.profiles import _ar1_scale, K_DEFAULT, SIGMA_DEFAULT, CLIP_DEFAULT, JITTER_RHO

REL = os.path.join(HERE, "release_v0.4.1")
CASES = {30: pn.case30, 57: pn.case57}
CASE = int(os.environ.get("SE_CASE", "30"))
POOL_N = int(os.environ.get("POOL_N", "36000"))
SEED = int(os.environ.get("FDIA_SEED", "123"))
WORKERS = max(2, min(10, (os.cpu_count() or 4) - 2))


def _case_buses(key):
    base = CASES[key]()
    return np.unique(np.concatenate([base.load["bus"].to_numpy(), base.gen["bus"].to_numpy()]))


def _solve_states_chunk(key, sf_chunk):
    """Solve the AC operating state per precomputed per-bus scale-factor row. Mirrors profiles._solve_states_chunk
    exactly (same shunt subtraction for SE/BDD consistency); only the case lookup dict differs."""
    base = CASES[key]()
    pp.runpp(base)
    nodelist = sorted(base.bus.index)
    base_load_p = base.load["p_mw"].to_numpy().copy()
    base_load_q = base.load["q_mvar"].to_numpy().copy()
    base_gen_p = base.gen["p_mw"].to_numpy().copy()
    load_buses = base.load["bus"].to_numpy()
    gen_buses = base.gen["bus"].to_numpy()
    all_buses = np.unique(np.concatenate([load_buses, gen_buses]))
    pos = {int(b): i for i, b in enumerate(nodelist)}
    out = []
    for sf in sf_chunk:
        b2s = dict(zip(all_buses.tolist(), sf.tolist()))
        base.load["p_mw"] = base_load_p * np.array([b2s[int(b)] for b in load_buses])
        base.load["q_mvar"] = base_load_q * np.array([b2s[int(b)] for b in load_buses])
        base.gen["p_mw"] = base_gen_p * np.array([b2s[int(b)] for b in gen_buses])
        try:
            pp.runpp(base, init="flat", max_iteration=50, tolerance_mva=1e-6)
        except Exception:
            continue
        z = base.res_bus.reindex(nodelist)[["p_mw", "q_mvar", "vm_pu", "va_degree"]].to_numpy().copy()
        if len(base.res_shunt):
            for b, ps, qs in zip(base.shunt.bus.to_numpy(), base.res_shunt.p_mw.to_numpy(), base.res_shunt.q_mvar.to_numpy()):
                if int(b) in pos:
                    z[pos[int(b)], 0] -= ps
                    z[pos[int(b)], 1] -= qs
        out.append(z)
    return out


def main():
    S = np.load(os.path.join(REL, "profile_nyiso_S.npy"))[:POOL_N]
    nbus = len(_case_buses(CASE))
    SF = _ar1_scale(S, nbus, K_DEFAULT, SIGMA_DEFAULT, CLIP_DEFAULT, JITTER_RHO, SEED)
    print(f"[ieee{CASE}] building pool n={len(S)} nbus_scaled={nbus} workers={WORKERS} ...", flush=True)
    t0 = time.time()
    if WORKERS > 1 and len(S) >= WORKERS:
        import multiprocessing as mp
        sf_chunks = np.array_split(SF, WORKERS)
        with mp.Pool(WORKERS) as pool:
            parts = pool.starmap(_solve_states_chunk, [(CASE, c) for c in sf_chunks])
        out = [z for part in parts for z in part]
    else:
        out = _solve_states_chunk(CASE, SF)
    states = np.asarray(out, dtype=np.float64)
    outp = os.path.join(REL, f"pool_ieee{CASE}.npz")
    np.savez_compressed(outp, X=states)
    print(f"[ieee{CASE}] pool {states.shape} built in {time.time()-t0:.0f}s -> {outp}", flush=True)


if __name__ == "__main__":
    main()
