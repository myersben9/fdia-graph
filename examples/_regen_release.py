#!/usr/bin/env python
"""Regenerate the 14/118/300 datasets for the new release, end to end with the fixed generator.

Pipeline: fetch real NYISO load upsampled to 1-minute -> per-system AC operating-state pool (built in
PARALLEL across worker processes) -> attack dataset (Aq/At/Al re-solve with pinned dispatch + AGC, theta
metered like V, generalized ramp, temporal_delta). Writes the pool and shard for each system into
release_v0.4.0/. Seeded (123) so it is reproducible. Guarded with __main__ for multiprocessing on Windows.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fdia_graph as fg
from fdia_graph.profiles import fetch_profile, generate_states

# Output dir is overridable so a new release (e.g. v0.4.1) can be built into a FRESH folder without disturbing the
# shards the current training/report jobs are still reading. Reuses any pool_*.npz already present in OUT (the benign
# operating states are unchanged by the attack-side / metadata fixes), so only attack generation re-runs.
OUT = os.environ.get("FDIA_RELEASE_DIR", os.path.join(os.path.dirname(__file__), "release_v0.4.0"))
# ~25 days of NYISO load UPSAMPLED TO 1-MINUTE -> ~36k distinct 1-min operating points (reference-paper
# resolution; tight next-step prediction so the temporal channel works). paper-scale, 50/50-balanced.
POOL_N = 36000; PER_FAMILY = 6000; N_BENIGN = 36000        # 6 families x 6000 = 36000 attacked == 36000 benign
WORKERS = max(2, min(10, (os.cpu_count() or 4) - 2))      # parallel pool build: each power flow is independent


def main():
    os.makedirs(OUT, exist_ok=True)
    # Reuse the cached scaling vector when present (it's only used to BUILD pools; if pools already exist it isn't
    # needed at all). Avoids a redundant NYISO network fetch and keeps the regen reproducible/offline.
    S_path = os.path.join(OUT, "profile_nyiso_S.npy")
    if os.path.exists(S_path):
        S = np.load(S_path); print(f"[profile] reusing cached S ({len(S)} 1-min samples)", flush=True)
    else:
        print(f"[profile] fetching NYISO load 2024-01-01..2024-01-30, upsampled to 1-minute  (workers={WORKERS})", flush=True)
        S = fetch_profile("nyiso", "2024-01-01", "2024-01-30", out=os.path.join(OUT, "profile_nyiso.csv"), resample_min=1)
        np.save(S_path, S)
        print(f"[profile] {len(S)} 1-min samples", flush=True)

    for C in (14, 118, 300):
        t0 = time.time()
        shard = os.path.join(OUT, f"ml_only_ieee{C}.h5"); pool_p = os.path.join(OUT, f"pool_ieee{C}.npz")
        if os.path.exists(shard):                          # resume: shard already built
            print(f"[ieee{C}] DONE (already present) -> {shard}", flush=True); continue
        if os.path.exists(pool_p):                         # resume: reuse an already-built pool
            print(f"[ieee{C}] reusing existing pool", flush=True); states = np.load(pool_p)["X"]
        else:
            print(f"[ieee{C}] building operating-state pool (n={POOL_N}, {WORKERS} workers) ...", flush=True)
            states = generate_states(C, S, n=POOL_N, seed=123, workers=WORKERS)
            np.savez_compressed(pool_p, X=states)
            print(f"[ieee{C}] pool built in {time.time()-t0:.0f}s", flush=True)
        print(f"[ieee{C}] pool {states.shape}; generating dataset (per_family={PER_FAMILY}, n_benign={N_BENIGN}) ...", flush=True)
        fg.generate(C, name=f"ieee{C}_v040", states=states, per_family=PER_FAMILY, n_benign=N_BENIGN, out=shard)
        print(f"[ieee{C}] DONE in {time.time()-t0:.0f}s -> {shard}", flush=True)

    print("[all] regeneration complete", flush=True)


if __name__ == "__main__":
    main()
