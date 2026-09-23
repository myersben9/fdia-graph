#!/usr/bin/env python
"""Build the v0.8.1 data release: the v0.8.0 recipe re-run with the corrected generator.

v0.8.0's engine recovered a scan's load at a bus holding both a load and a generator from the base
generation although the pools scale generation with load, so the stealthy families there moved the
wrong load (fdia-graph 0.19). The operating-point pools are unchanged: v0.8.0's `pool_ieee{C}.h5` is
copied, and `fg.generate` walks it into `timeline_ieee{C}.h5` with the same default knobs
(attacked_frac 0.5, every family incl. Am, 60-frame ramps, one-frame Ad/As/Ar draws, seed 123).
Resumable: a finished file is skipped. Writes `manifest.json` with sha256 and size for the
registry and the upload.

    LADDER=14,30 python examples/_build_timelines_v081.py
"""

import hashlib
import json
import os
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import h5py  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fdia_graph as fg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
POOLS = os.path.join(HERE, "release_v0.8.0")
OUT = os.path.join(HERE, "release_v0.8.1")
LADDER = [int(x) for x in os.environ.get("LADDER", "14,30,57,89,118,145,200,300").split(",")]
SEED = int(os.environ.get("SEED", "123"))
FRAMES = int(os.environ.get("FRAMES", "72000"))


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def pool_h5(C: int) -> str:
    """The v0.8.0 pool, copied unchanged (the fix is in the generator, not the operating points)."""
    out = os.path.join(OUT, f"pool_ieee{C}.h5")
    if not os.path.exists(out):
        shutil.copyfile(os.path.join(POOLS, f"pool_ieee{C}.h5"), out)
        print(f"[ieee{C}] pool copied from v0.8.0", flush=True)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    manifest_path = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    for C in LADDER:
        t0 = time.time()
        pool = pool_h5(C)
        out = os.path.join(OUT, f"timeline_ieee{C}.h5")
        if os.path.exists(out) and f"timeline_ieee{C}" in manifest:
            print(f"[ieee{C}] timeline exists, skip", flush=True)
            continue
        print(f"[ieee{C}] walking {FRAMES} frames ...", flush=True)
        fg.generate(C, f"ieee{C}_v081", states=pool, seed=SEED, out=out)
        with h5py.File(out, "r") as f:
            T, frac, n_ep = int(f.attrs["T"]), float(f.attrs["attacked_frac"]), int(f.attrs["n_episodes"])
        print(
            f"[ieee{C}] {T} frames, attacked {100 * frac:.0f}%, {n_ep} episodes, "
            f"{os.path.getsize(out) / 1e6:.0f} MB in {(time.time() - t0) / 60:.1f} min",
            flush=True,
        )
        # re-read before updating: several systems may build in parallel processes sharing the manifest
        manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
        for name in (f"pool_ieee{C}", f"timeline_ieee{C}"):
            p = os.path.join(OUT, f"{name}.h5")
            manifest[name] = {
                "file": f"{name}.h5",
                "sha256": sha256(p),
                "mb": round(os.path.getsize(p) / 1e6, 1),
            }
        json.dump(manifest, open(manifest_path, "w"), indent=2)
    print("\n[all] done:\n" + json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
