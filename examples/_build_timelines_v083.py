#!/usr/bin/env python
"""Build the v0.8.3 data release: single-snapshot Aq and Al, and temporal features from observed frames.

From fdia-graph 0.20 every Aq and Al episode is one frame, and the swing feature's scale is the recent
change of the observed frames instead of the noiseless pool, so every stored feature is a function of
the measurements alone. The operating-point pools are unchanged: v0.8.1's `pool_ieee{C}.h5` (identical
in v0.8.2) is copied, and `fg.generate` walks it into `timeline_ieee{C}.h5` with the default knobs
(attacked_frac 0.5, every family incl. Am, 60-frame ramps, one-frame Aq/Ad/As/Ar/Al draws, seed 123).
Resumable: a finished file is skipped. Each system writes its own manifest fragment (sha256 and size);
a final MERGE=1 run combines them into `manifest.json` for the registry and the upload. RELEASE_DIR
overrides where the release folders live (default: next to this script), so a worktree can build into
the main tree.

    LADDER=14,30 python examples/_build_timelines_v083.py
    MERGE=1 python examples/_build_timelines_v083.py   # once every worker is done: manifest.json
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")
import h5py  # noqa: E402

# Each worker registers its outputs in its own cache index: `fg.generate` writes the local-datasets
# index with a read-modify-write, which parallel workers sharing one cache would interleave.
os.environ.setdefault("FDIA_GRAPH_CACHE", os.path.join(tempfile.gettempdir(), f"fdia_build_{os.getpid()}"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fdia_graph as fg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("RELEASE_DIR", HERE)
POOLS = os.path.join(BASE, "release_v0.8.1")
OUT = os.path.join(BASE, "release_v0.8.3")
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
    """The v0.8.1 pool, copied unchanged (the change is in the generator, not the operating points).
    An existing copy is reused only when it is byte-identical to the source."""
    out = os.path.join(OUT, f"pool_ieee{C}.h5")
    src = os.path.join(POOLS, f"pool_ieee{C}.h5")
    if not os.path.exists(out) or sha256(out) != sha256(src):
        shutil.copyfile(src, out + ".part")  # an interrupted copy
        os.replace(out + ".part", out)  # never leaves a partial pool a rerun would trust
        print(f"[ieee{C}] pool copied from v0.8.1", flush=True)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for C in LADDER:
        t0 = time.time()
        pool = pool_h5(C)
        out = os.path.join(OUT, f"timeline_ieee{C}.h5")
        part = os.path.join(
            OUT, f"manifest_ieee{C}.json"
        )  # one fragment per system: parallel workers never share a file
        if _finished(C, out, part):
            print(f"[ieee{C}] timeline exists with these inputs, skip", flush=True)
            continue
        print(f"[ieee{C}] walking {FRAMES} frames ...", flush=True)
        fg.generate(C, f"ieee{C}_v083", states=pool, seed=SEED, out=out, frames=FRAMES)
        with h5py.File(out, "r") as f:
            T, frac, n_ep = int(f.attrs["T"]), float(f.attrs["attacked_frac"]), int(f.attrs["n_episodes"])
        print(
            f"[ieee{C}] {T} frames, attacked {100 * frac:.0f}%, {n_ep} episodes, "
            f"{os.path.getsize(out) / 1e6:.0f} MB in {(time.time() - t0) / 60:.1f} min",
            flush=True,
        )
        entry: dict = {"inputs": _inputs(C)}
        for name in (f"pool_ieee{C}", f"timeline_ieee{C}"):
            p = os.path.join(OUT, f"{name}.h5")
            entry[name] = {
                "file": f"{name}.h5",
                "sha256": sha256(p),
                "mb": round(os.path.getsize(p) / 1e6, 1),
            }
        with open(part + ".tmp", "w") as fh:  # atomic: a merging worker never reads half a fragment
            json.dump(entry, fh, indent=2)
        os.replace(part + ".tmp", part)
    print("[all] done; run once more with MERGE=1 after every worker has finished", flush=True)


def _inputs(C: int) -> dict:
    """The generation inputs system C was built with, the source pool by content; a resume skips a
    system only when they match."""
    return {"frames": FRAMES, "seed": SEED, "pool_sha256": sha256(os.path.join(POOLS, f"pool_ieee{C}.h5"))}


def _finished(C: int, out: str, part: str) -> bool:
    """A timeline and its fragment exist and the fragment records the current inputs."""
    if not (os.path.exists(out) and os.path.exists(part)):
        return False
    with open(part) as fh:
        return json.load(fh).get("inputs") == _inputs(C)


def merge_manifest() -> dict:
    """manifest.json from every finished system's fragment, written by a single MERGE=1 run after
    the parallel workers are done, so no worker ever writes a file another one reads."""
    missing = [C for C in LADDER if not os.path.exists(os.path.join(OUT, f"manifest_ieee{C}.json"))]
    if missing:  # a partial manifest would publish a release without these systems
        raise SystemExit(f"no finished fragment for ieee{missing}; build them before merging")
    manifest: dict = {}
    for f in sorted(os.listdir(OUT)):
        if f.startswith("manifest_ieee") and f.endswith(".json"):
            with open(os.path.join(OUT, f)) as fh:
                fragment = json.load(fh)
            fragment.pop("inputs", None)  # build bookkeeping, not an asset
            manifest.update(fragment)
    fd, tmp = tempfile.mkstemp(
        dir=OUT, suffix=".json"
    )  # atomic: two workers finishing together never interleave
    with os.fdopen(fd, "w") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp, os.path.join(OUT, "manifest.json"))
    return manifest


if __name__ == "__main__":
    if os.environ.get("MERGE"):
        print(json.dumps(merge_manifest(), indent=2))
    else:
        main()
