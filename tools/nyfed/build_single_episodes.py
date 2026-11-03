"""The single-episode files for studying one attack by hand: one Am on an interface per file, over the
whole day it falls in (288 five-minute scans), nothing else attacked. Each file is a normal timeline
(`fdia_graph.nyfed.io.open_timeline`): `z` with the attack, `z_benign` the same noisy scans without it,
the true states and the labels.

    python tools/nyfed/build_single_episodes.py OUT_DIR [--pool out/pool_2019_v3.h5]
                                                [--interface SPR/DUN-SOUTH] [--onset 2019-07-05T10:30:00 ...]

Without `--onset` it writes the four study days below. Run from the folder whose `data/` holds the inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fdia_graph.nyfed.build import production_builder

# The four study onsets (pool clock) and why each was picked, from the Am headroom survey on SPR/DUN-SOUTH
STUDY = {
    "2019-07-30T12:55:00": "smallest push (150 MW)",
    "2019-07-05T10:30:00": "typical push (about 390 MW)",
    "2019-06-27T19:25:00": "largest push (433 MW)",
    "2019-07-18T08:40:00": "most tampered meters",
}
DAY = 288  # five-minute scans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="out/single_episode")
    ap.add_argument("--pool", default="out/pool_2019_v3.h5")
    ap.add_argument("--interface", default="SPR/DUN-SOUTH")
    ap.add_argument("--onset", nargs="*", default=None, help="onsets as YYYY-MM-DDTHH:MM:SS (pool clock)")
    args = ap.parse_args()
    picks = {t: "given" for t in args.onset} if args.onset else STUDY
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    b = production_builder(args.pool, ("Am",), interfaces=(args.interface,), workers=1, survey=0)
    t = b.pool.time.astype("datetime64[s]")
    summary = {}
    for when, why in picks.items():
        onset = int(np.flatnonzero(t == np.datetime64(when))[0])
        lo = int(np.flatnonzero(t == np.datetime64(when[:10] + "T00:00:00"))[0])
        name = f"sd_{args.interface.split('/')[-1].lower().replace('-', '_')}_{when[:10]}"
        print(f"{name}: onset {when} ({why}), frames {lo}..{lo + DAY}", flush=True)
        s = b.build_episode(out / f"{name}.h5", onset, "Am", lo, lo + DAY)
        s.update(interface=args.interface, onset_time=when, why=why)
        it = b.families["Am"].interfaces[0]
        r = onset + s["length"] // 2 - 1  # the last rise scan, where the push peaks
        s.update(
            real_limit_mw=float(it.limit[r]),
            real_flow_mw=float(it.flow[r]),
            headroom_mw=float(it.limit[r] - it.flow[r]),
        )
        print("  ", s, flush=True)
        summary[name] = s
        (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
