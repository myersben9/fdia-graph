"""Write the frozen references that tests/test_frozen.py compares against.

    python tools/freeze_reference.py

Builds the same tiny IEEE-14 shard the test suite builds (seed 1, 80 benign, 12 per family), a
300-frame stream from the pool with the same seed, and the estimator and localizer scores on the
shard's test split, and stores them under tests/frozen/. Run it only when a change to the
generator or an estimator is intended (a data or method release); the readability series must
never need it. The references are written into a throwaway cache so the user's cache is untouched.
"""

import atexit
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
_CACHE = tempfile.mkdtemp(prefix="fdia_graph_freeze_")
os.environ["FDIA_GRAPH_CACHE"] = _CACHE
atexit.register(shutil.rmtree, _CACHE, ignore_errors=True)

import numpy as np  # noqa: E402

import fdia_graph as fg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "tests", "frozen")
os.makedirs(OUT, exist_ok=True)

# The single definition of what is frozen; the test imports these so the two can never drift.
from frozen_spec import SHARD_KW, STREAM_T, loc_scores, se_scores, shard_arrays, stream_arrays  # noqa: E402

shard_path = os.path.join(_CACHE, "tiny.h5")
fg.generate("ieee14", "tiny_ieee14", out=shard_path, **SHARD_KW)
arrays, attrs = shard_arrays(shard_path)
np.savez_compressed(os.path.join(OUT, "tiny_ieee14_shard.npz"), **arrays)
json.dump(attrs, open(os.path.join(OUT, "tiny_ieee14_shard_attrs.json"), "w"), indent=1, sort_keys=True)
print("shard:", len(arrays), "arrays,", len(attrs), "attrs")

from fdia_graph.generation import _load_states  # noqa: E402

X = _load_states(14, None)[:STREAM_T]
s = fg.generate_stream(14, states=X, seed=1)
np.savez_compressed(os.path.join(OUT, "ieee14_stream.npz"), **stream_arrays(s))
print("stream:", STREAM_T, "frames,", len(s["episodes"]), "episodes")

json.dump(
    se_scores("tiny_ieee14"), open(os.path.join(OUT, "tiny_ieee14_se.json"), "w"), indent=1, sort_keys=True
)
json.dump(
    loc_scores("tiny_ieee14"),
    open(os.path.join(OUT, "tiny_ieee14_localization.json"), "w"),
    indent=1,
    sort_keys=True,
)
print("wrote", OUT)
