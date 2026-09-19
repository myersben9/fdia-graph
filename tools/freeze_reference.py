"""Write the frozen references that tests/test_frozen.py compares against.

    python tools/freeze_reference.py

Builds the same tiny IEEE-14 timeline the test suite builds (frozen_spec.TIMELINE_KW) and the
estimator and localizer scores on its test split, and stores them under tests/frozen/. Run it only when a change to the
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
from frozen_spec import TIMELINE_KW, file_arrays, loc_scores, se_scores  # noqa: E402

path = os.path.join(_CACHE, "tiny.h5")
fg.generate("ieee14", "tiny_ieee14", out=path, **TIMELINE_KW)
arrays, attrs = file_arrays(path)
np.savez_compressed(os.path.join(OUT, "tiny_ieee14_timeline.npz"), **arrays)
json.dump(attrs, open(os.path.join(OUT, "tiny_ieee14_timeline_attrs.json"), "w"), indent=1, sort_keys=True)
print("timeline:", len(arrays), "arrays,", len(attrs), "attrs")

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
