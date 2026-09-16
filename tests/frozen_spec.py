"""What the frozen references contain, shared by tools/freeze_reference.py (writer) and
tests/test_frozen.py (reader), so the two cannot disagree about what is compared."""

import hashlib
from typing import Any, Dict, Tuple

import numpy as np

SHARD_KW = dict(per_family=12, n_benign=80, seed=1)  # the conftest tiny shard, exactly
STREAM_T = 300
CLEAN_ROWS = 256  # rows of the clean pool kept in the reference; the rest is covered by a hash


def shard_arrays(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Every dataset in the HDF5 file, keyed by its path, and every attribute as JSON-safe values."""
    import h5py

    arrays: Dict[str, np.ndarray] = {}
    attrs: Dict[str, Any] = {}

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            data = obj[()]
            key = name.replace("/", "__")
            if name.startswith("clean/"):
                # The clean group holds the WHOLE operating-point pool (thousands of timesteps, most
                # of the file). Freeze its first rows for the tolerance comparison and a hash of all
                # of it for the strict comparison, so the reference stays small.
                attrs[f"{name}::sha256"] = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
                data = data[:CLEAN_ROWS]
            arrays[key] = data
        for k, v in obj.attrs.items():
            attrs[f"{name}::{k}"] = jsonable(v)

    with h5py.File(path, "r") as f:
        for k, v in f.attrs.items():
            attrs[f"::{k}"] = jsonable(v)
        f.visititems(visit)
    return arrays, attrs


def jsonable(v: Any) -> Any:
    """Plain Python values for JSON: numpy scalars and arrays, bytes, and nested dicts."""
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf8", "replace")
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, float) and v != v:  # NaN never equals itself, so it would fail every comparison
        return "nan"
    return v


def stream_arrays(s: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out = {k: np.asarray(v) for k, v in s.items() if isinstance(v, np.ndarray)}
    out["episode_onset"] = np.array([e["onset"] for e in s["episodes"]], np.int64)
    out["episode_length"] = np.array([e["length"] for e in s["episodes"]], np.int64)
    out["episode_family"] = np.array([e["family"] for e in s["episodes"]], np.int64)
    out["episode_buses"] = np.array([",".join(map(str, e["buses"])) for e in s["episodes"]])
    return out


def se_scores(name: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    import fdia_graph as fg
    from fdia_graph.se import AdaptiveWeighting, JacobianWeighting, SubspacePrior, WLS

    train, test = fg.load(name, split="train"), fg.load(name, split="test")
    arms = {
        "wls": WLS(),
        "huber": AdaptiveWeighting(c=1.5),
        "prior+huber": SubspacePrior(rank_frac=0.2, reweight="huber", c=1.5),
        "jacobian": JacobianWeighting(),
    }
    return {k: jsonable(est.fit(train).score(test)) for k, est in arms.items()}


def loc_scores(name: str) -> Dict[str, Any]:
    import fdia_graph as fg
    from fdia_graph.localization import DeltaThreshold, ResidualLocalizer, SwingThreshold

    train, test = fg.load(name, split="train"), fg.load(name, split="test")
    arms = {
        "swing": SwingThreshold(fa_target=0.01),
        "delta": DeltaThreshold(fa_target=0.01),
        "residual": ResidualLocalizer(fa_target=0.01),
    }
    return {k: jsonable(loc.fit(train).score(test)) for k, loc in arms.items()}
