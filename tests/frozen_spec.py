"""What the frozen references contain, shared by tools/freeze_reference.py (writer) and
tests/test_frozen.py (reader), so the two cannot disagree about what is compared."""

from typing import Any

import numpy as np

TIMELINE_KW = dict(frames=1000, ramp_len=20, seed=3)  # the conftest tiny timeline, exactly


def file_arrays(path: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Every dataset in the HDF5 file, keyed by its path, and every attribute as JSON-safe values."""
    import h5py

    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {}

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            arrays[name.replace("/", "__")] = obj[()]
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


def se_scores(name: str) -> dict[str, dict[str, dict[str, float]]]:
    import fdia_graph as fg
    from fdia_graph.se import WLS, AdaptiveWeighting, JacobianWeighting, SubspacePrior

    train, test = fg.load(name, split="train"), fg.load(name, split="test")
    arms = {
        "wls": WLS(),
        "huber": AdaptiveWeighting(c=1.5),
        "prior+huber": SubspacePrior(rank_frac=0.2, reweight="huber", c=1.5),
        "jacobian": JacobianWeighting(),
    }
    return {k: jsonable(est.fit(train).score(test)) for k, est in arms.items()}


def loc_scores(name: str) -> dict[str, Any]:
    import fdia_graph as fg
    from fdia_graph.localization import DeltaThreshold, ResidualLocalizer, SwingThreshold

    train, test = fg.load(name, split="train"), fg.load(name, split="test")
    arms = {
        "swing": SwingThreshold(fa_target=0.01),
        "delta": DeltaThreshold(fa_target=0.01),
        "residual": ResidualLocalizer(fa_target=0.01),
    }
    return {k: jsonable(loc.fit(train).score(test)) for k, loc in arms.items()}
