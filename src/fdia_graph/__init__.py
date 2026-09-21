"""fdia-graph — load & generate ML-only dangerous FDIA localization datasets (realistic measurement graphs).

Quickstart
----------
    import fdia_graph as fg
    ds = fg.load("ieee118", split="train")          # auto-downloads + caches the latest shard
    loader = ds.loader(batch_size=64)               # ready-to-train PyTorch DataLoader
    for batch in loader:
        batch["node_x"], batch["edge_x"], batch["y"], batch["family"], ...

    # custom dataset with research knobs, then load it by name:
    fg.generate("ieee118", name="my_run", attacked_frac=0.5, attack_intensity=0.20, ramp_rate=0.003)
    ds = fg.load("my_run", split="train")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:  # the lazy names below, with their real signatures for type checkers
    from .engine import line_outage_candidates
    from .generation import generate
    from .profiles import fetch_profile, generate_states, load_profile
    from .streams import generate_stream, load_stream, windows
    from .torch_data import pyg_stream, torch_windows

# Re-exports so users write `fg.FdiaGraph` / `fg.load(...)` instead of reaching into submodules.
# FdiaGraph: torch Dataset over one .h5 shard; FAMILIES: attack-family names/ids; STEALTHY_FAMILIES: the
# BDD-evading subset (hard cases, e.g. Aq).
from .dataset import FAMILIES, STEALTHY_FAMILIES, FdiaGraph, family_ids

# download: ensure_local (resolved spec -> local .h5 path, fetching+caching if absent).
from .download import ensure_local

# registry = dataset "version control": list_datasets (known built-in + local), register_local (name a local
# dataset), resolve ((name, release) -> download spec).
from .registry import list_datasets, resolve
from .registry import register_local as register_local  # re-exported for `fg.register_local`

__version__ = "0.18.0"
# Public API for `from fdia_graph import *`; register_local/resolve/ensure_local stay out (internal plumbing).
__all__ = [
    "load",
    "generate",
    "generate_stream",
    "load_stream",
    "windows",
    "pyg_stream",
    "torch_windows",
    "load_profile",
    "fetch_profile",
    "generate_states",
    "line_outage_candidates",
    "list_datasets",
    "FdiaGraph",
    "FAMILIES",
    "STEALTHY_FAMILIES",
]


def load(
    name: str,
    split: Optional[str] = None,
    families: Optional[Sequence[Union[str, int]]] = None,
    include_gaps: bool = False,
    heldout: bool = False,
    format: str = "torch",
    release: Optional[str] = None,
    units: str = "physical",
    preload: bool = False,
    order: str = "time",
    seed: int = 0,
) -> FdiaGraph:
    """Load a dataset by name (built-in files auto-download; locally generated ones load from disk).

    name        : "ieee14/30/57/89/118/145/200/300" (transmission ladder), or a locally-generated name.
    split       : None (all) | "train" | "val" | "test"  (chronological 60/20/20).
    families    : optional subset, e.g. ["Aq","At","Al"] or [1,5,6].
    include_gaps: keep physics non-convergence NA rows of a v0.7.2 shard (default False; timelines have none).
    heldout     : unseen-attack protocol — exclude As/Ar from train/val (Boyaci et al. 2022).
    format      : "torch" (dict batches) | "pyg" (torch_geometric Data).
    release     : dataset VERSION. None -> the pinned release; an explicit tag e.g. "v0.7.2" -> that exact
                  version, for reproducible experiments.
    units       : "physical" -> [V p.u., P_inj MW, Q_inj MVAr, theta deg] (as stored; human-readable for plots);
                  "pu" -> everything per-unit on baseMVA with theta in radians (ML/physics). Same file either way.
    preload     : read the whole selected split into RAM once (~350MB for an ieee118 split) so .loader()
                  epochs skip per-record HDF5 overhead — much faster training loops.
    order       : "time" (default) keeps the file order, chronological on a timeline, so `ds.windows(W)` and
                  `ds.episodes` work; "random" is the same records in a permutation fixed by `seed`, the
                  record table two people get identically for the same seed (`.loader(shuffle=True)`
                  still reshuffles per epoch).
    seed        : the permutation seed for order="random".
    """
    # resolve() -> download spec, ensure_local() -> on-disk .h5 path (fetching if needed; local datasets
    # short-circuit to their file).
    from .dataset.base import check_order, check_split, check_units

    check_units(units)  # a wrong argument fails before any download
    check_split(split)
    check_order(order)
    if families is not None:
        family_ids(families)
    path = ensure_local(resolve(name, release=release))
    # Thin factory: the Dataset applies split/families/gaps/heldout and the export format lazily.
    return FdiaGraph(
        path,
        split=split,
        families=families,
        include_gaps=include_gaps,
        heldout=heldout,
        format=format,
        units=units,
        preload=preload,
        order=order,
        seed=seed,
    )


# The generators, profiles, stream and torch helpers pull in pandapower, torch or torch_geometric,
# so they are imported on first use rather than at `import fdia_graph`. Each name resolves to the
# real function (its own docstring and signature), not a wrapper that could drift from it.
# `from fdia_graph import *` resolves all of them (it binds every name in __all__), which loads
# those modules but still no optional dependency: each imports pandapower, torch or torch_geometric
# inside the functions that need them (tests/test_namespace.py checks this in a fresh interpreter).
_LAZY = {
    "generate": ".generation",
    "generate_stream": ".streams",
    "load_stream": ".streams",
    "windows": ".streams",
    "pyg_stream": ".torch_data",
    "torch_windows": ".torch_data",
    "load_profile": ".profiles",
    "fetch_profile": ".profiles",
    "generate_states": ".profiles",
    "line_outage_candidates": ".engine",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name], __name__), name)
        globals()[name] = value  # resolved once; later lookups skip __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
