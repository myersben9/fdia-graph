"""fdia-graph: load and generate datasets for locating false data injection attacks on a power grid.

Quickstart:
    import fdia_graph as fg
    ds = fg.load("ieee118", split="train")          # downloads and caches the pinned timeline
    for batch in ds.loader(batch_size=64):          # a PyTorch DataLoader
        batch["node_x"], batch["edge_x"], batch["y"], batch["family"], ...
    fg.generate("ieee118", name="my_run", attacked_frac=0.5, ramp_rate=0.003)   # your own dataset
    ds = fg.load("my_run", split="train")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:  # the lazy names below, with their real signatures for type checkers
    from .data.ieee_pool import generate_states
    from .data.profiles import fetch_profile, load_profile
    from .generation import generate

# Re-exports, so users write `fg.FdiaGraph` and `fg.load(...)` instead of reaching into submodules.
from .constants.families import FAMILIES, STEALTHY_FAMILIES
from .dataset import FdiaGraph
from .dataset.base import family_ids

# ensure_local: a resolved spec -> the local .h5 path, downloaded and cached if absent
from .download import ensure_local

# the registry of built-in and local datasets; resolve: (name, release) -> a download spec
from .registry import list_datasets, resolve
from .registry import register_local as register_local  # re-exported for `fg.register_local`

__version__ = "0.21.0"
# the public API of `from fdia_graph import *`; register_local, resolve and ensure_local stay out
__all__ = [
    "load",
    "generate",
    "load_profile",
    "fetch_profile",
    "generate_states",
    "list_datasets",
    "FdiaGraph",
    "FAMILIES",
    "STEALTHY_FAMILIES",
]


def load(
    name: str,
    split: Optional[str] = None,
    families: Optional[Sequence[Union[str, int]]] = None,
    heldout: bool = False,
    format: str = "torch",
    release: Optional[str] = None,
    units: str = "physical",
    preload: bool = False,
    order: str = "time",
    seed: int = 0,
) -> FdiaGraph:
    """Load a dataset by name: "ieee14" to "ieee300" (downloaded and cached), or a name you generated.

    split     None (all), "train", "val" or "test" (chronological 60/20/20)
    families  a subset, e.g. ["At", "Am"] or [5, 7] (old releases also carry Aq, Ad, As, Ar, Al)
    heldout   the unseen-attack protocol: As and Ar are kept out of train and val [BOY22]
    format    "torch" (dict batches) or "pyg" (torch_geometric Data)
    release   None: the pinned data release; a tag such as "v0.8.1": that exact release
    units     "physical": [V pu, P_inj MW, Q_inj MVAr, theta deg], as stored; "pu": per unit, theta in rad
    preload   read the whole split into RAM once (~350 MB for an ieee118 split), so epochs run faster
    order     "time": the file order, which `ds.windows(W)` and `ds.episodes` need; "random": the same
              records in a permutation fixed by `seed`
    """
    from .models.config import LoadOptions

    options = LoadOptions(split, units, order, format)  # checked before any download
    if families is not None:
        family_ids(families)
    path = ensure_local(resolve(name, release=release))
    return FdiaGraph(
        path,
        split=options.split,
        families=families,
        heldout=heldout,
        format=options.format,
        units=options.units,
        preload=preload,
        order=options.order,
        seed=seed,
    )


# The generator and the profile helpers pull in pandapower, torch or torch_geometric, so they are
# imported on first use, not at `import fdia_graph`. Each name is the real function, not a wrapper.
# `from fdia_graph import *` loads their modules but no optional dependency: each imports its
# optional packages inside the functions that need them (tests/test_namespace.py checks this).
def __getattr__(name: str) -> object:
    if name == "generate":
        from .generation import generate

        return generate
    if name in ("load_profile", "fetch_profile"):
        from .data import profiles

        return getattr(profiles, name)
    if name == "generate_states":
        from .data.ieee_pool import generate_states

        return generate_states
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
