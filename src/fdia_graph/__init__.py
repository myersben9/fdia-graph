"""fdia-graph — load & generate ML-only dangerous FDIA localization datasets (realistic measurement graphs).

Quickstart
----------
    import fdia_graph as fg
    ds = fg.load("ieee118", split="train")          # auto-downloads + caches the latest shard
    loader = ds.loader(batch_size=64)               # ready-to-train PyTorch DataLoader
    for batch in loader:
        batch["node_x"], batch["edge_x"], batch["y"], batch["family"], ...

    # custom dataset with research knobs, then load it by name:
    fg.generate("ieee118", name="my_run", per_family=5000, attack_intensity=0.20, ramp_rate=0.003)
    ds = fg.load("my_run", split="train")
"""
# --- Re-exports: pull the workhorse names up to the package top level so users
# --- can write `fg.FdiaGraph` / `fg.load(...)` instead of reaching into submodules.
# FdiaGraph        : the torch Dataset wrapper over one .h5 shard (see dataset.py).
# FAMILIES         : the full ordered tuple of attack-family names/ids the datasets use.
# STEALTHY_FAMILIES: the subset that evades bad-data detection (the hard cases, e.g. Ao).
from .dataset import FdiaGraph, FAMILIES, STEALTHY_FAMILIES
# registry.py is the dataset "version control":
# list_datasets  : enumerate the known built-in + locally-registered datasets.
# register_local : record a locally-generated dataset under a name so load() can find it.
# resolve        : turn (name, release) into a concrete download spec (release tag + filename).
from .registry import list_datasets, register_local, resolve
# download.py handles the network/cache side:
# ensure_local   : given a resolved spec, return a local .h5 path, fetching+caching if absent.
from .download import ensure_local

# Package version string, surfaced as fdia_graph.__version__ (standard convention).
__version__ = "0.1.0"
# The public surface: what `from fdia_graph import *` exposes and what we advertise as stable API.
# Note register_local/resolve/ensure_local are intentionally NOT here — they're internal plumbing.
__all__ = ["load", "generate", "list_datasets", "FdiaGraph", "FAMILIES", "STEALTHY_FAMILIES"]


def load(name, split=None, families=None, include_gaps=False, heldout=False, format="torch", release=None):
    """Load a dataset by name (built-in shard auto-downloads; local generated ones load from disk).

    name        : "ieee14"|"ieee118"|"ieee300", or a locally-generated dataset name.
    split       : None (all) | "train" | "val" | "test"  (chronological 60/20/20).
    families    : optional subset, e.g. ["Ao","ramp","LRA"] or [1,5,6].
    include_gaps: keep physics non-convergence NA rows (default False).
    heldout     : unseen-attack protocol — exclude As/Ar from train/val (Boyaci et al. 2022).
    format      : "torch" (dict batches) | "pyg" (torch_geometric Data).
    release     : dataset VERSION. None -> newest published release (default, always-current for the group);
                  an explicit tag e.g. "v0.2.0" -> that exact version, for reproducible experiments.
    """
    # Two-step resolution: resolve() maps (name, release) to a download spec (which GitHub
    # release + which asset), then ensure_local() downloads+caches it if needed and hands
    # back a concrete on-disk .h5 path. Local generated datasets short-circuit to their file.
    path = ensure_local(resolve(name, release=release))
    # Wrap that shard in an FdiaGraph. All the row-selection knobs (split/families/gaps/heldout)
    # and the export format are applied lazily by the Dataset, not here — this is a thin factory.
    return FdiaGraph(path, split=split, families=families, include_gaps=include_gaps, heldout=heldout, format=format)


def generate(system, name, **knobs):
    """Generate a custom dataset with research knobs and register it as `name` (loadable via load(name)).

    Knobs (all optional): per_family, families, attack_intensity, ramp_rate, ramp_len, n_benign,
    redundancy, split, seed, out. See fdia_graph.generate module for the full documented signature.
    Requires the generation extra: pip install 'fdia-graph[generate]'.
    """
    # Lazy import: the generator pulls in heavy deps (pandapower, the _core FdiaGenerator).
    # Importing at module top would force every `import fdia_graph` user to have the
    # optional [generate] extra installed just to call load(); we defer the cost to call time.
    from .generate import generate as _generate
    # Delegate to the real implementation. `system` names the grid (ieee14/118/300),
    # `name` is what the result registers as (so load(name) can retrieve it), and **knobs
    # forwards the research parameters untouched so this wrapper never goes stale as knobs change.
    return _generate(system, name=name, **knobs)
