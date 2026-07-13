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
from .dataset import FdiaGraph, FAMILIES, STEALTHY_FAMILIES
from .registry import list_datasets, register_local, resolve
from .download import ensure_local

__version__ = "0.1.0"
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
    path = ensure_local(resolve(name, release=release))
    return FdiaGraph(path, split=split, families=families, include_gaps=include_gaps, heldout=heldout, format=format)


def generate(system, name, **knobs):
    """Generate a custom dataset with research knobs and register it as `name` (loadable via load(name)).

    Knobs (all optional): per_family, families, attack_intensity, ramp_rate, ramp_len, n_benign,
    redundancy, split, seed, out. See fdia_graph.generate module for the full documented signature.
    Requires the generation extra: pip install 'fdia-graph[generate]'.
    """
    from .generate import generate as _generate
    return _generate(system, name=name, **knobs)
