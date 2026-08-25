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
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    import numpy as np

# Re-exports so users write `fg.FdiaGraph` / `fg.load(...)` instead of reaching into submodules.
# FdiaGraph: torch Dataset over one .h5 shard; FAMILIES: attack-family names/ids; STEALTHY_FAMILIES: the
# BDD-evading subset (hard cases, e.g. Aq).
from .dataset import FdiaGraph, FAMILIES, STEALTHY_FAMILIES
# registry = dataset "version control": list_datasets (known built-in + local), register_local (name a local
# dataset), resolve ((name, release) -> download spec).
from .registry import list_datasets, register_local, resolve
# download: ensure_local (resolved spec -> local .h5 path, fetching+caching if absent).
from .download import ensure_local

__version__ = "0.6.0"
# Public API for `from fdia_graph import *`; register_local/resolve/ensure_local stay out (internal plumbing).
__all__ = ["load", "generate", "load_profile", "fetch_profile", "generate_states",
           "line_outage_candidates", "list_datasets", "FdiaGraph", "FAMILIES", "STEALTHY_FAMILIES"]


def line_outage_candidates(system: Union[str, int], top_n: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """Rank single-line N-1 contingencies by base-case flow, screening out any that island the grid.

    Returns (accepted, rejected) lists of dicts. Use the accepted line indices as generate(..., outage=idx)
    to build one shard per post-contingency topology.
    """
    # Lazy: pulls in pandapower, which most SDK (loader) users don't have installed.
    from ._core import line_outage_candidates as _cands
    return _cands(system, top_n=top_n)


def fetch_profile(iso: str, start: str, end: str, out: Optional[str] = None,
                  resample_min: Optional[int] = None) -> np.ndarray:
    """Auto-download an ISO system-load series and return a normalized scaling vector S [T].

    iso is "caiso"/"nyiso"/"ercot"; start/end are 'YYYY-MM-DD'. NYISO needs no account or extra deps;
    CAISO/ERCOT use the gridstatus package (pip install 'fdia-graph[iso]'). resample_min (e.g. 1)
    time-interpolates the load to that minute cadence (upsample the 5-min feed to 1-min). See
    fdia_graph.profiles. Feed the result to generate_states(). Requires the generation extra.
    """
    from .profiles import fetch_profile as _fetch_profile
    return _fetch_profile(iso, start, end, out=out, resample_min=resample_min)


def load_profile(source: Union[str, Sequence[float], np.ndarray], path: Optional[str] = None,
                 column: Optional[str] = None) -> np.ndarray:
    """Ingest a load time series into a normalized scaling vector S [T] (see fdia_graph.profiles).

    Pluggable front of the pipeline: `source` is "caiso"/"nyiso" (+ a `path` to the ISO CSV directory),
    a generic CSV path (+ `column`), or an array of raw load values. Swap sources/time periods freely.
    Requires the generation extra: pip install 'fdia-graph[generate]'.
    """
    from .profiles import load_profile as _load_profile
    return _load_profile(source, path=path, column=column)


def generate_states(system: Union[str, int], profile: Union[np.ndarray, Sequence[float]], **knobs: Any) -> np.ndarray:
    """Turn a load profile into a pool of AC operating states [T,N,4] to inject attacks onto.

    Pass the result straight to generate(system, name, states=...). Knobs: k, sigma, clip, n, seed
    (see fdia_graph.profiles.generate_states). Requires the generation extra.
    """
    from .profiles import generate_states as _generate_states
    return _generate_states(system, profile, **knobs)


def load(name: str, split: Optional[str] = None, families: Optional[Sequence[Union[str, int]]] = None,
         include_gaps: bool = False, heldout: bool = False, format: str = "torch", release: Optional[str] = None,
         units: str = "physical") -> FdiaGraph:
    """Load a dataset by name (built-in shard auto-downloads; local generated ones load from disk).

    name        : "ieee14"|"ieee118"|"ieee300", or a locally-generated dataset name.
    split       : None (all) | "train" | "val" | "test"  (chronological 60/20/20).
    families    : optional subset, e.g. ["Aq","At","Al"] or [1,5,6].
    include_gaps: keep physics non-convergence NA rows (default False).
    heldout     : unseen-attack protocol — exclude As/Ar from train/val (Boyaci et al. 2022).
    format      : "torch" (dict batches) | "pyg" (torch_geometric Data).
    release     : dataset VERSION. None -> newest published release (default, always-current for the group);
                  an explicit tag e.g. "v0.2.0" -> that exact version, for reproducible experiments.
    units       : "physical" -> [V p.u., P_inj MW, Q_inj MVAr, theta deg] (as stored; human-readable for plots);
                  "pu" -> everything per-unit on baseMVA with theta in radians (ML/physics). Same shard either way.
    """
    # resolve() -> download spec, ensure_local() -> on-disk .h5 path (fetching if needed; local datasets
    # short-circuit to their file).
    path = ensure_local(resolve(name, release=release))
    # Thin factory: the Dataset applies split/families/gaps/heldout and the export format lazily.
    return FdiaGraph(path, split=split, families=families, include_gaps=include_gaps, heldout=heldout,
                     format=format, units=units)


def generate(system: Union[str, int], name: str, **knobs: Any) -> str:
    """Generate a custom dataset with research knobs and register it as `name` (loadable via load(name)).

    Knobs (all optional): per_family, families, attack_intensity, ramp_rate, ramp_len, n_benign,
    redundancy, split, seed, out. See fdia_graph.generation module for the full documented signature.
    Requires the generation extra: pip install 'fdia-graph[generate]'.
    """
    # Lazy: the generator pulls in heavy deps (pandapower, _core), so deferring keeps plain load() users
    # from needing the optional [generate] extra.
    from .generation import generate as _generate
    # **knobs forwarded untouched so this wrapper never goes stale as knobs change.
    return _generate(system, name=name, **knobs)
