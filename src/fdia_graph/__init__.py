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
    import datetime
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

__version__ = "0.7.2"
# Public API for `from fdia_graph import *`; register_local/resolve/ensure_local stay out (internal plumbing).
__all__ = ["load", "generate", "generate_stream", "load_stream", "windows", "pyg_stream", "torch_windows",
           "load_profile", "fetch_profile",
           "generate_states", "line_outage_candidates", "list_datasets", "FdiaGraph", "FAMILIES",
           "STEALTHY_FAMILIES"]


def line_outage_candidates(system: Union[str, int], top_n: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """Rank single-line N-1 contingencies by base-case flow, screening out any that island the grid.

    Returns (accepted, rejected) lists of dicts. Use the accepted line indices as generate(..., outage=idx)
    to build one shard per post-contingency topology.
    """
    # Lazy: pulls in pandapower, which most SDK (loader) users don't have installed.
    from ._core import line_outage_candidates as _cands
    return _cands(system, top_n=top_n)


def fetch_profile(iso: str, start: Union[str, "datetime.date", "datetime.datetime"],
                  end: Union[str, "datetime.date", "datetime.datetime"], out: Optional[str] = None,
                  resample_min: Optional[int] = None) -> np.ndarray:
    """Auto-download an ISO system-load series and return a normalized scaling vector S [T].

    iso is "caiso"/"nyiso"/"ercot"; start/end are 'YYYY-MM-DD' (or date/datetime). NYISO needs no account
    or extra deps; CAISO/ERCOT use the gridstatus package (pip install 'fdia-graph[iso]'). resample_min
    (e.g. 1) time-interpolates the load to that minute cadence (upsample the 5-min feed to 1-min). See
    fdia_graph.profiles. Feed the result to generate_states().
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

    name        : "ieee14/30/57/89/118/145/200/300" (transmission ladder), or a locally-generated name.
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

    Knobs (all optional): per_family, families, attack_intensity, ramp_rate, ramp_len, replay_tau, n_benign,
    redundancy, split, seed, out. See fdia_graph.generation module for the full documented signature.
    Requires the generation extra: pip install 'fdia-graph[generate]'.
    """
    # Lazy: the generator pulls in heavy deps (pandapower, _core), so deferring keeps plain load() users
    # from needing the optional [generate] extra.
    from .generation import generate as _generate
    # **knobs forwarded untouched so this wrapper never goes stale as knobs change.
    return _generate(system, name=name, **knobs)


def generate_stream(system: Union[str, int], **knobs: Any) -> Dict[str, Any]:
    """Build ONE continuous attacked time series for temporal models (LSTM/TGN), not a shuffled table.

    Returns a dict with node_x [T,N,4], clean [T,N,4] (noiseless attack-free SE target), y [T,N], family [T],
    temporal_delta/swing [T,N,2], and an episode list; saved to `out` (npz) if given. Knobs: states,
    attacked_frac, families, attack_intensity, ramp_rate, ramp_len, replay_tau, seed, out. Needs the generation extra.
    """
    from .streams import generate_stream as _gs
    return _gs(system, **knobs)


def load_stream(system: Union[str, int], release: Optional[str] = None) -> Dict[str, Any]:
    """Download the published continuous attacked stream for a system (dict: node_x, y, family, ...).

    Built-in systems 14/30/57/89/118/145/200/300. Feed to windows() for LSTM/TGN training. release pins a
    version. See fdia_graph.streams.load_stream.
    """
    from .streams import load_stream as _ls
    return _ls(system, release=release)


def windows(stream: Dict[str, Any], W: int, stride: int = 1, label: str = "any") -> Tuple[np.ndarray, np.ndarray]:
    """Slide a length-W window over a stream (from generate_stream/load_stream) into (Xw [n,W,N,4], yw) for an LSTM.

    label: "frame" -> yw [n,W,N] per-frame; "any" -> yw [n,N] attacked-anywhere-in-window; "last" -> yw [n,N]
    at the final frame. See fdia_graph.streams.windows.
    """
    from .streams import windows as _windows
    return _windows(stream, W, stride=stride, label=label)


def pyg_stream(system: Optional[Union[str, int]] = None, train_frac: float = 0.8, layer: str = "node_x",
               max_test: Optional[int] = None, release: Optional[str] = None,
               stream: Optional[Dict[str, Any]] = None) -> Tuple[List[Any], List[Any]]:
    """Continuous stream as ready PyTorch-Geometric graphs: (train, test) lists of Data objects.

    One Data(x=[N,4], edge_index, edge_attr, y=[N]) per scan, chronological train/test split — no
    conversion glue needed. See fdia_graph.torch_data.pyg_stream. Needs pip install "fdia-graph[pyg]".
    """
    from .torch_data import pyg_stream as _pyg
    return _pyg(system, train_frac=train_frac, layer=layer, max_test=max_test, release=release, stream=stream)


def torch_windows(system: Optional[Union[str, int]] = None, W: int = 16, stride: int = 8, label: str = "last",
                  per_bus: bool = True, train_frac: float = 0.8, layer: str = "node_x",
                  release: Optional[str] = None,
                  stream: Optional[Dict[str, Any]] = None) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
    """Continuous stream as LSTM-ready per-bus sequence tensors: ((Xtr, ytr), (Xte, yte)).

    Windows the stream, reshapes to per-bus sequences [n*N, W, 4], splits chronologically (boundary
    straddlers dropped). See fdia_graph.torch_data.torch_windows. Needs pip install "fdia-graph[torch]".
    """
    from .torch_data import torch_windows as _tw
    return _tw(system, W=W, stride=stride, label=label, per_bus=per_bus, train_frac=train_frac,
               layer=layer, release=release, stream=stream)
