"""The deprecated stream entry points, to be removed in a future release: `generate_stream` writes a timeline file
through `fdia_graph.timeline` and returns it as the stream dict, `load_stream` reads the v0.7.2
stream files, and `windows` slides over a stream dict. The timeline file is the dataset now:
`fg.generate` writes it and `fg.load(name, order="time")` reads it, with `ds.windows`.

A stream dict has, per frame, three aligned measurement layers ([|V|, Pinj, Qinj, angle] columns):
node_x (observed), benign (attack removed, noise kept), clean (noiseless truth), the same three for
branch flows, the static graph and meter masks, the labels, the two temporal features, and the
episode list (see the timeline module for what each layer means).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np

from .dataset.sequence import check_window_args, window_labels
from .models.data import Stream  # noqa: F401  re-exported: defined here before the models package
from .registry import AssetSpec
from .timeline import DEFAULT_FAMILIES


def stream_summary(s: dict[str, Any]) -> dict[str, Any]:
    """The two summary fields of a stream, derived from its arrays: `system` is the bus count and
    `attacked_frac` the fraction of frames with at least one attacked bus. `load_stream` uses this
    because the stream files carry the arrays only."""
    y = np.asarray(s["y"])
    return {
        "system": int(np.asarray(s["node_x"]).shape[1]),
        "attacked_frac": float((y.sum(axis=1) > 0).mean()),
    }


def generate_stream(
    system: Union[int, str],
    states: Optional[Union[str, np.ndarray]] = None,
    attacked_frac: float = 0.5,
    families: Sequence[str] = DEFAULT_FAMILIES,
    attack_intensity: float = 0.20,
    ramp_rate: float = 0.002,
    ramp_len: int = 60,
    replay_tau: Optional[int] = None,
    redundancy: Optional[dict] = None,
    seed: int = 123,
    out: Optional[str] = None,
    **knobs: Any,
) -> Stream:
    """Deprecated: `fg.generate` writes the timeline and `fg.load(name, order="time")` reads it.
    Builds one timeline file for `system` (`out`, default `stream_ieee{N}.h5` under the cache
    directory) and returns it as the stream dict. The parameters keep their pre-0.18 positions;
    the Am knobs and `corrupt_len` of `timeline.generate_timeline` pass through `knobs`."""
    from .dataset import FdiaGraph
    from .registry import CACHE_DIR, system_id
    from .timeline import generate_timeline

    warnings.warn(
        "generate_stream is deprecated and retires in 0.19: fg.generate writes the same timeline as "
        "one HDF5 file and fg.load(name, order='time') reads it",
        DeprecationWarning,
        stacklevel=2,
    )
    out = out or os.path.join(CACHE_DIR, f"stream_ieee{system_id(system)}.h5")
    path = generate_timeline(
        system,
        states=states,
        attacked_frac=attacked_frac,
        families=families,
        attack_intensity=attack_intensity,
        ramp_rate=ramp_rate,
        ramp_len=ramp_len,
        replay_tau=replay_tau,
        redundancy=redundancy,
        seed=seed,
        out=out,
        **knobs,
    )
    return stream_of(FdiaGraph(path))


_STREAM_FIELDS = (
    "node_x",
    "node_m",
    "edge_x",
    "edge_m",
    "y",
    "family",
    "timestep",
    "temporal_delta",
    "swing",
    "clean",
    "edge_clean",
    "benign",
    "edge_benign",
)


def stream_of(ds: Any) -> Stream:
    """A time-ordered, contiguous timeline view as the stream dict: the per-frame layers, the
    static graph and masks, and the episode list. A random order or a family subset is refused,
    since the frames of a stream are consecutive."""
    ds._check_timeline("stream_of")
    a = ds.export(_STREAM_FIELDS)  # only what the dict carries; edge_clean_full would cost a Yf pass
    ep = ds.episodes
    return Stream(
        node_x=a.node_x,
        benign=a.benign,
        clean=a.clean,
        edge_x=a.edge_x,
        edge_benign=a.edge_benign,
        edge_clean=a.edge_clean,
        edge_index=a.edge_index,
        edge_attr=ds.edge_attr_np,
        node_m=a.node_m[0],
        edge_m=a.edge_m[0],
        y=a.y,
        family=a.family,
        temporal_delta=a.temporal_delta,
        swing=a.swing,
        timestep=a.timestep,
        episodes=[
            dict(onset=int(o), length=int(n), family=int(f), buses=b.tolist())
            for o, n, f, b in zip(ep.onset, ep.length, ep.family, ep.buses)
        ],
        **stream_summary(a),
    )


def _asset_spec(name: str, file: str, release: str) -> AssetSpec:
    from .registry import _REPO

    return AssetSpec("builtin", name, file=file, release=release, repo=_REPO)


def load_stream(system: Union[int, str], release: Optional[str] = None) -> Stream:
    """Deprecated: `fg.load(system, order="time")` is the timeline. Returns the published continuous
    data of a built-in system as the stream dict: at a timeline release (v0.8.0 and later, the
    default) the timeline file read through `stream_of`; at an earlier release the stream file of
    that release. `release`: None -> the pinned data release; a tag pins a version."""
    from .download import ensure_local
    from .registry import STREAM_RELEASE, is_timeline_release, release_name, release_tag, system_id

    warnings.warn(
        "load_stream is deprecated and retires in 0.19: use fg.load(system, order='time'), the same "
        "frames with ds.windows and ds.episodes, without holding the whole file in memory",
        DeprecationWarning,
        stacklevel=2,
    )
    rel = release_name(release or STREAM_RELEASE)  # "0.7.2", "v0.7.2" and "data-v0.7.2" all spell one release
    if is_timeline_release(rel):
        from . import load

        return stream_of(load(f"ieee{system_id(system)}", order="time", release=rel))
    C = system_id(system)
    tag = release_tag(rel)  # the GitHub tag that carries the stream file
    z = np.load(ensure_local(_asset_spec(f"stream{C}", f"stream_ieee{C}.npz", tag)), allow_pickle=True)
    out = {k: z[k] for k in z.files}
    out["edge_index"] = np.asarray(out["edge_index"], dtype=np.int64)  # torch.long
    out["edge_attr"] = np.asarray(out["edge_attr"], dtype=np.float32)
    for m in ("node_m", "edge_m"):
        out[m] = np.asarray(out[m], dtype=np.uint8)
    return Stream(**out, **{k: v for k, v in stream_summary(out).items() if k not in out})


def windows(
    stream: dict[str, Any], W: int, stride: int = 1, label: str = "any"
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a length-W window over a stream. Returns (Xw [n,W,N,4], yw).

    label: "frame" -> per-frame per-bus labels yw [n,W,N]; "any" -> window-level per-bus label yw [n,N]
    (bus attacked at ANY frame in the window); "last" -> label at the final frame yw [n,N].
    """
    nx = stream["node_x"]
    y = stream["y"]
    T = len(nx)
    warnings.warn(
        "windows(stream, ...) is deprecated and retires in 0.19: use ds.windows(W, stride, label) on a "
        "timeline loaded with fg.load(name, order='time')",
        DeprecationWarning,
        stacklevel=2,
    )
    check_window_args(T, W, stride, label)
    starts = range(0, T - W + 1, stride)
    Xw = np.stack([nx[s : s + W] for s in starts])
    return Xw, window_labels(y, starts, W, label)
