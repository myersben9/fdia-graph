"""PyTorch-ready views of the continuous streams — no conversion glue in user code.

These wrap ``load_stream()`` so a model script starts at the tensors: ``pyg_stream()`` hands back
``torch_geometric.data.Data`` objects (one graph per scan, connectivity and branch physics attached),
``torch_windows()`` hands back per-bus sequence tensors for an LSTM/GRU. Both split chronologically
(train first, test last), matching how the stream would be consumed live. ``torch`` /
``torch_geometric`` are imported lazily so the base install stays torch-free.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the runtime torch-free
    import torch
    from torch_geometric.data import Data

__all__ = ["pyg_stream", "torch_windows"]


def _f32(a: Any) -> "torch.Tensor":
    """numpy -> float32 torch tensor, zero-copy when the array is already contiguous float32."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def _check_frac(train_frac: float) -> None:
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")


def _resolve_stream(system: Optional[Union[str, int]], release: Optional[str],
                    stream: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Accept either a system name (loaded here) or an already-loaded stream dict."""
    if stream is not None:
        return stream
    if system is None:
        raise ValueError("pass a system name (e.g. 'ieee118') or stream=<loaded stream dict>")
    from .streams import load_stream
    return load_stream(system, release=release)


def pyg_stream(
    system: Optional[Union[str, int]] = None,
    train_frac: float = 0.8,
    val_frac: float = 0.0,
    layer: str = "node_x",
    max_test: Optional[int] = None,
    release: Optional[str] = None,
    stream: Optional[Dict[str, Any]] = None,
) -> Tuple[List["Data"], ...]:
    """Continuous stream as ready PyTorch-Geometric graphs, split chronologically.

    Each scan becomes one ``Data(x=[N,4], edge_index=[2,E], edge_attr=[E,8], y=[N])``. The graph is
    static, so ``edge_index``/``edge_attr`` are shared tensors (one copy for the whole stream).
    Needs ``pip install "fdia-graph[pyg]"``.

    Args:
        system: system name (e.g. "ieee118") to load, or None if passing ``stream``.
        train_frac: fraction of scans (from the start) used for training.
        val_frac: if > 0, carve a validation span between train and test (returns 3 lists, not 2).
        layer: measurement layer to use as ``x`` -- "node_x" (attacked, default), "benign", or "clean".
        max_test: cap on the test list length (e.g. 1000 for a quick eval); None = all remaining.
        release: dataset release tag to pin; None = the installed default.
        stream: an already-loaded stream dict, to reuse instead of loading by ``system``.

    Returns:
        ``(train, test)`` lists of ``Data``, or ``(train, val, test)`` when ``val_frac`` > 0.
    """
    import torch
    from torch_geometric.data import Data
    _check_frac(train_frac)
    if max_test is not None and max_test < 0:
        raise ValueError(f"max_test must be >= 0, got {max_test}")
    s = _resolve_stream(system, release, stream)
    X = _f32(s[layer])                                                              # [T, N, 4]
    Y = _f32(s["y"])                                                                # [T, N]
    ei = torch.from_numpy(np.ascontiguousarray(s["edge_index"], dtype=np.int64))    # [2, E], shared
    ea = _f32(s["edge_attr"])                                                       # [E, 8], shared
    if not 0.0 <= val_frac < 1.0 or train_frac + val_frac >= 1.0:
        raise ValueError(f"need train_frac + val_frac < 1, got {train_frac} + {val_frac}")
    T = int(X.shape[0])
    ntr = int(train_frac * T)
    nva = int(val_frac * T)

    def graphs(a: int, b: int) -> List["Data"]:
        return [Data(x=X[t], edge_index=ei, edge_attr=ea, y=Y[t]) for t in range(a, b)]
    train = graphs(0, ntr)
    val = graphs(ntr, ntr + nva)
    te0 = ntr + nva
    nte = T - te0 if max_test is None else min(max_test, T - te0)
    test = graphs(te0, te0 + nte)
    return (train, val, test) if val_frac > 0 else (train, test)


def torch_windows(
    system: Optional[Union[str, int]] = None,
    W: int = 16,
    stride: int = 8,
    label: str = "last",
    per_bus: bool = True,
    train_frac: float = 0.8,
    val_frac: float = 0.0,
    layer: str = "node_x",
    release: Optional[str] = None,
    stream: Optional[Dict[str, Any]] = None,
) -> Tuple[Tuple["torch.Tensor", "torch.Tensor"], ...]:
    """Continuous stream as LSTM-ready sequence tensors, split chronologically.

    Slides a length-``W`` window over the stream. Windows straddling a split boundary are dropped, so no
    eval frame is ever seen in training. Needs ``pip install "fdia-graph[torch]"``.

    Args:
        system: system name (e.g. "ieee118") to load, or None if passing ``stream``.
        W: window length in scans.
        stride: step between window starts.
        label: per ``windows()`` -- "last" (label at the final frame), "any", or "frame" (per-frame labels).
        per_bus: True gives one sequence per bus, ``X [n*N, W, 4]`` (what nn.LSTM consumes); False keeps
            whole-grid windows ``[n, W, N, 4]``.
        train_frac: fraction of frames (from the start) used for training.
        val_frac: if > 0, carve a validation span between train and test (returns 3 tuples, not 2).
        layer: measurement layer to window -- "node_x" (default), "benign", or "clean".
        release: dataset release tag to pin; None = the installed default.
        stream: an already-loaded stream dict, to reuse instead of loading by ``system``.

    Returns:
        ``((Xtr, ytr), (Xte, yte))``, or ``((Xtr, ytr), (Xva, yva), (Xte, yte))`` when ``val_frac`` > 0.
    """
    import torch
    from .streams import windows as _windows
    _check_frac(train_frac)
    if not 0.0 <= val_frac < 1.0 or train_frac + val_frac >= 1.0:
        raise ValueError(f"need train_frac + val_frac < 1, got {train_frac} + {val_frac}")
    s = _resolve_stream(system, release, stream)
    if layer != "node_x":
        s = {**s, "node_x": s[layer]}
    T = int(np.asarray(s["node_x"]).shape[0])
    if not 1 <= W <= T:
        raise ValueError(f"W must be in [1, {T}] (stream length), got {W}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    Xw, yw = _windows(s, W, stride=stride, label=label)
    cut = int(train_frac * T)
    cut2 = cut + int(val_frac * T)
    starts = np.arange(0, T - W + 1, stride)
    tr = starts + W <= cut                            # window fully inside the train span
    va = (starts >= cut) & (starts + W <= cut2)       # window fully inside the val span
    te = starts >= cut2                               # window fully inside the test span (straddlers dropped)

    def _cvt(Xp: np.ndarray, yp: np.ndarray) -> Tuple["torch.Tensor", "torch.Tensor"]:
        if per_bus:
            n, Wn, N, C = Xp.shape
            X = _f32(Xp.transpose(0, 2, 1, 3).reshape(n * N, Wn, C))
            y = (_f32(yp.transpose(0, 2, 1).reshape(n * N, Wn))
                 if label == "frame" else _f32(yp.reshape(n * N)))
        else:
            X, y = _f32(Xp), _f32(yp)
        return X, y

    if val_frac > 0:
        return _cvt(Xw[tr], yw[tr]), _cvt(Xw[va], yw[va]), _cvt(Xw[te], yw[te])
    return _cvt(Xw[tr], yw[tr]), _cvt(Xw[te], yw[te])
