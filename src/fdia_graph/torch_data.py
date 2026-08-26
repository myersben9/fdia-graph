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


def _resolve_stream(system: Optional[Union[str, int]], release: Optional[str],
                    stream: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Accept either a system name (loaded here) or an already-loaded stream dict."""
    if stream is not None:
        return stream
    if system is None:
        raise ValueError("pass a system name (e.g. 'ieee118') or stream=<loaded stream dict>")
    from .streams import load_stream
    return load_stream(system, release=release)


def pyg_stream(system: Optional[Union[str, int]] = None, train_frac: float = 0.8,
               layer: str = "node_x", max_test: Optional[int] = None,
               release: Optional[str] = None,
               stream: Optional[Dict[str, Any]] = None) -> Tuple[List["Data"], List["Data"]]:
    """Continuous stream as ready PyTorch-Geometric graphs: ``(train, test)`` lists of ``Data``.

    Each scan is one ``Data(x=[N,4], edge_index=[2,E], edge_attr=[E,8], y=[N])`` — node measurements,
    connectivity, per-unit branch physics, and the per-bus attack label. ``edge_index``/``edge_attr``
    are shared tensors (the graph is static), so memory stays one stream copy. The split is
    chronological: the first ``train_frac`` of scans train, the rest test. ``max_test`` bounds the
    test list (e.g. 1000 for a quick eval). ``layer`` picks the measurement layer: ``"node_x"``
    (attacked, default), ``"benign"``, or ``"clean"``. Needs ``pip install "fdia-graph[pyg]"``.
    """
    import torch
    from torch_geometric.data import Data
    s = _resolve_stream(system, release, stream)
    X = torch.tensor(np.asarray(s[layer]), dtype=torch.float32)                     # [T, N, 4]
    Y = torch.tensor(np.asarray(s["y"]), dtype=torch.float32)                       # [T, N]
    ei = torch.from_numpy(np.ascontiguousarray(s["edge_index"], dtype=np.int64))    # [2, E], shared
    ea = torch.tensor(np.asarray(s["edge_attr"]), dtype=torch.float32)              # [E, 8], shared
    T = int(X.shape[0])
    ntr = int(train_frac * T)
    train = [Data(x=X[t], edge_index=ei, edge_attr=ea, y=Y[t]) for t in range(ntr)]
    nte = T - ntr if max_test is None else min(max_test, T - ntr)
    test = [Data(x=X[t], edge_index=ei, edge_attr=ea, y=Y[t]) for t in range(ntr, ntr + nte)]
    return train, test


def torch_windows(system: Optional[Union[str, int]] = None, W: int = 16, stride: int = 8,
                  label: str = "last", per_bus: bool = True, train_frac: float = 0.8,
                  layer: str = "node_x", release: Optional[str] = None,
                  stream: Optional[Dict[str, Any]] = None,
                  ) -> Tuple[Tuple["torch.Tensor", "torch.Tensor"], Tuple["torch.Tensor", "torch.Tensor"]]:
    """Continuous stream as LSTM-ready sequence tensors: ``((Xtr, ytr), (Xte, yte))``.

    Slides a length-``W`` window over the stream and returns float32 tensors. With ``per_bus=True``
    (default) each bus contributes its own sequence — ``X [n_windows*N, W, 4]``, ``y [n_windows*N]``
    — which is the shape a plain ``nn.LSTM`` consumes directly. ``per_bus=False`` keeps whole-grid
    windows ``[n, W, N, 4]``. ``label`` follows ``windows()``: ``"last"`` (label at the final frame),
    ``"any"``, or ``"frame"`` (per-frame labels, ``y [.., W]``). The split is chronological on the
    underlying frames; windows straddling the boundary are dropped, so no test frame is ever seen in
    training. ``layer`` picks ``"node_x"`` / ``"benign"`` / ``"clean"``. Needs
    ``pip install "fdia-graph[torch]"``.
    """
    import torch
    from .streams import windows as _windows
    s = _resolve_stream(system, release, stream)
    if layer != "node_x":
        s = {**s, "node_x": s[layer]}
    Xw, yw = _windows(s, W, stride=stride, label=label)
    T = int(np.asarray(s["node_x"]).shape[0])
    cut = int(train_frac * T)
    starts = np.arange(0, T - W + 1, stride)
    tr = starts + W <= cut          # window fully inside the train span
    te = starts >= cut              # window fully inside the test span (straddlers dropped)

    def _cvt(Xp: np.ndarray, yp: np.ndarray) -> Tuple["torch.Tensor", "torch.Tensor"]:
        if per_bus:
            n, Wn, N, C = Xp.shape
            X = torch.tensor(Xp.transpose(0, 2, 1, 3).reshape(n * N, Wn, C), dtype=torch.float32)
            y = (torch.tensor(yp.transpose(0, 2, 1).reshape(n * N, Wn), dtype=torch.float32)
                 if label == "frame" else torch.tensor(yp.reshape(n * N), dtype=torch.float32))
        else:
            X = torch.tensor(Xp, dtype=torch.float32)
            y = torch.tensor(yp, dtype=torch.float32)
        return X, y

    return _cvt(Xw[tr], yw[tr]), _cvt(Xw[te], yw[te])
