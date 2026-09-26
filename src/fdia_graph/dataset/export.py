"""A whole split at once: `summary`, and the arrays/tensors/frames of every kept record."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import pandas as pd
    import torch


import h5py
import numpy as np

from .. import schema
from ..models.choices import (  # noqa: F401  re-exported beside the code that reads them
    Format,
)
from ..models.config import ExportRequest
from ..models.data import ArraysBundle, Summary
from ..models.validation import expect
from .base import (
    _BENIGN_LAYERS,
    _CLEAN_LAYERS,
    _PREV_FIELDS,
    _UNIT_KIND,
    FAMILIES,
    DatasetBase,
    _torch,
)

_INT_KEYS = frozenset(
    {"family", "stealthy", "seq_id", "timestep", "prev_timestep", "edge_index"}
)  # int64 tensors


class ExportMixin(DatasetBase):
    def summary(self) -> Summary:
        # Cheap overview: read only the family column for this view's rows, tally per family.
        with h5py.File(self.path, "r") as f:
            fam = f[schema.FAMILY][:][self.idx]  # family codes for just the kept rows
        return Summary(
            system=self.system,
            N=self.N,
            E=self.E,
            n=len(self),
            families={FAMILIES[k]: int((fam == k).sum()) for k in FAMILIES if (fam == k).any()},
        )

    # ------------------------------------------------------------------ #
    #  The whole-split export: unlike __getitem__/.loader() (stream one record), it pulls the ENTIRE
    #  filtered split into memory. Every format shares `_arrays` as the single HDF5 read.
    # ------------------------------------------------------------------ #
    def _checked_fields(self, fields: Optional[Sequence[str]]) -> list[str]:
        """The requested fields, or every field the file carries; an unknown name is an error."""
        known = self._default_fields()
        if not fields:
            return known
        # on request only, a timeline's previous frame; its swing only when the file carries swing
        prev = [k for k in _PREV_FIELDS if self.is_timeline and (k != "prev_swing" or self.has_swing)]
        offered = known + prev
        unknown = [k for k in fields if k not in offered]
        expect(not (unknown), f"unknown field(s) {unknown}; this shard carries {known}")
        return list(fields)

    def _default_fields(self) -> list[str]:
        """Every per-record array the file carries, in the order `export` returns them."""
        return (
            ["node_x", "node_m", "edge_x", "edge_m", "y"]
            + (["temporal_delta"] if self.has_temporal else [])
            + (["swing"] if self.has_swing else [])
            + (["clean"] if self._clean_np is not None else [])
            + (["edge_clean"] if self._eclean_np is not None else [])
            + (["edge_clean_full"] if self.has_clean_full else [])
            + (list(_BENIGN_LAYERS) if self.has_benign else [])
            + ["family", "stealthy", "seq_id", "timestep"]
        )

    def _clean_layers(self, want: Sequence[str], ts: np.ndarray) -> dict[str, np.ndarray]:
        """The requested clean layers gathered per record through the pool timestep."""
        out: dict[str, np.ndarray] = {}
        if "clean" in want and self._clean_np is not None:
            out["clean"] = self._clean_np[ts]
        if "edge_clean" in want and self._eclean_np is not None:
            out["edge_clean"] = self._eclean_np[ts]
        if "edge_clean_full" in want:
            ecf = self._clean_flows_full()
            if ecf is not None:
                out["edge_clean_full"] = ecf[ts]
        return out

    def export(
        self,
        fields: Optional[Sequence[str]] = None,
        format: str = "numpy",
        device: Optional[Union[str, torch.device]] = None,
        flatten_features: bool = True,
    ) -> Union[ArraysBundle, pd.DataFrame]:
        """The whole selected split in one HDF5 read, as `format` asks: "numpy" (the default) and
        "torch" return an `ArraysBundle` of arrays or tensors (floats float32, ids int64, tensors on
        `device` when given), "tf" the same as TensorFlow tensors, "pandas" one row per record
        (`flatten_features` spreads the per-bus and per-branch readings into columns).

        Keys: node_x [n,N,4], node_m, edge_x [n,E,2], edge_m, y [n,N], family/stealthy/seq_id/
        timestep [n], plus the static graph edge_index [2,E] and edge_reactance [E], always included.
        On a timeline, `fields` may also ask for prev_node_x [n,N,4], prev_edge_x [n,E,2],
        prev_timestep [n] and prev_swing [n,N,2] (when the file carries swing): the readings of the
        frame emitted just before each record (file row - 1, whatever split or family it belongs to);
        they are never part of the default set.
        `fields` limits the per-record arrays read; a pandas frame carries every field and refuses
        `fields`, so a typo cannot pass unnoticed.
        """
        format = ExportRequest(format, None if fields is None else tuple(fields)).format
        if format == "pandas":
            return self._as_pandas(self._arrays(None), flatten_features)
        arrays = self._arrays(fields)
        if format == "torch":
            return self._as_torch(arrays, device)
        if format == "tf":
            return self._as_tf(arrays)
        return arrays

    def _arrays(self, fields: Optional[Sequence[str]]) -> ArraysBundle:
        """The numpy export every format is built from: one bulk read, in the view's order and
        units, the caller's field order kept."""
        want = self._checked_fields(fields)
        # Static graph arrays always included (tiny, and needed to interpret edges).
        out = {"edge_index": self.edge_index_np, "edge_reactance": self.edge_reactance_np}
        out.update(self._gather(want))
        if self._perm is not None:  # order="random": the gathered rows in the view's permutation
            out.update({k: out[k][self._perm] for k in want})
        return ArraysBundle.ordered(self._in_units(out))  # keeps the caller's field order

    def _gather(self, want: Sequence[str]) -> dict[str, np.ndarray]:
        """The requested per-record arrays for the kept rows, in file order: one bulk gather per
        field from data/, the benign layers from benign/, the clean layers through the timestep."""
        derived = set(_CLEAN_LAYERS) | set(_PREV_FIELDS)
        # self.idx is sorted-unique by construction, as h5py fancy-indexing requires
        with h5py.File(self.path, "r") as f:
            out = {k: f[schema.FIELD_PATH[k]][self.idx] for k in want if k not in derived}  # [n, ...] arrays
            out.update(self._derived_layers(f, want))
        return out

    def _derived_layers(self, f: h5py.File, want: Sequence[str]) -> dict[str, np.ndarray]:
        """The layers not stored per kept row: the clean layers through the pool timestep, the
        previous frame's readings through file row - 1."""
        out: dict[str, np.ndarray] = {}
        clean_want = [k for k in want if k in _CLEAN_LAYERS]
        if clean_want:
            out.update(self._clean_layers(clean_want, f[schema.TIMESTEP][self.idx]))
        prev_want = [k for k in want if k in _PREV_FIELDS]
        if prev_want:
            out.update(self._previous_frame(f, prev_want))
        return out

    def _previous_frame(self, f: h5py.File, want: Sequence[str]) -> dict[str, np.ndarray]:
        """The readings of the frame emitted just before each kept record (file row - 1, read
        whatever split or family that row belongs to; the first frame stands in for its own)."""
        rows, back = np.unique(np.maximum(self.idx - 1, 0), return_inverse=True)  # h5py needs sorted-unique
        return {k: f[schema.FIELD_PATH[_PREV_FIELDS[k]]][rows][back] for k in want}

    def _in_units(self, out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The returned arrays in self.units: power and angle arrays converted to per unit when asked,
        masks, labels and the swing z-score untouched."""
        if self.units == "pu":
            for k, kind in _UNIT_KIND.items():
                if k in out:
                    out[k] = self._to_units(out[k], kind)
        return out

    def _as_torch(self, arrays: ArraysBundle, device: Optional[Union[str, torch.device]]) -> ArraysBundle:
        """The arrays as torch tensors: ids int64, measurements float32, moved to `device` if given."""
        torch = _torch()
        out = {}
        for k, v in arrays.items():
            t = torch.as_tensor(v)
            t = t.long() if k in _INT_KEYS else t.float()
            out[k] = t.to(device) if device else t
        return ArraysBundle.ordered(out)

    def _as_tf(self, arrays: ArraysBundle) -> ArraysBundle:
        """The arrays as TensorFlow tensors (tensorflow installed); wrap in
        tf.data.Dataset.from_tensor_slices(...) for a pipeline."""
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError("TensorFlow is required for format='tf': pip install tensorflow") from e
        return ArraysBundle.ordered({k: tf.convert_to_tensor(v) for k, v in arrays.items()})

    def _as_pandas(self, a: ArraysBundle, flatten_features: bool) -> pd.DataFrame:
        """One row per record: the metadata columns (family name, stealthy flag, seq_id, timestep,
        n_attacked_buses) and, with `flatten_features`, the per-bus and per-branch readings as
        columns (V_b{n}, Pinj_b{n}, Pflow_e{n}, ...) and the per-bus labels attacked_b{n}."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for format='pandas': pip install pandas") from e
        df = pd.DataFrame(
            {
                "family": [FAMILIES[int(k)] for k in a["family"]],  # readable family name
                "family_id": a["family"],
                "stealthy": a["stealthy"].astype(bool),
                "seq_id": a["seq_id"],
                "timestep": a["timestep"],
                "n_attacked_buses": a["y"].sum(axis=1).astype(int),
            }
        )
        if not flatten_features:
            return df
        N, E = self.N, self.E
        for ci, nm in enumerate(["V", "Pinj", "Qinj", "theta"]):  # node_x[:, :, ci] is [n, N]
            df = pd.concat(
                [df, pd.DataFrame(a["node_x"][:, :, ci], columns=[f"{nm}_b{b}" for b in range(N)])], axis=1
            )
        for ci, nm in enumerate(["Pflow", "Qflow"]):  # edge_x[:, :, ci] is [n, E]
            df = pd.concat(
                [df, pd.DataFrame(a["edge_x"][:, :, ci], columns=[f"{nm}_e{e}" for e in range(E)])], axis=1
            )
        labels = pd.DataFrame(a["y"].astype(int), columns=[f"attacked_b{b}" for b in range(N)])
        return pd.concat([df, labels], axis=1)

    # The four exporters of 0.17 and earlier, one `export(format=...)` since 0.18; retire in 0.19.
    def to_numpy(self, fields: Optional[Sequence[str]] = None) -> ArraysBundle:
        _retiring("to_numpy", "export(fields)")
        return self._arrays(fields)

    def to_torch(
        self, fields: Optional[Sequence[str]] = None, device: Optional[Union[str, torch.device]] = None
    ) -> ArraysBundle:
        _retiring("to_torch", "export(fields, format='torch', device=device)")
        return self._as_torch(self._arrays(fields), device)

    def to_tf(self, fields: Optional[Sequence[str]] = None) -> ArraysBundle:
        _retiring("to_tf", "export(fields, format='tf')")
        return self._as_tf(self._arrays(fields))

    def to_pandas(self, flatten_features: bool = True) -> pd.DataFrame:
        _retiring("to_pandas", "export(format='pandas', flatten_features=...)")
        return self._as_pandas(self._arrays(None), flatten_features)


def _retiring(name: str, replacement: str) -> None:
    warnings.warn(
        f"{name} is deprecated and retires in 0.19: use ds.{replacement}, the one export of a split",
        DeprecationWarning,
        stacklevel=3,
    )
