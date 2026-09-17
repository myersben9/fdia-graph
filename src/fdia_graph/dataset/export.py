"""A whole split at once: `summary`, and the arrays/tensors/frames of every kept record."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

if TYPE_CHECKING:
    import pandas as pd
    import torch


import numpy as np
import h5py

from ..models.data import ArraysBundle, Summary
from .base import (
    FAMILIES,
    DatasetBase,
    _torch,
    _CLEAN_LAYERS,
    _UNIT_KIND,
)


class ExportMixin(DatasetBase):
    def summary(self) -> Summary:
        # Cheap overview: read only the family column for this view's rows, tally per family.
        with h5py.File(self.path, "r") as f:
            fam = f["data/family"][:][self.idx]  # family codes for just the kept rows
        return Summary(
            system=self.system,
            N=self.N,
            E=self.E,
            n=len(self),
            families={FAMILIES[k]: int((fam == k).sum()) for k in FAMILIES if (fam == k).any()},
        )

    # ------------------------------------------------------------------ #
    #  Whole-split exporters: unlike __getitem__/.loader() (stream one record), these pull the ENTIRE
    #  filtered split into memory. All share to_numpy() as the single HDF5 read, so views are identical.
    # ------------------------------------------------------------------ #
    def _checked_fields(self, fields: Optional[Sequence[str]]) -> List[str]:
        """The requested fields, or every field the file carries; an unknown name is an error."""
        known = self._default_fields()
        if not fields:
            return known
        unknown = [k for k in fields if k not in known]
        if unknown:
            raise ValueError(f"unknown field(s) {unknown}; this shard carries {known}")
        return list(fields)

    def _default_fields(self) -> List[str]:
        """Every per-record array the file carries, in the order to_numpy returns them."""
        return (
            ["node_x", "node_m", "edge_x", "edge_m", "y"]
            + (["temporal_delta"] if self.has_temporal else [])
            + (["swing"] if self.has_swing else [])
            + (["clean"] if self._clean_np is not None else [])
            + (["edge_clean"] if self._eclean_np is not None else [])
            + (["edge_clean_full"] if self.has_clean_full else [])
            + ["family", "stealthy", "seq_id", "timestep"]
        )

    def _clean_layers(self, want: Sequence[str], ts: np.ndarray) -> Dict[str, np.ndarray]:
        """The requested clean layers gathered per record through the pool timestep."""
        out: Dict[str, np.ndarray] = {}
        if "clean" in want and self._clean_np is not None:
            out["clean"] = self._clean_np[ts]
        if "edge_clean" in want and self._eclean_np is not None:
            out["edge_clean"] = self._eclean_np[ts]
        if "edge_clean_full" in want:
            ecf = self._clean_flows_full()
            if ecf is not None:
                out["edge_clean_full"] = ecf[ts]
        return out

    def to_numpy(self, fields: Optional[Sequence[str]] = None) -> ArraysBundle:
        """Return the whole selected split as a dict of numpy arrays (batched over records).

        Keys: node_x [n,N,4], node_m, edge_x [n,E,2], edge_m, y [n,N], family/stealthy/seq_id/timestep [n],
        plus the static graph: edge_index [2,E], edge_reactance [E]. `fields` optionally limits the per-record
        arrays read (the graph arrays are always included since they're tiny and needed to interpret edges).
        """
        want = self._checked_fields(fields)
        per_record = [k for k in want if k not in _CLEAN_LAYERS]
        clean_want = [k for k in want if k in _CLEAN_LAYERS]
        # Static graph arrays always included (tiny, and needed to interpret edges).
        out = {"edge_index": self.edge_index_np, "edge_reactance": self.edge_reactance_np}
        with h5py.File(self.path, "r") as f:
            d = f["data"]
            for k in per_record:
                # self.idx is sorted-unique by construction, as h5py fancy-indexing requires
                out[k] = d[k][self.idx]  # one bulk gather per field -> [n, ...] numpy array
            if clean_want:
                out.update(self._clean_layers(clean_want, d["timestep"][self.idx]))
        return ArraysBundle.ordered(self._in_units(out))  # keeps the caller's field order

    def _in_units(self, out: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """The returned arrays in self.units: power and angle arrays converted to per unit when asked,
        masks, labels and the swing z-score untouched."""
        if self.units == "pu":
            for k, kind in _UNIT_KIND.items():
                if k in out:
                    out[k] = self._to_units(out[k], kind)
        return out

    def to_torch(
        self, fields: Optional[Sequence[str]] = None, device: Optional[Union[str, "torch.device"]] = None
    ) -> ArraysBundle:
        """Same data as to_numpy(), but as torch tensors (floats stay float32, label ids stay int64).
        Handy when you want the full split resident as tensors rather than streamed via a DataLoader."""
        torch = _torch()
        np_ = self.to_numpy(fields)  # single source-of-truth HDF5 read
        int_keys = {
            "family",
            "stealthy",
            "seq_id",
            "timestep",
            "edge_index",
        }  # -> int64; measurements -> float32
        out = {}
        for k, v in np_.items():
            t = torch.as_tensor(v)
            t = t.long() if k in int_keys else t.float()
            out[k] = t.to(device) if device else t  # optionally move onto the target device
        return ArraysBundle.ordered(out)

    def to_tf(self, fields: Optional[Sequence[str]] = None) -> ArraysBundle:
        """Same data as to_numpy(), but as TensorFlow tensors (requires tensorflow installed).
        Returns a dict of tf.Tensors; wrap in tf.data.Dataset.from_tensor_slices(...) if you want a pipeline."""
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError("TensorFlow is required for to_tf(): pip install tensorflow") from e
        # Same single to_numpy() read, wrapped as tf.Tensors.
        return ArraysBundle.ordered({k: tf.convert_to_tensor(v) for k, v in self.to_numpy(fields).items()})

    def to_pandas(self, flatten_features: bool = True) -> "pd.DataFrame":
        """Return a pandas DataFrame — one row per record — for tabular analysis / filtering.

        Always includes the metadata columns (family name, stealthy flag, seq_id, timestep, and n_attacked
        buses). Because the measurement graph is 3-D (records × buses × channels), `flatten_features=True`
        additionally spreads the per-bus/branch measurements into flat columns (V_b{n}, Pinj_b{n}, Pflow_e{n},
        …) so the whole split is a plain table; set it False for just the metadata (much narrower)."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for to_pandas(): pip install pandas") from e
        a = self.to_numpy()  # same single HDF5 read backing every exporter
        # Metadata frame, one row per record; n_attacked_buses = row-sum of the [n,N] label matrix.
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
        if flatten_features:
            N, E = self.N, self.E
            # node_x[:, :, ci] is [n, N] for channel ci -> N columns V_b0.., Pinj_b0.., etc.
            for ci, nm in enumerate(["V", "Pinj", "Qinj", "theta"]):
                cols = pd.DataFrame(a["node_x"][:, :, ci], columns=[f"{nm}_b{b}" for b in range(N)])
                df = pd.concat([df, cols], axis=1)
            # edge_x[:, :, ci] is [n, E] for channel ci -> E columns Pflow_e0.., Qflow_e0..
            for ci, nm in enumerate(["Pflow", "Qflow"]):
                cols = pd.DataFrame(a["edge_x"][:, :, ci], columns=[f"{nm}_e{e}" for e in range(E)])
                df = pd.concat([df, cols], axis=1)
            # per-bus binary attack labels -> N columns attacked_b0..attacked_b{N-1}
            label_cols = pd.DataFrame(a["y"].astype(int), columns=[f"attacked_b{b}" for b in range(N)])
            df = pd.concat([df, label_cols], axis=1)
        return df
