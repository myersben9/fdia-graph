"""One record at a time: unit conversion, `__getitem__` in dict/PyG form, `collate`, and a ready DataLoader."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from torch_geometric.data import Data


import numpy as np

from ..models.data import BatchBundle, RecordBundle
from .base import (
    _BATCH_SCALARS,
    _BATCH_STACKED,
    DatasetBase,
    _torch,
)


class RecordsMixin(DatasetBase):
    def _to_units(self, arr: np.ndarray, kind: str) -> np.ndarray:
        """Convert a physical-unit array to self.units (no-op unless units=='pu'). kind:
        'node' [...,4]=[V,P,Q,theta] -> P,Q divided by baseMVA, theta deg->rad, V left (already p.u.);
        'edge' [...,2]=[P_from,Q_from] and 'td' [...,2]=[dP,dQ] -> divided by baseMVA (both power).
        Returns a float32 copy so on-disk data is never mutated."""
        if self.units != "pu":
            return arr
        b = self.baseMVA
        a = np.array(arr, dtype=np.float32, copy=True)
        if kind == "node":
            a[..., 1] /= b
            a[..., 2] /= b  # P_inj, Q_inj  MW/MVAr -> p.u.
            a[..., 3] = np.deg2rad(a[..., 3])  # theta  deg -> rad
        else:
            a /= b  # branch flows / temporal delta: power -> p.u.
        return a

    def __getitem__(self, i: int) -> Union[RecordBundle, Data]:
        """One record as a dict of tensors (or a PyG Data with format="pyg"): the measurements and
        masks, the labels and provenance, and whichever optional layers the file carries."""
        d, j = self._record_source(i)
        item = self._base_item(d, j)
        self._add_optional_layers(item, d, j)
        record = RecordBundle(**item)
        return self._to_pyg(record) if self.format == "pyg" else record

    def _record_source(self, i: int) -> tuple[Any, int]:
        """Where record i is read from: the preloaded arrays (position-aligned with the view) or the
        file's data group at the real file row self.idx[i]."""
        if self._mem is not None:
            return self._mem, i
        return self._h()["data"], int(self.idx[i])

    def _base_item(self, d: Any, j: int) -> dict[str, Any]:
        """The fields every record has: the static graph (shared tensors, not copies), the measurements
        in self.units, the masks, the label and the scalar provenance."""
        torch = _torch()
        # edge_attr needs the v0.5.0+ physics schema, so attach it only when present.
        if not hasattr(self, "_ei_t"):
            self._ei_t = self.edge_index
            self._ea_t = self.edge_attr if self.has_physics else None
        item = dict(
            edge_index=self._ei_t,  # [2,E] static connectivity (same tensor every record)
            node_x=torch.as_tensor(self._to_units(d["node_x"][j], "node"), dtype=torch.float32),  # [N,4]
            node_m=torch.as_tensor(d["node_m"][j], dtype=torch.float32),  # [N,4] availability mask
            edge_x=torch.as_tensor(self._to_units(d["edge_x"][j], "edge"), dtype=torch.float32),  # [E,2]
            edge_m=torch.as_tensor(d["edge_m"][j], dtype=torch.float32),  # [E,2] flow availability mask
            y=torch.as_tensor(d["y"][j], dtype=torch.float32),  # [N] per-bus attack label
            family=int(d["family"][j]),
            stealthy=int(d["stealthy"][j]),
            seq_id=int(d["seq_id"][j]),
            timestep=int(d["timestep"][j]),
        )
        if self._ea_t is not None:
            item["edge_attr"] = self._ea_t  # [E,8] per-unit line features; v0.5.0+ shards only
        return item

    def _add_optional_layers(self, item: dict[str, Any], d: Any, j: int) -> None:
        """The layers a file may carry: the temporal features and the noiseless truth at the record's
        pool timestep (node, metered flows, flows on every branch)."""
        torch = _torch()
        if self.has_temporal:  # [N,2] current-minus-previous-scan injection (v0.3+)
            item["temporal_delta"] = torch.as_tensor(
                self._to_units(d["temporal_delta"][j], "td"), dtype=torch.float32
            )
        if self.has_swing:  # [N,2] windowed relative swing (recent-window z-score)
            item["swing"] = torch.as_tensor(d["swing"][j], dtype=torch.float32)
        if self._clean_np is None:
            return
        t = item["timestep"]
        # torch.tensor COPIES: with units="physical" _to_units is a no-op view into the shared cached
        # table, and an aliased tensor would let one record's in-place edit corrupt every other record.
        item["clean"] = torch.tensor(self._to_units(self._clean_np[t], "node"), dtype=torch.float32)
        if self._eclean_np is not None:  # [E,2] exact true flows (unmetered zeroed)
            item["edge_clean"] = torch.tensor(self._to_units(self._eclean_np[t], "edge"), dtype=torch.float32)
        ecf = self._clean_flows_full()
        if ecf is not None:  # [E,2] exact true flows on EVERY branch, metered or not
            item["edge_clean_full"] = torch.tensor(self._to_units(ecf[t], "edge"), dtype=torch.float32)

    # Dict-record key -> PyG attribute name. Everything not listed keeps its dict name, so a field
    # added to __getitem__ shows up in PyG records and batches without touching this method. The
    # static [E,8] branch physics travel as edge_phys because PyG's edge_attr slot holds the per-record
    # branch flows, the same contract torch_data.pyg_stream follows.
    _PYG_RENAME = {"node_x": "x", "node_m": "node_mask", "edge_m": "edge_mask", "edge_attr": "edge_phys"}

    def _to_pyg(self, item: dict[str, Any]) -> Data:
        # Repackage the dict record as a torch_geometric Data object, mechanically. PyG batches every
        # per-node/per-edge tensor along dim 0 and every scalar into a [B] tensor, so masks, temporal
        # features, clean layers and metadata all ride along.
        try:
            from torch_geometric.data import Data
        except ImportError as e:
            raise ImportError("PyG is required for format='pyg': pip install 'fdia-graph[pyg]'") from e
        fields = {self._PYG_RENAME.get(k, k): v for k, v in item.items()}
        fields["edge_attr"] = item["edge_x"]  # same tensor as fields["edge_x"], no copy until batching
        if self.slack is not None:
            fields["slack"] = self.slack  # reference-bus index, batched to [B] like family
        return Data(**fields)

    @staticmethod
    def collate(batch: list[dict[str, Any]]) -> BatchBundle:
        """Dict-format collate: every record shares N and E, so per-record tensors stack into a batch
        dimension, the static graph rides along once, and scalar metadata becomes one long tensor.
        (PyG has its own loader.)"""
        torch = _torch()
        out = {k: torch.stack([b[k] for b in batch]) for k in _BATCH_STACKED if k in batch[0]}
        out["edge_index"] = batch[0]["edge_index"]  # same topology for every sample
        if "edge_attr" in batch[0]:  # absent on older no-physics shards
            out["edge_attr"] = batch[0]["edge_attr"]
        for k in _BATCH_SCALARS:
            out[k] = torch.as_tensor([b[k] for b in batch], dtype=torch.long)  # [B] per record
        return BatchBundle(**out)

    def loader(
        self, batch_size: int = 64, shuffle: Optional[bool] = None, num_workers: int = 0, **kw: Any
    ) -> DataLoader:
        """Return a ready DataLoader. Shuffle defaults on for train-like use; masks/labels included per batch."""
        _torch()
        from torch.utils.data import DataLoader

        if shuffle is None:
            shuffle = True  # default on (mostly used for training)
        if self.format == "pyg":
            # PyG's own DataLoader does graph-aware batching (offsets edge_index); no custom collate needed.
            from torch_geometric.loader import DataLoader as PyGLoader

            return PyGLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kw)
        # Plain-dict path: standard torch DataLoader wired to our stacking collate above.
        # self is a structural Dataset (__len__/__getitem__) but can't inherit torch's Dataset,
        # since torch is an optional lazy import; pyright can't see the duck-typed conformance.
        return DataLoader(
            self,  # type: ignore[reportArgumentType]
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate,
            **kw,
        )
