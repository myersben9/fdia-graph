"""One record at a time: unit conversion, `__getitem__` in dict/PyG form, `collate`, and a ready DataLoader."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    import h5py
    from torch.utils.data import DataLoader
    from torch_geometric.data import Data


import numpy as np

from .. import schema
from ..constants.channels import NodeChannel
from ..constants.records import ID_FIELDS, UNIT_KIND
from ..models.data import BatchBundle, RecordBundle
from .base import DatasetBase, _torch

if TYPE_CHECKING:
    # a record is read from the preloaded arrays or from the file's data group, indexed the same way
    RecordSource = Union[dict[str, np.ndarray], h5py.Group]


class RecordsMixin(DatasetBase):
    def _to_units(self, arr: np.ndarray, kind: str) -> np.ndarray:
        """A physical-unit array in self.units: unchanged unless units == "pu", then a float32 copy.
        `kind` "node": P and Q over baseMVA, theta degrees to radians; "edge" and "td": over baseMVA;
        "current": already per unit, as it is."""
        if self.units != "pu":
            return arr
        b = self.baseMVA
        a = np.array(arr, dtype=np.float32, copy=True)
        if kind == "current":
            return a
        if kind == "node":
            a[..., NodeChannel.P_INJ] /= b
            a[..., NodeChannel.Q_INJ] /= b  # P_inj, Q_inj  MW/MVAr -> p.u.
            a[..., NodeChannel.THETA] = np.deg2rad(a[..., NodeChannel.THETA])  # theta  deg -> rad
        else:
            a /= b  # branch flows / temporal delta: power -> p.u.
        return a

    def __getitem__(self, i: int) -> Union[RecordBundle, Data]:
        """One record as a dict of tensors (or a PyG Data with format="pyg"): the measurements and
        masks, the labels and provenance, and whichever optional layers the file carries."""
        pos = int(self._perm[i]) if self._perm is not None else i  # position in idx (the view order)
        d, j = self._record_source(pos)
        item = self._base_item(d, j)
        self._add_optional_layers(item, d, j)
        self._add_frame_layers(item, pos)
        record = RecordBundle(**item)
        return self._to_pyg(record) if self.format == "pyg" else record

    def _record_source(self, pos: int) -> tuple[RecordSource, int]:
        """Where the record at position `pos` of idx is read from: the preloaded arrays (aligned with
        idx) or the file's data group at the real file row self.idx[pos]."""
        if self._mem is not None:
            return self._mem, pos
        return self._h()[schema.Group.DATA], int(self.idx[pos])

    def _add_frame_layers(self, item: dict[str, Any], pos: int) -> None:
        """The layers a timeline stores per frame beside data/, in self.units: the attack-removed
        scan (noise kept) and, on a hybrid-meter file, the PMU branch-current phasors [E, 4], their
        mask and their attack-removed twin."""
        torch = _torch()
        for k, path in self._frame_layers().items():
            a = self._mem[k][pos] if self._mem is not None else self._h()[path][int(self.idx[pos])]
            a = self._to_units(a, UNIT_KIND[k]) if k in UNIT_KIND else a  # a mask has no unit
            item[k] = torch.as_tensor(np.asarray(a), dtype=torch.float32)

    def _base_item(self, d: RecordSource, j: int) -> dict[str, Any]:
        """The fields every record has: the static graph (shared tensors, not copies), the measurements
        in self.units, the masks, the label and the scalar provenance."""
        torch = _torch()
        # edge_attr needs the branch physics, so it is attached only when the file carries them.
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

    def _add_optional_layers(self, item: dict[str, Any], d: RecordSource, j: int) -> None:
        """The layers a file may carry: the temporal features and the noiseless truth at the record's
        pool timestep (node, metered flows, flows on every branch)."""
        torch = _torch()
        if self.has_temporal:  # [N,2] current-minus-previous-scan injection (v0.3+)
            delta = self._to_units(d["temporal_delta"][j], "td")
            item["temporal_delta"] = torch.as_tensor(delta, dtype=torch.float32)
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

    # Dict-record key -> PyG attribute name; a key not listed keeps its name. The static [E,8] branch
    # physics travel as edge_phys because PyG's edge_attr slot holds the per-record branch flows.
    _PYG_RENAME = {"node_x": "x", "node_m": "node_mask", "edge_m": "edge_mask", "edge_attr": "edge_phys"}

    def _to_pyg(self, item: RecordBundle) -> Data:
        # The dict record as a torch_geometric Data object. PyG batches every per-node and per-edge
        # tensor along dim 0 and every scalar into a [B] tensor, so every field rides along.
        try:
            from torch_geometric.data import Data
        except ImportError as e:
            raise ImportError("PyG is required for format='pyg': pip install 'fdia-graph[pyg]'") from e
        fields: dict[str, object] = {self._PYG_RENAME.get(str(k), str(k)): v for k, v in item.items()}
        fields["edge_attr"] = item["edge_x"]  # same tensor as fields["edge_x"], no copy until batching
        if self.slack is not None:
            fields["slack"] = self.slack  # reference-bus index, batched to [B] like family
        return Data(**fields)

    @staticmethod
    def collate(batch: Sequence[RecordBundle]) -> BatchBundle:
        """Dict-format collate: per-record tensors stack into a batch dimension, the static graph is
        kept once, and each scalar field becomes one long tensor [B]. (PyG has its own loader.)"""
        torch = _torch()
        once = ("edge_index", "edge_attr", *ID_FIELDS)  # every other field of a batch is stacked
        stacked = [k for k in BatchBundle._order if k not in once and k in batch[0]]
        out = {k: torch.stack([b[k] for b in batch]) for k in stacked}
        out["edge_index"] = batch[0]["edge_index"]  # same topology for every sample
        if "edge_attr" in batch[0]:  # absent on older no-physics shards
            out["edge_attr"] = batch[0]["edge_attr"]
        for k in ID_FIELDS:
            out[k] = torch.as_tensor([b[k] for b in batch], dtype=torch.long)  # [B] per record
        return BatchBundle(**out)

    def loader(
        self, batch_size: int = 64, shuffle: Optional[bool] = None, num_workers: int = 0, **kw: Any
    ) -> DataLoader:
        """A ready DataLoader (PyG's own with format="pyg"). `shuffle` defaults to True."""
        _torch()
        from torch.utils.data import DataLoader

        if shuffle is None:
            shuffle = True  # default on (mostly used for training)
        if self.format == "pyg":
            # PyG's DataLoader batches graphs itself (it offsets edge_index).
            from torch_geometric.loader import DataLoader as PyGLoader

            return PyGLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kw)
        # self has __len__ and __getitem__ but cannot inherit torch's Dataset (torch is an optional,
        # lazy import), so pyright cannot see that it conforms.
        return DataLoader(
            self,  # type: ignore[reportArgumentType]
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate,
            **kw,
        )
