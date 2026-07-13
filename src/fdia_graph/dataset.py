"""FdiaGraph: a PyTorch-ready dataset over one ml_only_ieee{C}.h5 shard.

The static graph (edge_index, edge_reactance) is read once; per-record tensors are sliced lazily so the
whole file is never loaded. Each item is the realistic PING-style measurement graph:
  node_x [N,4]=[|V|,P_inj,Q_inj,theta]  node_m [N,4] availability mask
  edge_x [E,2]=[P_from,Q_from]          edge_m [E,2] availability mask
  y [N] per-bus attack label  + family / stealthy / gap / seq_id / timestep
Masked measurements (mask==0) are already zeroed; the model consumes the masks.
"""
import numpy as np

FAMILIES = {0: "benign", 1: "Ao", 2: "Ad", 3: "As", 4: "Ar", 5: "ramp", 6: "LRA"}
STEALTHY_FAMILIES = {1, 5, 6}          # Ao, ramp, LRA — evade classical bad-data detection
_SPLIT = {"train": 0, "val": 1, "test": 2}
_HELDOUT_TRAIN_EXCLUDE = {3, 4}        # As, Ar reserved for test-only in the unseen-attack protocol (Boyaci et al. 2022)


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("PyTorch is required: pip install 'fdia-graph[torch]'") from e


class FdiaGraph:
    """torch.utils.data.Dataset over an .h5 shard. Use `.loader(...)` for a ready DataLoader, or index items."""

    def __init__(self, path, split=None, families=None, include_gaps=False, heldout=False, format="torch"):
        import h5py
        self.path, self.format = path, format
        self._f = None
        with h5py.File(path, "r") as f:
            self.system = int(f.attrs.get("system", f.attrs.get("N", 0)))
            self.N = int(f.attrs["N"]); self.E = int(f.attrs["E"])
            self.edge_index_np = f["graph/edge_index"][:].astype(np.int64)
            self.edge_reactance_np = f["graph/edge_reactance"][:].astype(np.float32)
            self.has_temporal = "temporal_delta" in f["data"]      # v0.3+ ships a temporal-delta feature
            fam = f["data/family"][:]; gap = f["data/gap"][:]
            sp = f["data/split"][:] if "data/split" in f else None
        keep = np.ones(len(fam), bool)
        if not include_gaps:
            keep &= (gap == 0)
        if split is not None:
            if sp is None:
                raise ValueError(f"{path} has no split; run the split step first")
            keep &= (sp == _SPLIT[split])
            if heldout and _SPLIT[split] in (0, 1):        # unseen-attack protocol: drop As/Ar from train/val
                keep &= ~np.isin(fam, list(_HELDOUT_TRAIN_EXCLUDE))
        if families is not None:
            fam_ids = [k for k, v in FAMILIES.items() if v in families] if isinstance(next(iter(families)), str) else list(families)
            keep &= np.isin(fam, fam_ids)
        self.idx = np.nonzero(keep)[0]

    # ---- torch tensors for the static graph (lazy) ----
    @property
    def edge_index(self):
        return _torch().as_tensor(self.edge_index_np)

    @property
    def edge_reactance(self):
        return _torch().as_tensor(self.edge_reactance_np)

    def _h(self):
        if self._f is None:
            import h5py
            self._f = h5py.File(self.path, "r")
        return self._f

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        torch = _torch()
        j = int(self.idx[i]); d = self._h()["data"]
        item = dict(
            node_x=torch.as_tensor(d["node_x"][j], dtype=torch.float32),
            node_m=torch.as_tensor(d["node_m"][j], dtype=torch.float32),
            edge_x=torch.as_tensor(d["edge_x"][j], dtype=torch.float32),
            edge_m=torch.as_tensor(d["edge_m"][j], dtype=torch.float32),
            y=torch.as_tensor(d["y"][j], dtype=torch.float32),
            family=int(d["family"][j]), stealthy=int(d["stealthy"][j]),
            seq_id=int(d["seq_id"][j]), timestep=int(d["timestep"][j]))
        if self.has_temporal:                                     # [N,2] current-minus-previous-scan injection
            item["temporal_delta"] = torch.as_tensor(d["temporal_delta"][j], dtype=torch.float32)
        if self.format == "pyg":
            return self._to_pyg(item)
        return item

    def _to_pyg(self, item):
        try:
            from torch_geometric.data import Data
        except ImportError as e:
            raise ImportError("PyG is required for format='pyg': pip install 'fdia-graph[pyg]'") from e
        return Data(x=item["node_x"], edge_index=self.edge_index,
                    edge_attr=item["edge_x"], y=item["y"],
                    node_mask=item["node_m"], edge_mask=item["edge_m"],
                    family=item["family"], stealthy=item["stealthy"])

    @staticmethod
    def collate(batch):
        torch = _torch()
        out = {}
        fkeys = ["node_x", "node_m", "edge_x", "edge_m", "y"] + (["temporal_delta"] if "temporal_delta" in batch[0] else [])
        for k in fkeys:
            out[k] = torch.stack([b[k] for b in batch])
        for k in ("family", "stealthy", "seq_id", "timestep"):
            out[k] = torch.as_tensor([b[k] for b in batch], dtype=torch.long)
        return out

    def loader(self, batch_size=64, shuffle=None, num_workers=0, **kw):
        """Return a ready DataLoader. Shuffle defaults on for train-like use; masks/labels included per batch."""
        torch = _torch()
        from torch.utils.data import DataLoader
        if shuffle is None:
            shuffle = True
        if self.format == "pyg":
            from torch_geometric.loader import DataLoader as PyGLoader
            return PyGLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kw)
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=self.collate, **kw)

    def summary(self):
        import h5py
        with h5py.File(self.path, "r") as f:
            fam = f["data/family"][:][self.idx]
        return dict(system=self.system, N=self.N, E=self.E, n=len(self),
                    families={FAMILIES[k]: int((fam == k).sum()) for k in FAMILIES if (fam == k).any()})

    # ------------------------------------------------------------------ #
    #  Whole-split export in the framework the researcher prefers.
    #  __getitem__/.loader() stream one record at a time (good for training); the exporters below pull the
    #  ENTIRE filtered split into memory at once (good for analysis / non-PyTorch stacks). All of them share
    #  to_numpy() as the single HDF5 read so the different framework views are guaranteed identical.
    # ------------------------------------------------------------------ #
    def to_numpy(self, fields=None):
        """Return the whole selected split as a dict of numpy arrays (batched over records).

        Keys: node_x [n,N,4], node_m, edge_x [n,E,2], edge_m, y [n,N], family/stealthy/seq_id/timestep [n],
        plus the static graph: edge_index [2,E], edge_reactance [E]. `fields` optionally limits the per-record
        arrays read (the graph arrays are always included since they're tiny and needed to interpret edges).
        """
        import h5py
        want = fields or (["node_x", "node_m", "edge_x", "edge_m", "y"] + (["temporal_delta"] if self.has_temporal else [])
                          + ["family", "stealthy", "seq_id", "timestep"])
        idx = self.idx
        out = {"edge_index": self.edge_index_np, "edge_reactance": self.edge_reactance_np}
        with h5py.File(self.path, "r") as f:
            d = f["data"]
            for k in want:
                # h5py fancy-indexing needs sorted, unique indices; self.idx is already sorted-unique by construction
                out[k] = d[k][idx]
        return out

    def to_torch(self, fields=None, device=None):
        """Same data as to_numpy(), but as torch tensors (floats stay float32, label ids stay int64).
        Handy when you want the full split resident as tensors rather than streamed via a DataLoader."""
        torch = _torch()
        np_ = self.to_numpy(fields)
        int_keys = {"family", "stealthy", "seq_id", "timestep", "edge_index"}
        out = {}
        for k, v in np_.items():
            t = torch.as_tensor(v)
            t = t.long() if k in int_keys else t.float()
            out[k] = t.to(device) if device else t
        return out

    def to_tf(self, fields=None):
        """Same data as to_numpy(), but as TensorFlow tensors (requires tensorflow installed).
        Returns a dict of tf.Tensors; wrap in tf.data.Dataset.from_tensor_slices(...) if you want a pipeline."""
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError("TensorFlow is required for to_tf(): pip install tensorflow") from e
        return {k: tf.convert_to_tensor(v) for k, v in self.to_numpy(fields).items()}

    def to_pandas(self, flatten_features=True):
        """Return a pandas DataFrame — one row per record — for tabular analysis / filtering.

        Always includes the metadata columns (family name, stealthy flag, seq_id, timestep, and n_attacked
        buses). Because the measurement graph is 3-D (records × buses × channels), `flatten_features=True`
        additionally spreads the per-bus/branch measurements into flat columns (V_b{n}, Pinj_b{n}, Pflow_e{n},
        …) so the whole split is a plain table; set it False for just the metadata (much narrower)."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for to_pandas(): pip install pandas") from e
        a = self.to_numpy()
        df = pd.DataFrame({
            "family": [FAMILIES[int(k)] for k in a["family"]],
            "family_id": a["family"], "stealthy": a["stealthy"].astype(bool),
            "seq_id": a["seq_id"], "timestep": a["timestep"],
            "n_attacked_buses": a["y"].sum(axis=1).astype(int),
        })
        if flatten_features:
            N, E = self.N, self.E
            # spread each measurement channel across buses/branches into named columns (self-documenting headers)
            for ci, nm in enumerate(["V", "Pinj", "Qinj", "theta"]):
                cols = pd.DataFrame(a["node_x"][:, :, ci], columns=[f"{nm}_b{b}" for b in range(N)])
                df = pd.concat([df, cols], axis=1)
            for ci, nm in enumerate(["Pflow", "Qflow"]):
                cols = pd.DataFrame(a["edge_x"][:, :, ci], columns=[f"{nm}_e{e}" for e in range(E)])
                df = pd.concat([df, cols], axis=1)
            label_cols = pd.DataFrame(a["y"].astype(int), columns=[f"attacked_b{b}" for b in range(N)])
            df = pd.concat([df, label_cols], axis=1)
        return df
