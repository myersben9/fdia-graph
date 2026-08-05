"""FdiaGraph: a PyTorch-ready dataset over one ml_only_ieee{C}.h5 shard.

The static graph (edge_index, edge_reactance) is read once; per-record tensors are sliced lazily so the
whole file is never loaded. Each item is the realistic PING-style measurement graph:
  node_x [N,4]=[|V|,P_inj,Q_inj,theta]  node_m [N,4] availability mask
  edge_x [E,2]=[P_from,Q_from]          edge_m [E,2] availability mask
  y [N] per-bus attack label  + family / stealthy / gap / seq_id / timestep
Masked measurements (mask==0) are already zeroed; the model consumes the masks.
"""
# numpy is the only hard dependency at import time; torch/h5py/pandas/tf are imported
# lazily inside methods so the package installs and imports without the heavy stack.
import numpy as np

# Integer attack-family code -> human name. These integers are the on-disk `data/family`
# values; the whole SDK speaks in these codes and only maps to names for display.
FAMILIES = {0: "benign", 1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 5: "At", 6: "Al"}
STEALTHY_FAMILIES = {1, 5, 6}          # Aq, At, Al — evade classical bad-data detection
_FAMILY_ALIAS = {"Ao": 1, "SLS": 1, "ramp": 5, "LRA": 6}   # backward-compatible family-name aliases
# On-disk `data/split` codes. Splitting is precomputed and stored, not derived here.
_SPLIT = {"train": 0, "val": 1, "test": 2}
_HELDOUT_TRAIN_EXCLUDE = {3, 4}        # As, Ar reserved for test-only in the unseen-attack protocol (Boyaci et al. 2022)


def _torch():
    # Lazy torch import + friendly install hint. Every method that needs tensors calls
    # this instead of importing torch at module top, so `import fdia_graph` stays light.
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("PyTorch is required: pip install 'fdia-graph[torch]'") from e


class FdiaGraph:
    """torch.utils.data.Dataset over an .h5 shard. Use `.loader(...)` for a ready DataLoader, or index items."""

    def __init__(self, path, split=None, families=None, include_gaps=False, heldout=False, format="torch",
                 units="physical"):
        # __init__ does exactly ONE full pass over the file's small metadata arrays
        # (family/gap/split) to decide which record rows this view keeps. The big
        # measurement arrays (node_x/edge_x/…) are never read here — only in __getitem__.
        import h5py
        self.path, self.format = path, format           # remember file + output flavor ("torch"/"pyg")
        # Unit system for the RETURNED measurements. The file stores engineering units — node_x = [V (p.u.),
        # P_inj (MW), Q_inj (MVAr), theta (deg)], edge flows in MW/MVAr — which is human-readable but mixes scales.
        # units="pu" converts on the fly (lossless, deterministic): P/Q and branch flows -> per-unit on baseMVA, theta
        # -> radians; V is already p.u. So the SAME shard serves both a physical view (plots, interpretation) and a
        # normalized view (ML training, physics), without duplicating the data. Note temporal_delta scales with power
        # (-> p.u. too); swing is a dimensionless z-score and is never rescaled.
        if units not in ("physical", "pu"):
            raise ValueError("units must be 'physical' or 'pu'")
        self.units = units
        self._f = None                                  # lazy per-worker h5py handle, opened on first __getitem__
        with h5py.File(path, "r") as f:                 # open read-only; closed at block exit (metadata is copied out)
            # `system` = which IEEE case (14/118/300); fall back to N, then 0, for older files.
            self.system = int(f.attrs.get("system", f.attrs.get("N", 0)))
            self.N = int(f.attrs["N"]); self.E = int(f.attrs["E"])   # bus count N, branch count E — fixed graph size
            # Power base for p.u. conversion. Stored on v0.4.1+ files; default 100 MVA (the case base for 14/118/300).
            self.baseMVA = float(f.attrs.get("baseMVA", 100.0))
            # Static graph, read ONCE and cached as numpy: edge_index [2,E] (from/to bus per branch),
            # edge_reactance [E]. Same for every record, so it lives on the object, not per-item.
            self.edge_index_np = f["graph/edge_index"][:].astype(np.int64)
            self.edge_reactance_np = f["graph/edge_reactance"][:].astype(np.float32)
            # v0.5.0+ full per-unit branch physics and bus shunts. Absent on older shards, so each is
            # optional and the attribute is None when the file predates the schema. has_physics lets a
            # caller branch once rather than probing every field.
            self._phys = {}
            for _k in ("edge_r", "edge_x", "edge_b", "edge_g", "edge_tap", "edge_shift",
                       "edge_status", "edge_is_trafo", "bus_shunt_g", "bus_shunt_b",
                       # v0.5.0 static per-bus attributes (type/limits/base_kv/zero-inj/attackable ...).
                       "bus_type", "bus_vmin", "bus_vmax", "bus_base_kv", "bus_is_zero_inj",
                       "bus_has_gen", "bus_base_pd", "bus_base_qd", "bus_attackable"):
                self._phys[_k] = f[f"graph/{_k}"][:].astype(np.float64) if f"graph/{_k}" in f else None
            self.has_physics = self._phys["edge_x"] is not None
            # Forward-compat (reserved for v0.6.0 mixed-topology shards): a PER-RECORD line status
            # data/edge_status [n_records, E] takes precedence over the static graph/edge_status when
            # present. v0.5.0 shards do not carry it, so this stays None today; the branch exists so
            # v0.5.0 loaders already read v0.6.0 shards correctly.
            self.edge_status_per_record = f["data/edge_status"][:] if "data/edge_status" in f else None
            self.has_temporal = "temporal_delta" in f["data"]      # v0.3+ ships a temporal-delta feature
            self.has_swing = "swing" in f["data"]                  # v0.4.1+ ships a windowed relative-swing feature
            # Copy the three tiny per-record metadata columns into RAM for filtering.
            fam = f["data/family"][:]; gap = f["data/gap"][:]      # family code + gap flag, one int per record
            sp = f["data/split"][:] if "data/split" in f else None # split code, or None on unsplit files
        # Build a boolean keep-mask over ALL records, then AND in each active filter.
        keep = np.ones(len(fam), bool)                  # start by keeping everything
        if not include_gaps:
            keep &= (gap == 0)                          # drop "gap" records (missing/skipped scans) unless asked to keep them
        if split is not None:
            if sp is None:
                raise ValueError(f"{path} has no split; run the split step first")
            keep &= (sp == _SPLIT[split])               # restrict to the requested train/val/test partition
            if heldout and _SPLIT[split] in (0, 1):        # unseen-attack protocol: drop As/Ar from train/val
                # Held-out protocol: As & Ar must be UNSEEN during training, so exclude them
                # from train(0)/val(1); test(2) still contains them to measure generalization.
                keep &= ~np.isin(fam, list(_HELDOUT_TRAIN_EXCLUDE))
        if families is not None:
            # `families` may be given as names ({"Aq","Ad"}, or legacy "SLS"/"Ao" aliases) or raw codes
            # ({1,2}); detect which by peeking at the first element, then normalize to int codes.
            if isinstance(next(iter(families)), str):
                fam_ids = [k for k, v in FAMILIES.items() if v in families] + [_FAMILY_ALIAS[n] for n in families if n in _FAMILY_ALIAS]
            else:
                fam_ids = list(families)
            keep &= np.isin(fam, fam_ids)               # keep only the requested attack families
        # Materialize the kept row positions. This is SORTED and UNIQUE by construction,
        # which later lets to_numpy() use h5py fancy-indexing directly. len(idx) == len(dataset).
        self.idx = np.nonzero(keep)[0]

    # ---- torch tensors for the static graph (lazy) ----
    # These wrap the cached numpy graph arrays as tensors on demand; computed fresh each
    # access (cheap, tiny) so no torch tensor is stored on the object.
    @property
    def edge_index(self):
        return _torch().as_tensor(self.edge_index_np)   # [2,E] long connectivity for message passing

    @property
    def edge_reactance(self):
        # DEPRECATED. Mixes ohms (lines) with vk_percent (transformers), so the two branch types are
        # not numerically comparable. Use edge_x, which is per unit for every branch.
        return _torch().as_tensor(self.edge_reactance_np)

    def _p(self, key):
        v = self._phys.get(key)
        if v is None:
            raise AttributeError(f"{key} needs a v0.5.0+ shard; this file predates the physics schema")
        return _torch().as_tensor(v)

    # Per-unit branch physics, ppc order (lines then transformers), aligned with edge_index.
    @property
    def edge_r(self):  return self._p("edge_r")            # series resistance
    @property
    def edge_x(self):  return self._p("edge_x")            # series reactance
    @property
    def edge_b(self):  return self._p("edge_b")            # charging susceptance
    @property
    def edge_g(self):  return self._p("edge_g")            # charging conductance, transformer iron losses
    @property
    def edge_tap(self):  return self._p("edge_tap")        # transformer turns ratio, 1.0 for lines
    @property
    def edge_shift(self):  return self._p("edge_shift")    # phase shift, degrees
    @property
    def edge_status(self):  return self._p("edge_status")  # 1 in service, 0 out (static; see edge_status_per_record)
    @property
    def edge_is_trafo(self):  return self._p("edge_is_trafo")
    # Static per-bus attributes, pandapower == ppc bus order (verified identity for case14/118/300).
    @property
    def bus_type(self):  return self._p("bus_type")            # 1 PQ, 2 PV, 3 reference
    @property
    def bus_vmin(self):  return self._p("bus_vmin")            # voltage limits, pu
    @property
    def bus_vmax(self):  return self._p("bus_vmax")
    @property
    def bus_base_kv(self):  return self._p("bus_base_kv")      # nominal voltage, kV
    @property
    def bus_is_zero_inj(self):  return self._p("bus_is_zero_inj")  # generator's own zero-injection set
    @property
    def bus_has_gen(self):  return self._p("bus_has_gen")      # generator or slack on the bus
    @property
    def bus_base_pd(self):  return self._p("bus_base_pd")      # base-case load, MW
    @property
    def bus_base_qd(self):  return self._p("bus_base_qd")      # base-case load, MVAr
    @property
    def bus_attackable(self):  return self._p("bus_attackable")  # carries |p_mw|>0 load (IEEE-300 141/183 rule)
    # Shunts are a BUS property, on the Ybus diagonal, so they were not expressible in an edge schema.
    @property
    def bus_shunt_g(self):  return self._p("bus_shunt_g")
    @property
    def bus_shunt_b(self):  return self._p("bus_shunt_b")

    @property
    def edge_attr(self):
        """[E,6] stacked per-unit branch features, the drop-in replacement for edge_reactance."""
        t = _torch()
        return t.stack([self.edge_r, self.edge_x, self.edge_b, self.edge_g,
                        self.edge_tap, self.edge_shift], dim=1)

    def _h(self):
        # Lazily open (and cache) the read-only h5py handle. Opening on first access —
        # not in __init__ — is what makes this fork-safe: each DataLoader worker process
        # opens its OWN handle, avoiding a shared-handle crash across processes.
        if self._f is None:
            import h5py
            self._f = h5py.File(self.path, "r")
        return self._f

    def __len__(self):
        return len(self.idx)                            # number of records this filtered view exposes

    def _to_units(self, arr, kind):
        """Convert a physical-unit array to self.units (no-op unless units=='pu'). kind:
        'node' [...,4]=[V,P,Q,theta] -> P,Q divided by baseMVA, theta deg->rad, V left (already p.u.);
        'edge' [...,2]=[P_from,Q_from] and 'td' [...,2]=[dP,dQ] -> divided by baseMVA (both are power).
        Returns a float32 copy so the on-disk data is never mutated."""
        if self.units != "pu":
            return arr
        b = self.baseMVA
        a = np.array(arr, dtype=np.float32, copy=True)
        if kind == "node":
            a[..., 1] /= b; a[..., 2] /= b                 # P_inj, Q_inj  MW/MVAr -> p.u.
            a[..., 3] = np.deg2rad(a[..., 3])              # theta  deg -> rad
        else:
            a /= b                                         # branch flows / temporal delta: power -> p.u.
        return a

    def __getitem__(self, i):
        # Build ONE record dict. `i` indexes this filtered view (0..len-1); map it back to
        # the true file row `j` via self.idx, then slice exactly that one row from each
        # dataset — h5py reads only row j off disk, never the whole array.
        torch = _torch()
        j = int(self.idx[i]); d = self._h()["data"]     # j = real file row; d = the "data" group of measurements
        item = dict(
            node_x=torch.as_tensor(self._to_units(d["node_x"][j], "node"), dtype=torch.float32),  # [N,4]=[|V|,P_inj,Q_inj,theta]
            node_m=torch.as_tensor(d["node_m"][j], dtype=torch.float32),   # [N,4] availability mask (1=measured)
            edge_x=torch.as_tensor(self._to_units(d["edge_x"][j], "edge"), dtype=torch.float32),   # [E,2]=[P_from,Q_from] flows
            edge_m=torch.as_tensor(d["edge_m"][j], dtype=torch.float32),   # [E,2] branch-flow availability mask
            y=torch.as_tensor(d["y"][j], dtype=torch.float32),            # [N] per-bus attack label (multi-label target)
            family=int(d["family"][j]), stealthy=int(d["stealthy"][j]),   # scalar metadata: attack type + stealth flag
            seq_id=int(d["seq_id"][j]), timestep=int(d["timestep"][j]))   # provenance: which sequence + which scan in it
        if self.has_temporal:                                     # [N,2] current-minus-previous-scan injection
            # Only present on v0.3+ files; lets temporal models see change-since-last-scan.
            item["temporal_delta"] = torch.as_tensor(self._to_units(d["temporal_delta"][j], "td"), dtype=torch.float32)
        if self.has_swing:                                        # [N,2] windowed relative swing (recent-window z-score)
            item["swing"] = torch.as_tensor(d["swing"][j], dtype=torch.float32)
        if self.format == "pyg":
            return self._to_pyg(item)                   # optionally repackage as a PyG Data object
        return item                                     # default: plain dict of tensors

    def _to_pyg(self, item):
        # Convert the dict into a torch_geometric Data object so PyG models/loaders can
        # consume it. The static edge_index/edge_reactance are shared across records; masks
        # ride along as extra attributes PyG will batch automatically.
        try:
            from torch_geometric.data import Data
        except ImportError as e:
            raise ImportError("PyG is required for format='pyg': pip install 'fdia-graph[pyg]'") from e
        return Data(x=item["node_x"], edge_index=self.edge_index,      # node features + connectivity
                    edge_attr=item["edge_x"], y=item["y"],            # edge features + labels
                    node_mask=item["node_m"], edge_mask=item["edge_m"],   # carry the availability masks through
                    family=item["family"], stealthy=item["stealthy"])     # keep metadata for stratified eval

    @staticmethod
    def collate(batch):
        # Custom collate for the plain-dict format: because every record shares the same
        # N and E, tensor fields stack cleanly into a leading batch dim; scalar metadata
        # becomes one long tensor. (PyG uses its own loader, so this is the dict path only.)
        torch = _torch()
        out = {}
        # Feature keys to stack; include temporal_delta only if this file carries it.
        fkeys = ["node_x", "node_m", "edge_x", "edge_m", "y"] + (["temporal_delta"] if "temporal_delta" in batch[0] else [])
        for k in fkeys:
            out[k] = torch.stack([b[k] for b in batch])           # [B, N/E, C] batched features/labels/masks
        for k in ("family", "stealthy", "seq_id", "timestep"):
            out[k] = torch.as_tensor([b[k] for b in batch], dtype=torch.long)   # [B] scalar metadata per record
        return out

    def loader(self, batch_size=64, shuffle=None, num_workers=0, **kw):
        """Return a ready DataLoader. Shuffle defaults on for train-like use; masks/labels included per batch."""
        torch = _torch()
        from torch.utils.data import DataLoader
        if shuffle is None:
            shuffle = True                              # default to shuffling (this dataset is used mostly for training)
        if self.format == "pyg":
            # PyG needs its OWN DataLoader (graph-aware batching that offsets edge_index);
            # no custom collate — Data objects know how to concatenate themselves.
            from torch_geometric.loader import DataLoader as PyGLoader
            return PyGLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kw)
        # Plain-dict path: standard torch DataLoader wired to our stacking collate above.
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=self.collate, **kw)

    def summary(self):
        # Cheap human-readable overview: reads only the family column, sub-selected to this
        # view's rows, and tallies how many records each present family contributes.
        import h5py
        with h5py.File(self.path, "r") as f:
            fam = f["data/family"][:][self.idx]         # family codes for just the kept rows
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
        # Decide which per-record arrays to pull: caller's `fields`, or the full default set
        # (measurements + labels + temporal_delta when present + scalar metadata).
        want = fields or (["node_x", "node_m", "edge_x", "edge_m", "y"] + (["temporal_delta"] if self.has_temporal else []) + (["swing"] if self.has_swing else [])
                          + ["family", "stealthy", "seq_id", "timestep"])
        idx = self.idx
        # Static graph arrays always included (tiny, and needed to interpret edges).
        out = {"edge_index": self.edge_index_np, "edge_reactance": self.edge_reactance_np}
        with h5py.File(self.path, "r") as f:
            d = f["data"]
            for k in want:
                # h5py fancy-indexing needs sorted, unique indices; self.idx is already sorted-unique by construction
                out[k] = d[k][idx]                      # one bulk gather per field -> [n, ...] numpy array
        # Apply the unit system to the power/angle measurement arrays (masks, labels, swing untouched).
        if self.units == "pu":
            if "node_x" in out: out["node_x"] = self._to_units(out["node_x"], "node")
            if "edge_x" in out: out["edge_x"] = self._to_units(out["edge_x"], "edge")
            if "temporal_delta" in out: out["temporal_delta"] = self._to_units(out["temporal_delta"], "td")
        return out

    def to_torch(self, fields=None, device=None):
        """Same data as to_numpy(), but as torch tensors (floats stay float32, label ids stay int64).
        Handy when you want the full split resident as tensors rather than streamed via a DataLoader."""
        torch = _torch()
        np_ = self.to_numpy(fields)                     # single source-of-truth HDF5 read; every view derives from it
        # These fields are indices/labels -> int64; everything else (measurements) -> float32.
        int_keys = {"family", "stealthy", "seq_id", "timestep", "edge_index"}
        out = {}
        for k, v in np_.items():
            t = torch.as_tensor(v)
            t = t.long() if k in int_keys else t.float()   # cast per key so dtypes are predictable
            out[k] = t.to(device) if device else t         # optionally move straight onto the target device
        return out

    def to_tf(self, fields=None):
        """Same data as to_numpy(), but as TensorFlow tensors (requires tensorflow installed).
        Returns a dict of tf.Tensors; wrap in tf.data.Dataset.from_tensor_slices(...) if you want a pipeline."""
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError("TensorFlow is required for to_tf(): pip install tensorflow") from e
        # Same single to_numpy() read, wrapped as tf.Tensors so TF users get an identical view.
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
        a = self.to_numpy()                             # same single HDF5 read backing every exporter
        # Metadata frame: one row per record. n_attacked_buses = row-sum of the [n,N] label matrix.
        df = pd.DataFrame({
            "family": [FAMILIES[int(k)] for k in a["family"]],   # readable family name
            "family_id": a["family"], "stealthy": a["stealthy"].astype(bool),
            "seq_id": a["seq_id"], "timestep": a["timestep"],
            "n_attacked_buses": a["y"].sum(axis=1).astype(int),  # how many buses attacked in this record
        })
        if flatten_features:
            N, E = self.N, self.E
            # spread each measurement channel across buses/branches into named columns (self-documenting headers)
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
