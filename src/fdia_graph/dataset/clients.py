"""The timeline as federated clients: each utility sees only its own meters, buses and labels.

A client's data is `z[:, meters.owner == c]` and the labels of the buses it owns. A bus's label is set
only by the meters at that bus, so nothing of another client enters a view (no halo of neighbouring
meters). A view reads its arrays when it is made and holds no handle to the file. The operator, who
runs state estimation over every meter, gets the same view over the full scan.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from fdia_graph import schema
from fdia_graph.constants.meters import ANGLE_KINDS, BUS_KINDS, POWER_NOISE_FLOOR_MW
from fdia_graph.errors import NotATimeline
from fdia_graph.formulas.temporal import wrap_degrees
from fdia_graph.models.choices import Split
from fdia_graph.models.frames import MeterLayers
from fdia_graph.models.inputs import ReadableLayout
from fdia_graph.schema import Attr, Static
from fdia_graph.sensing.meters import MeterPlan
from fdia_graph.systems.ownership import Ownership


class ClientView:
    """One client's selection, read from the file when made: `z [T, M_c]` its observed scans, `labels
    [T, N_c]` the attack labels of its buses, `meters` its rows of the meter plan (bus positions
    global), `buses [N_c]` the bus positions it owns, and `frames` the selected frame indices
    (increasing; a week split has gaps where the other splits' weeks are)."""

    def __init__(self, z: np.ndarray, labels: np.ndarray, meters, buses: np.ndarray, frames: np.ndarray):
        self.z, self.labels, self.meters, self.buses, self.frames = z, labels, meters, buses, frames

    def node_features(self) -> np.ndarray:
        """`[T, N_c, 4]` per owned bus its |V| (p.u.), voltage angle (degrees), P and Q injection (MW,
        MVAr) as metered, NaN where the bus carries no meter of that kind (|V| at the voltmeter and PMU buses, the
        angle only at PMU buses)."""
        out = np.full((len(self.z), len(self.buses), len(BUS_KINDS)), np.nan, self.z.dtype)
        pos = {int(b): i for i, b in enumerate(self.buses)}
        for j, kind in enumerate(BUS_KINDS):
            rows = np.flatnonzero(self.meters.kind == kind)
            out[:, [pos[int(b)] for b in self.meters.bus[rows]], j] = self.z[:, rows]
        return out

    def temporal(self) -> np.ndarray:
        """`[T, M_c]` each meter's change from the previous selected frame, zero at the first frame and
        after a gap in the selection (the change would span other splits' weeks). The change of an
        angle reading (degrees) is wrapped to (-180, 180]; `z` itself stays raw."""
        d = np.zeros_like(self.z)
        d[1:] = np.diff(self.z, axis=0)
        d[1:][np.diff(self.frames) != 1] = 0.0
        angle = np.isin(self.meters.kind, ANGLE_KINDS)
        d[:, angle] = wrap_degrees(d[:, angle])
        return d


class Dataset:
    """An open timeline file written with the owners of its buses (`h`, read only). `clients` are
    the clients' names (the utilities of the New York case); `client(name)`
    is the view of one, `operator()` the view of every meter. `split` selects the train, val or test
    frames (all frames when None). `plan` is the meter table the file's masks describe, `z` the flat
    reading vector of that table (one column per meter), `own` the `Ownership` of the buses."""

    def __init__(self, path: str | Path):
        self.h = h5py.File(path, "r")
        if Attr.CLIENT_NAMES not in self.h.attrs:
            self.h.close()
            raise NotATimeline(
                f"{path} is not a timeline that records its clients (`SearchSource(..., own=...)`); a "
                "file in the earlier New York layout reads with fdia-graph 0.21 (pip install fdia-graph==0.21.0)"
            )
        try:  # the one layout: a file that declares another is refused, never misread
            ReadableLayout(
                str(path), **{k: str(self.h.attrs[k]) for k in schema.LAYOUT_ATTRS if k in self.h.attrs}
            )
        except NotATimeline:
            self.h.close()
            raise
        self.clients = str(self.h.attrs[Attr.CLIENT_NAMES]).split(",")
        edges = self.h[schema.EDGE_INDEX][:]
        bus_client = self.h[schema.path(schema.Group.GRAPH, Static.BUS_CLIENT)][:]
        self.own = Ownership(np.array(self.clients), bus_client, bus_client[edges[0]] != bus_client[edges[1]])
        masks = MeterLayers(*(self.h[name][0] for name in (schema.NODE_M, schema.EDGE_M, schema.PMU_I_M)))
        floor = float(self.h.attrs.get(Attr.POWER_FLOOR_MW, POWER_NOISE_FLOOR_MW))
        self.plan = MeterPlan.from_masks(masks, edges, bus_client, floor)

    def _frames(self, split: str | None) -> np.ndarray:
        """The frame indices of `split` exactly: week splits interleave, so a span from its first to its
        last frame would take in the other splits' weeks."""
        col = self.h[schema.SPLIT][:]
        return np.arange(len(col)) if split is None else np.flatnonzero(col == Split.values().index(split))

    def _rows(self, name: str, frames: np.ndarray) -> np.ndarray:
        """Dataset `name` at the increasing `frames`, read one contiguous run at a time."""
        ds = self.h[name]
        if not len(frames):
            return ds[0:0]
        cut = np.flatnonzero(np.diff(frames) != 1) + 1
        runs = zip(np.r_[0, cut], np.r_[cut, len(frames)])
        return np.concatenate([ds[int(frames[a]) : int(frames[b - 1]) + 1] for a, b in runs])

    def _view(self, meters: np.ndarray, buses: np.ndarray, split: str | None) -> ClientView:
        frames = self._frames(split)
        z = self.z(frames)[:, meters]
        labels = self._rows(schema.Y, frames)[:, buses].astype(bool)
        return ClientView(z, labels, self.plan.select(meters), buses, frames)

    def z(self, frames: np.ndarray, benign: bool = False) -> np.ndarray:
        """`[len(frames), M]` the flat reading vector of `plan` at the increasing `frames`, float32 as
        stored: the observed scans, or with `benign` the same scans without the attack."""
        if benign:
            return self._flat((schema.NODE_BENIGN, schema.EDGE_BENIGN, schema.PMU_I_BENIGN), frames)
        return self._flat((schema.NODE_X, schema.EDGE_X, schema.PMU_I), frames)

    def tampered(self, frames: np.ndarray) -> np.ndarray:
        """`[len(frames), M]` bool, the meters of `plan` the attack moves beyond their noise."""
        return self._flat((schema.NODE_TAMPER, schema.EDGE_TAMPER, schema.PMU_I_TAMPER), frames).astype(bool)

    def _flat(self, names: tuple[str, str, str], frames: np.ndarray) -> np.ndarray:
        """`[len(frames), M]` the flat vector of `plan` gathered from the bus, flow and PMU current layers
        `names` (`MeterPlan.gather`'s order)."""
        return self.plan.gather(*(self._rows(name, frames) for name in names))

    def client(self, name: str, split: str | None = None) -> ClientView:
        """Client `name`'s meters, the buses it owns and their labels, nothing of another client."""
        c = self.clients.index(name)
        meters, buses = np.flatnonzero(self.plan.owner == c), np.flatnonzero(self.own.bus_client == c)
        return self._view(meters, buses, split)

    def operator(self, split: str | None = None) -> ClientView:
        """Every meter (the full scan for state estimation) and every bus's label."""
        return self._view(np.arange(len(self.plan.kind)), np.arange(len(self.own.bus_client)), split)
