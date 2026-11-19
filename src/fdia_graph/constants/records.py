"""The fields of a record, by group: the names a record, a batch and an export carry, in their order.

The names are the public vocabulary of the loader (`rec["node_x"]`, `ds.export(["clean"])`); where
each one is stored in a file is `schema.FIELD_PATH`.
"""

from __future__ import annotations

SCAN_FIELDS = ("node_x", "node_m", "edge_x", "edge_m")  # one scan: the readings and their meter masks
ID_FIELDS = ("family", "stealthy", "seq_id", "timestep")  # one whole number per record: label and origin
TEMPORAL_FIELDS = ("temporal_delta", "swing")  # the two temporal features
# The noiseless truth, stored once per pool timestep and resolved through a record's timestep.
CLEAN_FIELDS = ("clean", "edge_clean", "edge_clean_full")
BENIGN_FIELDS = ("benign", "edge_benign")  # the attack-removed layer, one row per frame
# The optional layers of a record, in the order every bundle lists them.
LAYER_FIELDS = (*TEMPORAL_FIELDS, *CLEAN_FIELDS, *BENIGN_FIELDS)
# The PMU branch-current layers of a hybrid-meter timeline, one row per frame.
CURRENT_FIELDS = ("pmu_i", "pmu_i_m", "pmu_i_benign")
# The previous frame's readings (the row emitted just before each record, whatever its split or
# family): the field an export offers on request -> the field it is read from. `prev_pmu_i` exists on
# hybrid-meter timelines only.
PREV_SOURCE_FIELD = {
    "prev_node_x": "node_x",
    "prev_edge_x": "edge_x",
    "prev_timestep": "timestep",
    "prev_swing": "swing",
    "prev_pmu_i": "pmu_i",
}
# The previous-frame fields every timeline can offer: all but the PMU currents.
COMMON_PREV_FIELDS = tuple(k for k in PREV_SOURCE_FIELD if k != "prev_pmu_i")
# The unit conversion each returned array takes under units="pu": "node" (P and Q over baseMVA, the
# angle to radians), "edge" and "td" (over baseMVA), "current" (already per unit: unchanged).
# A mask, a label and the swing take none.
UNIT_KIND = {
    "node_x": "node",
    "prev_node_x": "node",
    "clean": "node",
    "benign": "node",
    "edge_x": "edge",
    "prev_edge_x": "edge",
    "edge_clean": "edge",
    "edge_clean_full": "edge",
    "edge_benign": "edge",
    "temporal_delta": "td",
    "pmu_i": "current",
    "pmu_i_benign": "current",
    "prev_pmu_i": "current",
}
# The fields exported as int64 tensors; every other field is float32.
INT_FIELDS = frozenset({*ID_FIELDS, "prev_timestep", "edge_index"})
# The pandas column of each flow in a file's `edge_feat` legend.
FLOW_COLUMNS = {"P_from": "Pflow", "Q_from": "Qflow"}
