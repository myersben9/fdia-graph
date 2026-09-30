"""The file protocol: every group, dataset and attribute name an fdia-graph HDF5 file carries,
defined once. The writer (`timeline`, `generation`), the readers (`dataset`) and the tools agree
through these names and nowhere else; `tools/readability.py` refuses a path-shaped literal
("data/...", "graph/...") in any other module. Field names (`node_x`, `clean`, ...) are the
public vocabulary of a record and stay as words in the code; `FIELD_PATH` maps them here.
"""

from __future__ import annotations

from .models.choices import (  # noqa: F401  re-exported beside the code that reads them
    Split,
)


class Group:
    """The six groups of a timeline file (a v0.7.2 shard has no benign/, episodes/ or attack/)."""

    DATA = "data"  # one row per record: measurements, masks, labels, provenance, temporal features
    BENIGN = "benign"  # the attack-removed measurement layers, one row per frame
    CLEAN = "clean"  # the noiseless true state, once per pool timestep
    GRAPH = "graph"  # the static topology and branch physics
    EPISODES = "episodes"  # the attack episode table, buses ragged
    ATTACK = "attack"  # designed magnitudes (ragged) and tamper masks


def path(group: str, name: str) -> str:
    """The dataset path of `name` in `group`, the one way a path is spelled."""
    return f"{group}/{name}"


# data/: per-record datasets
NODE_X = path(Group.DATA, "node_x")
EDGE_X = path(Group.DATA, "edge_x")
NODE_M = path(Group.DATA, "node_m")
EDGE_M = path(Group.DATA, "edge_m")
Y = path(Group.DATA, "y")
FAMILY = path(Group.DATA, "family")
STEALTHY = path(Group.DATA, "stealthy")
SEQ_ID = path(Group.DATA, "seq_id")
TIMESTEP = path(Group.DATA, "timestep")
SPLIT = path(Group.DATA, "split")
TEMPORAL_DELTA = path(Group.DATA, "temporal_delta")
SWING = path(Group.DATA, "swing")
GAP = path(Group.DATA, "gap")  # shards only: the gap rows between pool timesteps
EDGE_STATUS = path(Group.DATA, "edge_status")  # v0.6.0 shards: per-record branch status
# benign/, clean/, attack/: per-frame layers
NODE_BENIGN = path(Group.BENIGN, "node_benign")
EDGE_BENIGN = path(Group.BENIGN, "edge_benign")
NODE_CLEAN = path(Group.CLEAN, "node_clean")
EDGE_CLEAN = path(Group.CLEAN, "edge_clean")
NODE_TAMPER = path(Group.ATTACK, "node_tamper")
EDGE_TAMPER = path(Group.ATTACK, "edge_tamper")
MAG_PTR = path(Group.ATTACK, "mag_ptr")
MAG_BUS = path(Group.ATTACK, "mag_bus")
MAG = path(Group.ATTACK, "mag")
# the PMU branch-current phasors of a hybrid-meter timeline [WU26, eqs. 19-20] (the plan's D10); absent
# from a v0.8.3-meter file
PMU_I = path(Group.DATA, "pmu_i")
PMU_I_M = path(Group.DATA, "pmu_i_m")
PMU_I_BENIGN = path(Group.BENIGN, "pmu_i_benign")
PMU_I_TAMPER = path(Group.ATTACK, "pmu_i_tamper")
CURRENT_LAYERS = (PMU_I, PMU_I_M, PMU_I_BENIGN, PMU_I_TAMPER)  # all of them or none, by meter model
# graph/
EDGE_INDEX = path(Group.GRAPH, "edge_index")
EDGE_REACTANCE = path(Group.GRAPH, "edge_reactance")  # the pre-0.5 branch feature, kept for old callers


class Static:
    """The per-unit branch physics and per-bus attributes of graph/ (v0.5.0+), each optional in a
    file; `STATIC_PHYSICS` is the reader's table of them in order."""

    EDGE_R = "edge_r"
    EDGE_X = "edge_x"  # the series reactance (`ds.branch_x`), not the flow layer data/edge_x
    EDGE_B = "edge_b"
    EDGE_G = "edge_g"
    EDGE_GS = "edge_gs"
    EDGE_BS = "edge_bs"
    EDGE_TAP = "edge_tap"
    EDGE_SHIFT = "edge_shift"
    EDGE_STATUS = "edge_status"
    EDGE_IS_TRAFO = "edge_is_trafo"
    BUS_SHUNT_G = "bus_shunt_g"
    BUS_SHUNT_B = "bus_shunt_b"
    BUS_TYPE = "bus_type"
    BUS_VMIN = "bus_vmin"
    BUS_VMAX = "bus_vmax"
    BUS_BASE_KV = "bus_base_kv"
    BUS_IS_ZERO_INJ = "bus_is_zero_inj"
    BUS_HAS_GEN = "bus_has_gen"
    BUS_BASE_PD = "bus_base_pd"
    BUS_BASE_QD = "bus_base_qd"
    BUS_ATTACKABLE = "bus_attackable"


STATIC_PHYSICS = tuple(v for k, v in vars(Static).items() if not k.startswith("_"))
# episodes/: one row per episode
EPISODE_ONSET, EPISODE_LENGTH, EPISODE_FAMILY = "onset", "length", "family"
EPISODE_BUS_PTR, EPISODE_BUS_IDX = "bus_ptr", "bus_idx"
# episodes/ with the fewest-tamper knob: one row per episode the search ran on
EPISODE_MIN_EPISODE, EPISODE_MIN_DEVICES, EPISODE_MIN_CHANNELS = "min_episode", "min_devices", "min_channels"
EPISODE_MIN_PROVEN, EPISODE_MIN_EVALUATED, EPISODE_MIN_LOWER = (
    "min_proven",
    "min_evaluated",
    "min_lower_bound",
)
EPISODE_MIN_SUPPORT_PTR, EPISODE_MIN_SUPPORT_IDX = "min_support_ptr", "min_support_idx"
EPISODE_MIN_UNSOLVED = "min_unsolved"
# episodes/ for the overload attack Am [WU26]: one row per Am episode
EPISODE_AM_EPISODE, EPISODE_AM_LINE, EPISODE_AM_RATING = "am_episode", "am_line", "am_rating_mva"
EPISODE_AM_REACHED, EPISODE_AM_EMITTED = "am_reached_mva", "am_emitted_mva"
EPISODE_AM_TARGET = "am_target_mva"  # the goal at the window's end; one am_* row per target line (D17)

# record field -> dataset path: the loader's vocabulary on disk (clean fields resolve per timestep)
FIELD_PATH = {
    "node_x": NODE_X,
    "node_m": NODE_M,
    "edge_x": EDGE_X,
    "edge_m": EDGE_M,
    "y": Y,
    "family": FAMILY,
    "stealthy": STEALTHY,
    "seq_id": SEQ_ID,
    "timestep": TIMESTEP,
    "temporal_delta": TEMPORAL_DELTA,
    "swing": SWING,
    "benign": NODE_BENIGN,
    "edge_benign": EDGE_BENIGN,
    "clean": NODE_CLEAN,
    "edge_clean": EDGE_CLEAN,
    "pmu_i": PMU_I,
    "pmu_i_m": PMU_I_M,
    "pmu_i_benign": PMU_I_BENIGN,
}


class Attr:
    """Every attribute key a writer emits: the header the readers test, the provenance and unit
    legends, the timeline's own counts, the generation knobs, and the graph/ group's legends."""

    # the header
    KIND = "kind"  # "timeline" marks a timeline; absent on a v0.7.2 shard
    SYSTEM = "system"
    N = "N"
    E = "E"
    T = "T"
    N_RECORDS = "n_records"
    BASEMVA = "baseMVA"
    SEED = "seed"
    # legends and provenance
    NODE_FEAT = "node_feat"
    EDGE_FEAT = "edge_feat"
    NODE_UNITS = "node_units"
    EDGE_UNITS = "edge_units"
    TOPOLOGY = "topology"
    OUTAGE_LINE = "outage_line"
    OUTAGE_BRANCH_POS = "outage_branch_pos"
    OUTAGE_LINE_NAME = "outage_line_name"
    OUTAGE_FROM_BUS = "outage_from_bus"
    OUTAGE_TO_BUS = "outage_to_bus"
    OUTAGE_BASE_FLOW_MW = "outage_base_flow_mw"
    # the timeline's own counts
    FAMILIES = "families"
    ATTACKED_FRAC = "attacked_frac"
    N_EPISODES = "n_episodes"
    FALLBACK_BENIGN = "fallback_benign"
    # the generation knobs a timeline records
    TARGET_ATTACKED_FRAC = "target_attacked_frac"
    RAMP_RATE = "ramp_rate"
    RAMP_LEN = "ramp_len"
    AM_LEN = "am_len"
    HOPS = "hops"
    MIN_TAMPER = "min_tamper"
    MIN_BUDGET = "min_budget"
    AM_ATTACK = "am_attack"
    STEALTH_SCALE = "stealth_scale"
    RATING_SOURCE = "rating_source"  # the overload attack's line ratings (D15)
    RATING_MARGIN = "rating_margin"
    LOAD_CAP = "load_cap"  # the overload attack's load-plausibility cap (D16)
    N_LINES = "n_lines"  # the lines one overload episode drives at once (D17)
    SUPPORT_METHOD = "support_method"  # how the overload attack chose its supports: "search" or "rref"
    METER_MODEL = "meter_model"  # written on a hybrid-meter file only (the plan's D10)
    CURRENT_FEAT = "current_feat"  # the legend of pmu_i, on a hybrid-meter file
    CURRENT_UNITS = "current_units"
    MAX_LOAD_MW = "max_load_mw"
    V_LO = "v_lo"  # the widest bus voltage limits of the case, what a false state must stay in
    V_HI = "v_hi"
    VBUS_FRAC = "vbus_frac"
    PMU_FRAC = "pmu_frac"
    FLOW_FRAC = "flow_frac"
    # on graph/ and attack/
    EDGE_FEAT_STATIC = "edge_feat_static"
    BUS_FEAT_STATIC = "bus_feat_static"
    EDGE_REACTANCE_DEPRECATED = "edge_reactance_deprecated"
    YBUS_RECONSTRUCTIBLE = "ybus_reconstructible"
    TAMPER = "tamper"  # on attack/: what a 1 in the tamper masks means


ATTR_KEYS = frozenset(v for k, v in vars(Attr).items() if not k.startswith("_"))

KIND_TIMELINE = "timeline"

# The attack families, their codes and the names older releases used for them (models.choices).
from .models.choices import FAMILIES, FAMILY_ALIAS, STEALTHY_FAMILIES  # noqa: E402,F401

# The partition codes stored in data/split.
SPLIT_CODE: dict[str, int] = {Split.TRAIN: 0, Split.VAL: 1, Split.TEST: 2}
