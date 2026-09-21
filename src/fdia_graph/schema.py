"""The file protocol: every group, dataset and attribute name an fdia-graph HDF5 file carries,
defined once. The writer (`timeline`, `generation`), the readers (`dataset`) and the tools agree
through these names and nowhere else; `tools/readability.py` refuses a path-shaped literal
("data/...", "graph/...") in any other module. Field names (`node_x`, `clean`, ...) are the
public vocabulary of a record and stay as words in the code; `FIELD_PATH` maps them here.
"""

from __future__ import annotations


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
# graph/
EDGE_INDEX = path(Group.GRAPH, "edge_index")
EDGE_REACTANCE = path(Group.GRAPH, "edge_reactance")  # the pre-0.5 branch feature, kept for old callers
STATIC_PHYSICS = (  # the per-unit branch physics and per-bus attributes of graph/ (v0.5.0+), optional each
    "edge_r",
    "edge_x",
    "edge_b",
    "edge_g",
    "edge_gs",
    "edge_bs",
    "edge_tap",
    "edge_shift",
    "edge_status",
    "edge_is_trafo",
    "bus_shunt_g",
    "bus_shunt_b",
    "bus_type",
    "bus_vmin",
    "bus_vmax",
    "bus_base_kv",
    "bus_is_zero_inj",
    "bus_has_gen",
    "bus_base_pd",
    "bus_base_qd",
    "bus_attackable",
)
# episodes/: one row per episode
EPISODE_ONSET, EPISODE_LENGTH, EPISODE_FAMILY = "onset", "length", "family"
EPISODE_BUS_PTR, EPISODE_BUS_IDX = "bus_ptr", "bus_idx"

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
}


class Attr:
    """File attribute keys (the header) and the values the readers test against."""

    KIND = "kind"  # "timeline" marks a timeline; absent on a v0.7.2 shard
    SYSTEM = "system"
    N = "N"
    E = "E"
    T = "T"
    BASEMVA = "baseMVA"
    SEED = "seed"
    FAMILIES = "families"
    ATTACKED_FRAC = "attacked_frac"
    N_EPISODES = "n_episodes"
    FALLBACK_BENIGN = "fallback_benign"
    TAMPER = "tamper"  # on attack/: what a 1 in the tamper masks means


KIND_TIMELINE = "timeline"

# The attack families, their codes and the names older releases used for them.
FAMILIES = {0: "benign", 1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 5: "At", 6: "Al", 7: "Am"}
STEALTHY_FAMILIES = {1, 5, 6, 7}  # Aq, At, Al, Am: local false states that pass the residual test
FAMILY_ALIAS = {"Ao": 1, "SLS": 1, "ramp": 5, "LRA": 6}  # backward-compatible family-name aliases
# The chronological partition codes stored in data/split.
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
