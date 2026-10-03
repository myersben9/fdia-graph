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
# the PMU branch-current phasors of a hybrid-meter timeline [WU26 eqs. 19-20] [D10]; absent
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
EPISODE_MIN_EVALUATED = "min_evaluated"
EPISODE_MIN_SUPPORT_PTR, EPISODE_MIN_SUPPORT_IDX = "min_support_ptr", "min_support_idx"
# episodes/ for the overload attack Am [WU26]: one row per Am episode
EPISODE_AM_EPISODE, EPISODE_AM_LINE, EPISODE_AM_RATING = "am_episode", "am_line", "am_rating_mva"
EPISODE_AM_REACHED, EPISODE_AM_EMITTED = "am_reached_mva", "am_emitted_mva"
EPISODE_AM_TARGET = "am_target_mva"  # the goal at the window's end; one am_* row per target line [D17]

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
    # the split-first placement: the splits, the attacked fraction each reached, and per split (rows) and
    # family (columns, placed_families) the episodes requested, built, moved, dropped and short
    SPLIT_FRAC = "split_frac"
    SPLIT_SIZES = "split_sizes"
    SPLIT_ATTACKED_FRAC = "split_attacked_frac"
    PLACED_FAMILIES = "placed_families"
    EPISODES_REQUESTED = "episodes_requested"
    EPISODES_BUILT = "episodes_built"
    EPISODE_REDRAWS = "episode_redraws"
    EPISODES_DROPPED = "episodes_dropped"
    EPISODE_SHORTFALL = "episode_shortfall"
    JITTER_KEYED = "jitter_keyed"  # 1: every frame's jitter from its own (seed, timestep) stream
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
    RATING_SOURCE = "rating_source"  # the overload attack's line ratings [D15]
    RATING_MARGIN = "rating_margin"
    LOAD_CAP = "load_cap"  # the overload attack's load-plausibility cap [D16]
    N_LINES = "n_lines"  # the lines one overload episode drives at once [D17]
    SUPPORT_METHOD = "support_method"  # how the overload attack chose its supports: "search" or "rref"
    SETTINGS = "settings"  # the TimelineSettings the walk ran, as JSON (`TimelineSettings.as_record`)
    SETTINGS_HASH = "settings_hash"  # `results.config_hash` of those settings, what a result cites
    METER_MODEL = "meter_model"  # written on a hybrid-meter file only [D10]
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
from .models.data import Decision  # noqa: E402

# The partition codes stored in data/split.
SPLIT_CODE: dict[str, int] = {Split.TRAIN: 0, Split.VAL: 1, Split.TEST: 2}


# ---- the paper-to-code map ---------------------------------------------------------------------------
# What `tools/equation_map.py` indexes. Code cites an equation of [WU26] with one tag form,
# `[WU26 eq. 28]`, `[WU26 eqs. 21-23]` or `[WU26 Alg. 1]`, and one of our decisions with `[D9]` or
# `[E13]`, in the docstring or a comment of the function that implements it; a test cites what it
# checks the same way, or names it in its test name (`test_wu26_eq28_...`).

# [WU26]'s equations the package implements, by number, with what each states.
WU26_EQUATIONS: dict[str, str] = {
    "3": "PMU pseudo-measurements of the neighbouring voltages, from the branch currents (KKT form)",
    "12": "the attack: fewest tampered measurements over the window (l0)",
    "13": "SCADA active injection under attack",
    "14": "SCADA reactive injection under attack",
    "15": "SCADA active branch flow under attack",
    "16": "SCADA reactive branch flow under attack",
    "17": "PMU voltage magnitude under attack",
    "18": "PMU voltage angle under attack",
    "19": "PMU branch current, real part, under attack",
    "20": "PMU branch current, imaginary part, under attack",
    "21": "bus voltage limits",
    "22": "generator active output limits",
    "23": "generator reactive output limits",
    "24": "the target flow grows snapshot by snapshot",
    "25": "the target flow reaches its rating S_max over the window",
    "26": "the SE linearization with the secure and nonsecure PMU rows",
    "27": "the secure set h^S: a trusted PMU's own |V| and angle rows",
    "28": "the attack under defense: fewest nonsecure measurements",
    "29": "the trusted rows' state deviation is zero",
    "30": "the nonsecure set at t",
    "31": "trust accumulates: h^S_t = h^S_t-1 + dh^S_t",
    "32": "the rows one trusted PMU adds",
    "33": "the defense effect: the rise in the attack's cost",
    "34": "the DQN's greedy action",
    "35": "the DQN's loss over a mini-batch",
    "36": "the target Q value",
    "37": "the gradient step",
    "Alg. 1": "the DRL-based defense (training loop)",
}


_ATTACK, _DEFENSE = "docs/plans/WU_MSFDIA_PLAN.md, Decisions", "docs/plans/WU_DEFENSE_PLAN.md, Decisions"
DECISIONS: dict[str, Decision] = {
    "D1": Decision(
        "the attack cost counts devices (a SCADA terminal and a PMU per bus) moved beyond noise",
        "ours",
        _ATTACK,
    ),
    "D2": Decision("the fewest-tamper search runs for every generated family, At and Am", "ours", _ATTACK),
    "D3": Decision("PGLib-OPF v23.07 rate_a as the alternative line ratings", "ours", _ATTACK),
    "D4": Decision(
        "PMUs read the current phasor of every branch at their bus (eqs. 19-20)", "paper", _ATTACK
    ),
    "D5": Decision(
        "the overload goal (24)-(25) holds on the noiseless reading of the false state", "paper", _ATTACK
    ),
    "D6": Decision("generation makes the multi-snapshot At and Am only", "ours", _ATTACK),
    "D7": Decision("At's stealth bound and tamper count use the meters' rated accuracy", "ours", _ATTACK),
    "D8": Decision(
        "Am's tamper threshold is the paper's case-study noise, 0.03 pu SCADA, 0.01 pu PMU", "paper", _ATTACK
    ),
    "D9": Decision(
        "the goal adds the share k/T of the gap to the rating to each snapshot's true flow", "ours", _ATTACK
    ),
    "D10": Decision("hybrid meters: an angle only at a PMU, and PMU branch currents", "paper", _ATTACK),
    "D11": Decision("Am has no between-snapshot bound: noise only sets the l0 threshold", "paper", _ATTACK),
    "D12": Decision("what the meters measure is its own knob of the meter plan", "ours", _ATTACK),
    "D13": Decision(
        "the previous frame's PMU currents are offered for the eq. (3) pseudo-measurements", "ours", _ATTACK
    ),
    "D14": Decision("generators of the support move freely within (22)-(23)", "paper", _ATTACK),
    "D15": Decision("a line's rating defaults to 1.25 times its peak flow over the pool", "ours", _ATTACK),
    "D16": Decision(
        "the support's edge keeps (22)-(23), and a moved load changes by at most load_cap [YUA11]",
        "ours",
        _ATTACK,
    ),
    "D17": Decision(
        "an overload episode drives two lines by default, as the paper's case studies", "paper", _ATTACK
    ),
    "D18": Decision(
        "the attacker's area: the paper's where stated, else Sec. III-A's rules (1-2 checked, 3-4 our score)",
        "paper",
        _ATTACK,
    ),
    "E1": Decision(
        "trust is incremental: a trusted PMU keeps the offset its bus had before its slot (eqs. 29-30)",
        "paper",
        _DEFENSE,
    ),
    "E2": Decision("a trusted PMU pins its own |V| and angle only (eqs. 27, 32)", "paper", _DEFENSE),
    "E3": Decision(
        "the defense's cost is the devices tampered at any snapshot (Fig. 12); eq. (33)'s net l0 also reported",
        "paper",
        _DEFENSE,
    ),
    "E4": Decision("1-minute attack snapshots interpolated from the 5-minute pool", "ours", _DEFENSE),
    "E5": Decision("IEEE-118: 10 snapshots, one trust slot each (Figs. 10-11)", "paper", _DEFENSE),
    "E6": Decision("Algorithm 1 line 8 ends an episode with no reward, as written", "paper", _DEFENSE),
    "E7": Decision(
        "the DQN's state is Fig. 1's: the readings, the target lines' loading, the trusted set",
        "paper",
        _DEFENSE,
    ),
    "E8": Decision("IEEE-118's 100 tests run as 10 training sessions of 10 test windows", "ours", _DEFENSE),
    "E9": Decision(
        "the Q network is ours (two hidden layers of 128); its hyperparameters are the paper's",
        "ours",
        _DEFENSE,
    ),
    "E10": Decision(
        "detection with PMU support is the residual test with trusted PMUs reading truth", "ours", _DEFENSE
    ),
    "E11": Decision(
        "the single-snapshot trusted-meter classes stay as the linear analogue", "ours", _DEFENSE
    ),
    "E12": Decision("IEEE-1354 is not reproduced", "ours", _DEFENSE),
    "E13": Decision(
        "one support is held for the window (a support per trust slot never changed an answer, removed)",
        "ours",
        _DEFENSE,
    ),
    "E14": Decision(
        "reproduction ratings: each target's last flow plus 0.10 pu (Fig. 4's scale); k times the peak as a sensitivity",
        "ours",
        _DEFENSE,
    ),
    "E15": Decision("generation keeps the linear ramp of D9", "ours", _DEFENSE),
    "E16": Decision(
        "the attack minimizes the l1 of the window's summed attack, eq. (12) read literally ([37]'s l1)",
        "paper",
        _DEFENSE,
    ),
    "E17": Decision(
        "the reproduction meters every branch at both ends, each node's SCADA its incident flows ([29])",
        "paper",
        _DEFENSE,
    ),
    "E19": Decision(
        "the faithful attack's goal is (24)-(25) as written: the end at S_max, the flow never falling; "
        "D9's ramp stays for generation",
        "paper",
        _DEFENSE,
    ),
    "E18": Decision(
        "the reproduction's overload rho, per-snapshot l1 weight tau and trust region are ours",
        "ours",
        _DEFENSE,
    ),
}
