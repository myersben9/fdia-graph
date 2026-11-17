"""The attacker's area [WU26] Sec. III-A and the window ratings of the overload attack: the paper's own
area where it states one, the region its four principles pick otherwise (rules 1 and 2 as checks, 3 and 4
as a score of ours), the "delta" ratings matched to its Fig. 4 magnitudes, and the validity of every
attack the search returns (the goal met, every untampered channel within its noise, trusted buses true)."""

import numpy as np
import pytest

pytest.importorskip("pandapower")

from fdia_graph.constants.channels import NodeChannel  # noqa: E402
from fdia_graph.engine.attacks.area import area_score_total  # noqa: E402
from fdia_graph.engine.attacks.minimize import _Window  # noqa: E402
from fdia_graph.engine.core import FdiaGenerator  # noqa: E402
from fdia_graph.errors import ConfigError  # noqa: E402
from fdia_graph.generation import _load_states  # noqa: E402
from fdia_graph.models.config import (  # noqa: E402
    OverloadSettings,  # noqa: E402
    SearchSettings,
    TrustSchedule,
)
from fdia_graph.models.frames import FrameKnobs  # noqa: E402

from . import case_studies  # noqa: E402


@pytest.fixture(scope="module")
def g118():
    return FdiaGenerator(118, seed=1)


def _lines(g, system: int) -> list[int]:
    return [case_studies.branch(g, *pair) for pair in case_studies.SCENARIOS[system][0]]


def test_the_search_uses_the_area_it_is_given(g118):
    # [WU26] Sec. III-A Fig. 9
    area = case_studies.area(g118)
    states = list(_load_states(118, None)[:4])
    goal = g118.overload_goal(states, *_lines(g118, 118))
    k = FrameKnobs(area=area)
    got = g118.window_area(states, goal, k)
    assert set(area) <= set(got.tolist())  # the paper's buses, grown only over zero-injection boundaries


def test_the_rules_check_1_and_2_and_charge_for_size(g118):
    # [WU26] Sec. III-A rules 1-4
    states = np.stack(_load_states(118, None)[:10])
    lines = _lines(g118, 118)
    seeds = np.unique(g118.ei[:, lines])
    chosen = g118.rule_area(seeds, lines, states)
    assert chosen is not None and g118._connected(chosen)
    scores = [g118.area_score(g118.local_region(seeds, h), lines, states) for h in (1, 2, 3, 4)]
    best = max((s for s in scores if s.observable and s.connected), key=area_score_total)
    assert np.array_equal(chosen, best.buses)
    assert all(
        0.0 <= v <= 1.0 for s in scores for v in (s.load_share, s.spread, s.stability, s.observability)
    )
    assert not g118._connected(np.array([0, 117]))  # rule 2 fails on two far buses


def test_area_rule_is_a_validated_choice():
    assert SearchSettings(area_rule="rules").area_rule == "rules"
    with pytest.raises(ConfigError):
        SearchSettings(area_rule="nearest")


def test_delta_ratings_end_each_goal_at_its_flow_plus_the_step():
    # [WU26 eqs. 24-25]
    g = FdiaGenerator(14, seed=1)
    window = list(_load_states(14, None)[:6])
    g.use_line_ratings(OverloadSettings(rating_source="delta", rating_delta=0.1), np.stack(window))
    lines = _lines(g, 14)
    goal = g.overload_goal(window, *lines)
    flows = g.clean_flows_from_states(np.stack(window))
    want = np.hypot(flows[-1, lines, 0], flows[-1, lines, 1]) + 0.1 * g.case.base_mva
    assert np.allclose(goal.targets_at(len(window) - 1), want)
    with pytest.raises(ConfigError):
        OverloadSettings(rating_source="delta", rating_delta=-0.1)


def test_every_attack_the_search_returns_is_valid():
    # [WU26 eqs. 12, 24-25, 29]
    """Scenario 2 at k = 1.1 with the paper's trust schedule: the goal is met at every snapshot (the last
    at both ratings), no untampered channel moves beyond its noise, and a trusted bus is exactly true."""
    g = FdiaGenerator(14, seed=1)
    X = _load_states(14, None)
    lines = [case_studies.branch(g, *pair) for pair in case_studies.SCENARIOS[14][1]]
    flow = g.flow_metered.copy()
    flow[lines] = True
    flow[case_studies.branch(g, 6, 11)] = True
    g = case_studies.paper_metering(g, case_studies.PMUS[14], flow)
    window = list(X[:8])
    g.use_line_ratings(OverloadSettings(rating_margin=1.1), np.stack(window))
    k = FrameKnobs(2, g.operating_limits(X), True, 256, 1.0, 0.5, 2, area=case_studies.area(g))
    goal = g.overload_goal(window, *lines)
    trust = TrustSchedule([int(b) for b in case_studies.buses(g, [4, 6, 1, 13])], [1, 3, 5, 7])
    r = g.min_tamper(window, goal, k, trust=trust)
    assert r.devices >= 1
    w = _Window(g, window, goal, k, trust=trust)
    prev = w.prev
    for t in range(len(window)):
        free = w.support_at(t, r.support)
        Xa = g.goal_state(goal, t, window[t], free, k)
        assert Xa is not None
        reached = g.clean_flows_from_states(Xa[None])[0, lines]
        assert np.allclose(np.hypot(reached[:, 0], reached[:, 1]), goal.targets_at(t), atol=1e-6)
        for b in w.pinned[t]:
            assert np.array_equal(
                Xa[b, [NodeChannel.V, NodeChannel.THETA]], window[t][b, [NodeChannel.V, NodeChannel.THETA]]
            )
        node, edge, _, now = w._snapshot(t, free, prev)
        sig_node, sig_edge = w.sigma[t]
        quiet = (~node) & (w.node_m > 0)
        assert (np.abs(now.node)[quiet] <= sig_node[quiet] + 1e-12).all()
        quiet_e = (~edge) & (w.edge_m > 0)
        assert (np.abs(now.edge)[quiet_e] <= sig_edge[quiet_e] + 1e-12).all()
        prev = now


def test_delta_ratings_generate_am_episodes(tmp_path):
    """The timeline path takes the delta ratings: every eligible line is rated per window, so Am
    episodes are built (and the workers' generator carries the step)."""
    import h5py

    import fdia_graph as fg
    from fdia_graph import schema
    from fdia_graph.engine.core import FdiaGenerator
    from fdia_graph.models.config import MeterSettings
    from fdia_graph.models.plan import GeneratorSpec

    out = tmp_path / "delta.h5"
    am = OverloadSettings(rating_source="delta", rating_delta=0.1)
    fg.generate("ieee14", "delta", out=str(out), frames=300, families=["Am"], seed=3, am_attack=am)
    with h5py.File(out, "r") as f:
        assert len(f[schema.Group.EPISODES][schema.EPISODE_AM_EPISODE]) >= 1
    g = FdiaGenerator.from_spec(GeneratorSpec(14, 3, None, MeterSettings(), None, 10.0))
    assert g._rating_delta == 10.0
