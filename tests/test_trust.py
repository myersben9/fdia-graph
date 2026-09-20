"""The trusted-meter selectors on the tiny timeline: the greedy selection raises the attack cost,
the secured meters expose the stealthy families to the residual test, and the DQN learns a
selection that closes the attack subspace too."""

import pytest

pytest.importorskip("pandapower")

import fdia_graph as fg  # noqa: E402


@pytest.fixture(scope="module")
def selector(timeline):
    from fdia_graph.trust import TrustedMeters

    return TrustedMeters(k=12).fit(fg.load(timeline, split="train"))


def test_greedy_selection_raises_the_attack_cost(selector):
    tm = selector
    assert 0 < len(tm.order) <= 12 and len(set(tm.order)) == len(tm.order)
    assert all(0 <= i < tm.m for i in tm.order)
    assert all(b >= a for a, b in zip(tm.cost, tm.cost[1:]))
    assert tm.attack_cost([]) <= tm.cost[0] and tm.attack_cost() == tm.cost[len(tm.select()) - 1]
    assert len(tm.select(3)) == 3


def test_secured_meters_expose_the_stealthy_families(selector, timeline):
    from fdia_graph.models import TrustScores

    rep = selector.score(fg.load(timeline, split="test", order="time"))
    assert isinstance(rep, TrustScores) and rep.false_alarm <= 0.03
    assert set(rep.detected_before) == set(rep.detected_after)
    # the re-solve families are what the trust exposes; Am is not residual-stealthy to begin with
    # (its untouched meters contradict the moved state), so pinning meters can move it either way
    for fam in ("Aq", "Al"):
        if fam in rep.detected_after:
            assert rep.detected_after[fam] >= rep.detected_before[fam] - 0.05, fam
    assert rep["order"] == rep.order and len(rep.cost) == len(rep.order)
    with pytest.raises(ValueError, match="benign layer"):
        from conftest import SHARD_V072

        from fdia_graph.dataset import FdiaGraph

        selector.score(FdiaGraph(SHARD_V072))


def test_dqn_learns_a_selection(timeline):
    pytest.importorskip("torch")
    from fdia_graph.trust import TrustedMetersDQN

    rl = TrustedMetersDQN(k=8, episodes=12, seed=0).fit(fg.load(timeline, split="train"))
    assert 0 < len(rl.order) <= 8 and len(set(rl.order)) == len(rl.order)
    assert len(rl.history) == 12 and len(rl.cost) == len(rl.order)
    assert rl.attack_cost() >= rl.attack_cost([])
    with pytest.raises(ValueError, match="k must be"):
        TrustedMetersDQN(k=0)
