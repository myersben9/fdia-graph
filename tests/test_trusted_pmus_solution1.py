"""[WU26]'s Solution 1 (`trust.TrustedPMUs`, p. 657 steps 1-5; docs/plans/WU_DEFENSE_PLAN.md PR C) on the
paper's IEEE-14 scenarios: one PMU per configuration step, each a PMU reading the sparsest attack that
moves a target line, scored by the trusted-PMU MDP.

The paper's sequences are 1, 4, 6, 13 (lines 3-4 and 6-11) and 4, 6, 1, 13 (lines 1-2 and 4-5). On 20 pool
frames at k = 1.1, Solution 1 here trusts 4, 6, 13, 1 and 4, 1, 6, 13: the first PMU matches the paper's
second scenario, and the final set, all four PMUs, is the same, so the defended cost is the search's
(6 devices and 11 channels, 9 and 28)."""

import os

import numpy as np
import pytest

pytest.importorskip("pandapower")

import test_trusted_pmus as tt  # noqa: E402

from fdia_graph.models import WuDefenseConfig  # noqa: E402
from fdia_graph.trust import TrustedPMUs, WuDefenseEnv  # noqa: E402

SLOW = pytest.mark.skipif(
    not os.environ.get("FDIA_SLOW"), reason="set FDIA_SLOW=1: 20-frame IEEE-14 searches"
)


def _env(scenario: int, frames: int, budget: int = 256) -> WuDefenseEnv:
    tt.WINDOW, window = frames, tt.WINDOW
    try:
        g, states, k, goal = tt._setup(scenario, 1.1)
    finally:
        tt.WINDOW = window
    pmus = [int(b) for b in g.wu26_buses([1, 4, 6, 13])]
    return WuDefenseEnv(g, states, goal, k._replace(min_budget=budget), WuDefenseConfig(pmus, (1, 3, 5, 7)))


def _numbers(env: WuDefenseEnv, order: list[int]) -> list[int]:
    number = env.g.base.bus["name"].astype(int).to_numpy()
    return [int(number[env.config.pmus[i]]) for i in order]


def test_solution_1_trusts_each_pmu_once_and_the_rewards_add_up():
    env = _env(1, 8)
    sol = TrustedPMUs(env).fit()
    assert len(sol.order) == len(set(sol.order)) <= 4
    assert sum(sol.rewards) == env.cost - env.undefended and sol.costs[-1] == env.cost
    assert sol.seconds < 10  # the reductions alone, not the searches that score them


def test_the_first_pmu_reads_the_sparsest_attack_on_a_target_line():
    # [WU26 eq. 26]
    """At the first step no PMU is trusted, and bus 4's PMU reads the sparsest attack that moves line 1-2
    or 4-5 (MATPOWER numbers)."""
    env = _env(1, 8)
    env.reset()
    assert _numbers(env, [TrustedPMUs(env).next_action()]) == [4]


@SLOW
@pytest.mark.parametrize("scenario, order, final", [(0, [4, 6, 13, 1], 11.0), (1, [4, 1, 6, 13], 28.0)])
def test_the_ieee14_sequences(scenario, order, final):
    env = _env(scenario, 20)
    sol = TrustedPMUs(env).fit()
    assert _numbers(env, sol.order) == order and env.cost == final
    assert np.isclose(sum(sol.rewards), final - env.undefended)
