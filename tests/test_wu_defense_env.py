"""[WU26]'s trusted-PMU MDP (`trust.WuDefenseEnv`, docs/plans/WU_DEFENSE_PLAN.md PR B) on the paper's
IEEE-14 scenario for lines 1-2 and 4-5: the configuration model, the reward as the rise in the search's
cost under the schedule (eq. 33), the cache, the state of Fig. 1, and the paper's line-8 break.

A short window keeps the searches quick: 8 pool frames, the PMUs trusted at snapshots 1, 3, 5 and 7
(zero-based, the paper's 2, 4, 6, 8), ratings 1.1 times each target line's peak flow."""

import numpy as np
import pytest

pytest.importorskip("pandapower")

import test_trusted_pmus as tt  # noqa: E402

from fdia_graph.models import WuDefenseConfig  # noqa: E402
from fdia_graph.models.frames import MinimizerResult  # noqa: E402
from fdia_graph.models.validation import ConfigError  # noqa: E402
from fdia_graph.trust import WuDefenseEnv  # noqa: E402

SLOTS = (1, 3, 5, 7)


@pytest.fixture(scope="module")
def env():
    tt.WINDOW, window = 8, tt.WINDOW
    try:
        g, states, k, goal = tt._setup(1, 1.1)
    finally:
        tt.WINDOW = window
    pmus = [int(b) for b in g.wu26_buses([1, 4, 6, 13])]
    return WuDefenseEnv(g, states, goal, k._replace(min_budget=256), WuDefenseConfig(pmus, SLOTS))


@pytest.mark.parametrize(
    "bad",
    [
        dict(pmus=[], slots=[1]),
        dict(pmus=[0, 0], slots=[1, 2]),
        dict(pmus=[0, 3], slots=[2, 1]),
        dict(pmus=[0], slots=[1, 2]),
        dict(pmus=[0, 3], slots=[1, 2], unit="watts"),
        dict(pmus=[0.5], slots=[1]),
    ],
)
def test_the_configuration_refuses_what_it_cannot_mean(bad):
    with pytest.raises(ConfigError):
        WuDefenseConfig(**bad)


def test_a_candidate_must_carry_a_pmu(env):
    bus2 = int(env.g.wu26_buses([2])[0])  # no PMU in the paper's IEEE-14 plan
    with pytest.raises(ConfigError):
        WuDefenseEnv(env.g, env.states, env.goal, env.k, WuDefenseConfig([bus2], (1,)))


def test_the_reward_is_the_rise_in_the_search_cost(env):
    """Each step's reward is the search's channel count under the longer schedule less the one before
    (eq. 33 per step), and the rewards of an episode sum to the defended cost less the undefended one."""
    env.reset()
    before, total = env.cost_of(()), 0.0
    for action in (1, 2, 0, 3):  # the paper's order for these lines: buses 4, 6, 1, 13
        _, reward, done = env.step(action)
        after = (
            float(env.result_of(env.trusted).channels)
            if env.result_of(env.trusted).devices >= 0
            else env.closed_cost
        )
        assert reward == after - before
        before, total = after, total + reward
        if done:
            break
    assert total == env.cost - env.undefended


def test_a_schedule_is_searched_once(env):
    """Replaying an episode answers every schedule from the cache: no new search, the same rewards."""
    env.reset()
    first = [env.step(a)[1] for a in (1, 2)]
    solves = env.solves
    env.reset()
    assert [env.step(a)[1] for a in (1, 2)] == first and env.solves == solves


def test_the_state_is_what_fig_1_lists(env):
    """The node readings, the flows, the target lines' load rate, the trusted mask and the step index."""
    obs = env.reset()
    N, E, L, P = env.g.C, env.g.E, len(env.goal.lines), env.n_actions
    assert obs.shape == (N * 4 + E * 2 + L + P + 1,)
    assert np.all(obs[N * 4 + E * 2 : N * 4 + E * 2 + L] > 0)  # the lines carry flow
    assert not obs[-P - 1 : -1].any() and obs[-1] == 0
    obs, _, _ = env.step(2)
    assert obs[-P - 1 : -1].tolist() == [0, 0, 1, 0] and obs[-1] == 1 / len(SLOTS)
    assert env.valid().tolist() == [True, True, False, True]


def test_line_8_ends_the_episode_without_a_reward(env):
    """Algorithm 1's line 8: a defended cost below the undefended one ends the episode, no reward."""
    fake = MinimizerResult(np.array([1]), 1, 1, False, 1, 0, 0)
    env.cache[(3,)] = fake  # a schedule whose cost is below the undefended one
    env.reset()
    breaks = env.breaks
    _, reward, done = env.step(3)
    assert (reward, done, env.breaks, env.trusted) == (0.0, True, breaks + 1, ())
    del env.cache[(3,)]


def test_an_infeasible_attack_costs_everything_it_could_tamper(env):
    fake = MinimizerResult(np.array([1]), -1, -1, False, 1, 0, 0)
    env.cache[(0,)] = fake
    env.reset()
    _, reward, done = env.step(0)
    assert done and reward == env.closed_cost - env.undefended
    del env.cache[(0,)]


def test_the_search_minimizes_the_unit_the_reward_counts(env):
    """In channels (eq. 33's unit) the search puts the channel count first: its answer never tampers
    more channels than the device-first search's, which may trade channels for devices."""
    assert env.k.objective == "channels"
    channels_first = env.result_of(())
    devices_first = env.g.min_tamper(env.states, env.goal, env.k._replace(objective="devices"))
    assert channels_first.channels <= devices_first.channels
    assert devices_first.devices <= channels_first.devices


def test_a_step_outside_the_window_is_refused(env):
    with pytest.raises(ConfigError):
        WuDefenseEnv(
            env.g, env.states, env.goal, env.k, WuDefenseConfig(env.config.pmus, (1, len(env.states)))
        )


@pytest.mark.parametrize("action", [-1, 1.5, 4, 2])
def test_only_a_pmu_still_on_offer_is_an_action(env, action):
    """A negative index, a fraction, an index past the PMUs, and a PMU trusted already are refused
    before the schedule changes."""
    env.reset()
    env.step(2)
    with pytest.raises(ConfigError):
        env.step(action)
    assert env.trusted == (2,)


def test_the_infeasible_cost_counts_only_what_the_plan_meters(env):
    """The device ceiling holds the devices with a metered channel, not every bus's terminal."""
    from fdia_graph.formulas.attacks import tampered_devices
    from fdia_graph.models import WuDefenseConfig as Config

    devices = WuDefenseEnv(env.g, env.states, env.goal, env.k, Config(env.config.pmus, SLOTS, unit="devices"))
    w = devices.window
    ceiling = tampered_devices(w.node_m > 0, w.edge_m > 0, w.pmu, env.g.ei[0], w.i_m > 0, env.g.ei[1])
    assert devices.closed_cost == len(ceiling) <= env.g.C + len(env.g.meters.pmu)


def test_the_state_reads_only_metered_channels(env):
    obs = env.reset()
    N, E = env.g.C, env.g.E
    node = obs[: N * 4].reshape(N, 4)
    edge = obs[N * 4 : N * 4 + E * 2].reshape(E, 2)
    assert not node[env.window.node_m == 0].any() and not edge[env.window.edge_m == 0].any()
