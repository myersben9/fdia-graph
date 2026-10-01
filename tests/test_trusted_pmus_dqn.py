"""[WU26]'s Solution 2 (`trust.TrustedPMUsDQN`, Algorithm 1; docs/plans/WU_DEFENSE_PLAN.md PR D): the
stated hyperparameters, a constructed MDP where the right first PMU is known, seeded determinism, and one
short run on the paper's IEEE-14 scenario."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from fdia_graph.models import WuDqnConfig  # noqa: E402
from fdia_graph.models.validation import ConfigError  # noqa: E402
from fdia_graph.trust import TrustedPMUsDQN  # noqa: E402


class _Chain:
    """Three PMUs, two steps: trusting PMU 2 first earns 5, anything else 0, and the second step earns 1.
    The state is the trusted mask and the step index."""

    n_actions, breaks = 3, 0
    config = SimpleNamespace(slots=(0, 1))

    def reset(self):
        self.trusted = []
        return self._obs()

    def valid(self):
        mask = np.ones(3, bool)
        mask[self.trusted] = False
        return mask if len(self.trusted) < 2 else np.zeros(3, bool)

    def step(self, action):
        reward = (5.0 if action == 2 else 0.0) if not self.trusted else 1.0
        self.trusted.append(action)
        return self._obs(), reward, len(self.trusted) == 2

    def _obs(self):
        mask = np.zeros(3)
        mask[self.trusted] = 1.0
        return np.r_[mask, len(self.trusted) + 1.0]


def test_the_stated_hyperparameters_are_the_defaults():
    c = WuDqnConfig()
    assert (c.gamma, c.lr, c.buffer, c.batch, c.target_every, c.epsilon_decay, c.episodes) == (
        0.9,
        0.005,
        2500,
        25,
        20,
        0.002,
        250,
    )
    with pytest.raises(ConfigError):
        WuDqnConfig(gamma=1.5)


def test_algorithm_1_learns_the_pmu_that_pays():
    dqn = TrustedPMUsDQN([_Chain()]).fit()
    assert dqn.order(_Chain())[0] == 2
    assert len(dqn.history) == 250 and dqn.breaks == 0


def test_a_seeded_run_repeats():
    a = TrustedPMUsDQN([_Chain()], WuDqnConfig(episodes=40)).fit()
    b = TrustedPMUsDQN([_Chain()], WuDqnConfig(episodes=40)).fit()
    assert a.history == b.history


def test_a_short_run_on_the_ieee14_scenario():
    """Three iterations on the paper's lines 1-2 and 4-5 (8 pool frames, k = 1.1): the policy's order is
    a sequence of distinct PMUs of the four."""
    pytest.importorskip("pandapower")
    import test_trusted_pmus as tt

    from fdia_graph.models import WuDefenseConfig
    from fdia_graph.trust import WuDefenseEnv

    tt.WINDOW, window = 8, tt.WINDOW
    try:
        g, states, k, goal = tt._setup(1, 1.1)
    finally:
        tt.WINDOW = window
    pmus = [int(b) for b in g.wu26_buses([1, 4, 6, 13])]
    env = WuDefenseEnv(g, states, goal, k._replace(min_budget=64), WuDefenseConfig(pmus, (1, 3, 5, 7)))
    dqn = TrustedPMUsDQN([env], WuDqnConfig(episodes=3)).fit()
    order = dqn.order(env)
    assert len(order) == len(set(order)) and set(order) <= {0, 1, 2, 3}


def test_the_environments_must_share_their_pmus_and_steps():
    other = _Chain()
    other.n_actions = 4
    with pytest.raises(ConfigError):
        TrustedPMUsDQN([_Chain(), other])
    with pytest.raises(ConfigError):
        TrustedPMUsDQN([])


def test_a_minibatch_larger_than_the_buffer_is_refused():
    """Training samples a minibatch only once the buffer holds one, so a batch over the buffer would
    leave the network untrained."""
    with pytest.raises(ConfigError):
        WuDqnConfig(buffer=10, batch=25)
