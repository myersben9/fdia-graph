"""Which meters to trust so that stealthy attacks stop being stealthy [WU26], sklearn style.

    from fdia_graph.trust import TrustedMeters, TrustedMetersDQN

    train = fg.load("ieee118", split="train")
    test = fg.load("ieee118", split="test")
    tm = TrustedMeters(k=20).fit(train)   # the greedy row-reduction selection on the Jacobian
    tm.order, tm.cost                     # the meters secured in order, the attack cost after each
    print(tm.score(test))                 # per family: residual detection with and without the trust

    rl = TrustedMetersDQN(k=20, episodes=200).fit(train)   # the MDP with a DQN policy (needs torch)

A stealthy attack must leave the secured meters untouched, which pins its state change to the
null space of the secured rows of the measurement Jacobian; every secured meter shrinks that
space, and the selection is which meter to secure next. Both classes share the attack-cost
kernel (`formulas.trust`) and the scoring: the secured meters are set back to their un-attacked
reading on every attacked record and the residual test of a WLS estimator says whether the attack
now shows. Needs the [se] extra; the DQN the [torch] extra.

`WuDefenseEnv` is [WU26]'s own MDP on the overload attack: the agent trusts one PMU per configuration
step, and the reward is the rise in the fewest-tamper search's cost under the schedule (eq. 33).
"""

from .base import TrustedMeters, TrustSelector
from .defense import WuDefenseEnv
from .dqn import TrustedMetersDQN
from .secured import secured_copy

__all__ = ["TrustSelector", "TrustedMeters", "TrustedMetersDQN", "WuDefenseEnv", "secured_copy"]
