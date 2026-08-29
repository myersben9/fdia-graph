"""State estimation on fdia-graph shards, sklearn style.

    from fdia_graph.se import WLS, AdaptiveWeighting, ResidualRemoval, SubspacePrior

    train = fg.load("ieee118", split="train")
    test = fg.load("ieee118", split="test")
    est = SubspacePrior(rank_frac=0.5, reweight="huber", c=2.5).fit(train)
    xhat = est.estimate(test)      # [n, 2(N-1)] = [theta rad | V pu] at non-slack buses
    print(est.score(test))         # per-family angle/voltage MAE vs the clean truth

Every class shares one chord-Newton iteration, Jacobian and starting point and differs only in the
state space and the weights, so results compare estimators rather than implementations. Needs the
se extra (torch + pandapower) and v0.7.2+ shards.
"""

from .base import SEBase
from .methods import WLS, AdaptiveWeighting, ResidualRemoval, SubspacePrior

__all__ = ["SEBase", "WLS", "AdaptiveWeighting", "ResidualRemoval", "SubspacePrior"]
