"""Per-bus FDIA localization on fdia-graph shards, sklearn style.

    from fdia_graph.localization import SwingThreshold, DeltaThreshold, ResidualLocalizer

    train = fg.load("ieee118", split="train")
    test = fg.load("ieee118", split="test")
    loc = SwingThreshold(fa_target=0.01).fit(train)
    flag = loc.localize(test)      # [n, N] bool, which buses are called attacked
    print(loc.score(test))         # per-family strict acc / node F1 / DR + benign false alarms

Every class shares one calibration protocol (per-bus thresholds set on benign training records at
the same false-alarm budget) and one metrics suite, and differs only in the per-bus score, so
results compare detection signals rather than implementations. The threshold methods are
numpy-only; ResidualLocalizer composes an estimator from fdia_graph.se and needs the [se] extra.
"""

from .base import LocalizerBase
from .methods import DeltaThreshold, ResidualLocalizer, SwingThreshold

__all__ = ["LocalizerBase", "SwingThreshold", "DeltaThreshold", "ResidualLocalizer"]
