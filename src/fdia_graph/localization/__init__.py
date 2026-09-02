"""Per-bus FDIA localization on fdia-graph shards, sklearn style.

    from fdia_graph.localization import SwingThreshold, DeltaThreshold, ResidualLocalizer, BusCNN

    train = fg.load("ieee118", split="train")
    test = fg.load("ieee118", split="test")
    loc = SwingThreshold(fa_target=0.01).fit(train)
    flag = loc.localize(test)      # [n, N] bool, which buses are called attacked
    print(loc.score(test))         # per-family strict acc / node F1 / macro F1 / DR + benign FA

    # the papers' best learned localizer, in their zero-shot protocol (needs torch)
    zs = dict(families=[0, 1, 2])
    cnn = BusCNN().fit(fg.load("ieee118", split="train", **zs), val=fg.load("ieee118", split="val", **zs))
    print(cnn.score(fg.load("ieee118", split="test", families=[0, 1, 2, 3, 4]))["all"]["macro_f1"])

Every class shares one calibration protocol (per-bus thresholds set on benign training records at
the same false-alarm budget) and one metrics suite, and differs only in the per-bus score, so
results compare detection signals rather than implementations. The threshold methods are
numpy-only; ResidualLocalizer composes an estimator from fdia_graph.se and needs the [se] extra;
BusCNN and BusMLP train on the records they are given and need the [torch] extra.
"""

from .base import LocalizerBase
from .learned import BusCNN, BusMLP, LearnedLocalizer, full14, kcl_residual
from .methods import DeltaThreshold, ResidualLocalizer, SwingThreshold

__all__ = [
    "LocalizerBase",
    "SwingThreshold",
    "DeltaThreshold",
    "ResidualLocalizer",
    "LearnedLocalizer",
    "BusCNN",
    "BusMLP",
    "full14",
    "kcl_residual",
]
