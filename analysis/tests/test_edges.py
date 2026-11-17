"""The analysis toolkit's part of tests/test_edges.py:
the tests that run an estimator, a localizer, a trust selector or an analysis formula. Of that module:
The public edges: wrong arguments fail early with a message naming what is allowed, and the
filters mean what the docs say. The other test files prove the happy path; this file is the rest."""

import os

import numpy as np
import pytest

import fdia_graph as fg
from fdia_analysis.models import (
    GateConfig,
    HuberConfig,
    JacobianWeightingConfig,
    LocalizerConfig,
    PriorConfig,
    RemovalConfig,
    SolveConfig,
)

# ---- estimators and localizers ---------------------------------------------------------------------------


def test_estimator_constructor_checks():
    from fdia_analysis.se import (
        WLS,
        AdaptiveWeighting,
        GatedPrior,
        JacobianWeighting,
        ResidualRemoval,
        SubspacePrior,
    )

    with pytest.raises(ValueError, match="SolveConfig.npass must be >= 1"):
        WLS(solve=SolveConfig(npass=0))
    with pytest.raises(ValueError, match="c must be"):
        AdaptiveWeighting(huber=HuberConfig(c=0.0))
    with pytest.raises(ValueError, match="threshold"):
        ResidualRemoval(removal=RemovalConfig(threshold=0.0))
    with pytest.raises(ValueError, match="rank_frac"):
        SubspacePrior(prior=PriorConfig(rank_frac=1.5))
    with pytest.raises(ValueError, match="reweight"):
        SubspacePrior(prior=PriorConfig(reweight="bogus"))
    with pytest.raises(ValueError, match="reweight"):
        JacobianWeighting(weighting=JacobianWeightingConfig(reweight="bogus"))
    with pytest.raises(ValueError, match="gate"):
        GatedPrior(GateConfig())
    with pytest.raises(ValueError, match="gate_factor"):
        GatedPrior(GateConfig(gate="oracle", gate_factor=2.0))


def test_estimator_fit_and_score_input_checks(timeline, splits):
    from fdia_analysis.se import WLS

    with pytest.raises(ValueError, match="units='physical'"):
        WLS().fit(fg.load(timeline, split="train", units="pu"))
    with pytest.raises(ValueError, match="benign records"):
        WLS().fit(fg.load(timeline, split="train", families=["At"]))
    est = WLS().fit(splits["train"])
    with pytest.raises(ValueError, match="xhat must be"):
        est.score(splits["test"], xhat=np.zeros((3, est.SD)))


def test_localizer_input_checks(timeline, splits):
    from fdia_analysis.localization import SwingThreshold

    with pytest.raises(ValueError, match="fa_target"):
        SwingThreshold(localizer=LocalizerConfig(fa_target=1.5))
    with pytest.raises(ValueError, match="benign records"):
        SwingThreshold().fit(fg.load(timeline, split="train", families=["At"]))
    loc = SwingThreshold().fit(splits["train"])
    with pytest.raises(ValueError, match="scores must be"):
        loc.score(splits["test"], scores=np.zeros((2, 2)))


# ---- opt-in: the real IEEE-118 timeline --------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("FDIA_SLOW"), reason="set FDIA_SLOW=1: downloads the 317 MB IEEE-118 timeline"
)
def test_ieee118_estimator_sanity():
    """WLS on the published IEEE-118 timeline: the benign angle error is a fraction of a degree and
    every class scores finite. A loose bound, so a data or solver regression shows without pinning
    the paper's numbers here (docs/analysis/estimation/README.md holds those)."""
    from fdia_analysis.se import WLS

    train, test = fg.load("ieee118", split="train"), fg.load("ieee118", split="test")
    s = WLS().fit(train).score(test)
    assert s.benign is not None and 0 < s.benign.angle_mae_deg < 0.5
    assert all(np.isfinite(v.angle_mae_deg) and np.isfinite(v.voltage_mae_pu) for v in s.values())
