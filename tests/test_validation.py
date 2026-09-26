"""The validation engine (models.validation): rules declared on fields, one message shape, and the
public entry points that build their config models before any work."""

from dataclasses import dataclass
from typing import Annotated, Optional

import numpy as np
import pytest

from fdia_graph.models.choices import Choice, Iso
from fdia_graph.models.config import FederatedSettings, LoadOptions, PriorConfig, SplitFractions, WindowSpec
from fdia_graph.models.validation import AtLeast, ConfigError, InRange, Integer, OneOf, Positive, Validated


class Mode(Choice):
    FAST = "fast"
    SLOW = "slow"


@dataclass(frozen=True)
class Knobs(Validated):
    rate: Annotated[float, Positive()] = 1.0
    count: Annotated[int, Integer(), AtLeast(1)] = 1
    share: Annotated[float, InRange(0, 1, hi_closed=True)] = 0.5
    mode: Annotated[str, OneOf(Mode)] = "fast"
    cap: Annotated[Optional[float], Positive()] = None

    def invariants(self):
        yield self.count <= 10 or self.mode == "slow", "more than 10 needs mode='slow'"


def test_rules_run_in_order_and_name_the_model_and_field():
    assert Knobs(rate=2.0, count=3).count == 3
    with pytest.raises(ConfigError, match=r"^Knobs\.rate must be > 0, got -1$"):
        Knobs(rate=-1)
    with pytest.raises(ConfigError, match=r"^Knobs\.count must be an integer, got 2\.5$"):
        Knobs(count=2.5)  # Integer runs before AtLeast
    with pytest.raises(ConfigError, match=r"^Knobs\.share must be in \(0, 1\], got 0$"):
        Knobs(share=0)
    assert issubclass(ConfigError, ValueError)  # every caller catching ValueError keeps working


def test_one_of_stores_the_canonical_string():
    k = Knobs(mode=Mode.SLOW)
    assert k.mode == "slow" and type(k.mode) is str
    with pytest.raises(ConfigError, match=r"^Knobs\.mode must be one of \['fast', 'slow'\], got 'medium'$"):
        Knobs(mode="medium")


def test_an_optional_field_skips_its_rules_when_none():
    assert Knobs(cap=None).cap is None
    with pytest.raises(ConfigError, match="Knobs.cap must be > 0"):
        Knobs(cap=0.0)


def test_invariants_run_after_the_fields():
    assert Knobs(count=20, mode="slow").count == 20
    with pytest.raises(ConfigError, match=r"^Knobs: more than 10 needs mode='slow'$"):
        Knobs(count=20)


def test_the_iso_choice_folds_case():
    assert Iso("NYISO") is Iso.NYISO


def test_the_config_models_state_the_old_rules():
    assert LoadOptions("train", "pu").units == "pu"
    with pytest.raises(ConfigError, match="LoadOptions.split must be one of"):
        LoadOptions(split="holdout")
    with pytest.raises(ConfigError, match=r"PriorConfig.rank_frac must be in \(0, 1\]"):
        PriorConfig(rank_frac=0.0)
    with pytest.raises(ConfigError, match="need train_frac \\+ val_frac < 1"):
        SplitFractions(0.7, 0.4)
    with pytest.raises(ConfigError, match="need integers 1 <= W <= 10"):
        WindowSpec(T=10, W=11)
    with pytest.raises(ConfigError, match="the partition has 3 clients but K=2"):
        FederatedSettings(K=2, partition_clients=3)
    with pytest.raises(ConfigError, match="pass those, not epochs"):
        FederatedSettings(epochs=5)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(units="kw"), "LoadOptions.units must be one of"),
        (dict(split="holdout"), "LoadOptions.split must be one of"),
        (dict(order="shuffled"), "LoadOptions.order must be one of"),
        (dict(format="pygg"), r"LoadOptions.format must be one of \['torch', 'pyg'\]"),
    ],
)
def test_load_rejects_a_bad_argument_before_any_download(kwargs, message, monkeypatch):
    import fdia_graph as fg

    def no_download(*a, **k):
        raise AssertionError("a bad argument reached the download")

    monkeypatch.setattr(fg, "ensure_local", no_download)  # the name load() calls
    with pytest.raises(ConfigError, match=message):
        fg.load("ieee14", **kwargs)


def test_the_old_check_functions_warn_and_still_validate():
    from fdia_graph.dataset import check_order, check_split, check_units

    with pytest.warns(DeprecationWarning, match="LoadOptions checks"):
        check_units("pu")
    with pytest.warns(DeprecationWarning), pytest.raises(ConfigError, match="split must be one of"):
        check_split("holdout")
    with pytest.warns(DeprecationWarning):
        check_order("time")
        check_split(None)


def test_estimators_and_localizers_build_their_models():
    from fdia_graph.localization import SwingThreshold
    from fdia_graph.se import AdaptiveWeighting, GatedPrior, SubspacePrior

    with pytest.raises(ConfigError, match="HuberConfig.c must be > 0"):
        AdaptiveWeighting(c=0)
    with pytest.raises(ConfigError, match="PriorConfig.reweight must be one of"):
        SubspacePrior(reweight="tukey")
    with pytest.raises(ConfigError, match="LocalizerConfig.fa_target must be in"):
        SwingThreshold(fa_target=1.0)
    with pytest.raises(ConfigError, match="pass gate=<fitted localizer> or gate='oracle'"):
        GatedPrior(gate="orcale")
    assert SubspacePrior(reweight="huber").reweight == "huber"


def test_the_formula_options_share_the_message():
    from fdia_graph.formulas.projection import meters_to_buses
    from fdia_graph.localization.learned import FEATURE_SETS
    from fdia_graph.models.choices import Features

    with pytest.raises(ValueError, match="reduce must be one of"):
        meters_to_buses(np.zeros((1, 2)), [np.array([0])], "mean")
    assert FEATURE_SETS["full14+prev"] == 16 and set(FEATURE_SETS) == set(Features)


def test_a_view_is_refused_through_the_capability_table(timeline):
    import fdia_graph as fg
    from fdia_graph.dataset.base import CAPABILITIES
    from fdia_graph.errors import MissingCapability
    from fdia_graph.models.choices import Capability

    assert set(CAPABILITIES) == set(Capability)  # every capability has its test and its phrase
    pu = fg.load(timeline, split="test", units="pu")
    pu.require("timeline", "benign_layer", by="anything")  # what a timeline offers passes
    with pytest.raises(MissingCapability, match="^the estimator needs units='physical'"):
        pu.require("physical_units", by="the estimator")
    with pytest.raises(MissingCapability, match="needs order='time'"):
        fg.load(timeline, split="test", order="random").windows(4)


def test_data_conditions_raise_named_errors(timeline):
    import fdia_graph as fg
    from fdia_graph.errors import DataConditionError, NoBenignRecords
    from fdia_graph.localization import SwingThreshold

    attacked_only = fg.load(timeline, split="train", families=["Ad"])
    with pytest.raises(NoBenignRecords, match="fit needs benign records"):
        SwingThreshold().fit(attacked_only)
    assert issubclass(NoBenignRecords, DataConditionError) and issubclass(DataConditionError, ValueError)
