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


def test_a_malformed_value_gets_the_engine_message_not_a_raw_type_error():
    from fdia_graph.models.config import HuberConfig, SolveConfig, TrustConfig

    with pytest.raises(ConfigError, match=r"^HuberConfig\.c must be > 0, got 'bad'$"):
        HuberConfig(c="bad")
    for bad in (dict(npass=1.5), dict(iters=2.0)):
        with pytest.raises(ConfigError, match="must be an integer"):
            SolveConfig(**bad)
    with pytest.raises(ConfigError, match="TrustConfig.k must be an integer"):
        TrustConfig(k=1.5)
    assert SolveConfig(npass=np.int64(3)).npass == 3  # numpy integers are integers


def test_the_gate_exempts_the_models_package_only(tmp_path, monkeypatch):
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import readability

    sibling = tmp_path / "models_extra.py"
    sibling.write_text("def f():\n    raise ValueError('x')\n")
    monkeypatch.setattr(readability, "_MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(readability, "ROOT", str(tmp_path))
    assert [h[2] for h in readability.hand_checks(str(sibling))] == ["ValueError"]


def test_the_gate_must_be_oracle_or_a_callable_localizer():
    from types import SimpleNamespace

    from fdia_graph.models.config import GateConfig

    assert (
        GateConfig("oracle").is_oracle and not GateConfig(SimpleNamespace(localize=lambda ds: None)).is_oracle
    )
    for bad in (SimpleNamespace(localize=None), np.zeros(3), "orcale"):
        with pytest.raises(ConfigError, match="pass gate=<fitted localizer> or gate='oracle'"):
            GateConfig(bad)


def test_an_outage_that_is_neither_a_name_nor_an_integer_is_refused():
    from fdia_graph.engine.core import _line_id

    net = SimpleNamespaceNet()
    for bad in (1.5, None):
        with pytest.raises(ConfigError, match="an outage is a line name or an integer line index"):
            _line_id(net, bad)
    assert _line_id(net, np.int64(2)) == 2


class SimpleNamespaceNet:
    """The part of a pandapower net `_line_id` reads: lines 0..3."""

    def __init__(self):
        import pandas as pd

        self.line = pd.DataFrame({"name": ["a", "b", "c", "d"]})


def test_parsers_are_models_that_keep_the_input_and_give_the_result():
    from fdia_graph.models.inputs import FamilySelection, OutageRef, ReleaseName, SystemRef

    assert SystemRef("IEEE118").number == 118 and ReleaseName("data-v0.8.3").numbers == (0, 8, 3)
    assert FamilySelection(["Ao", 2, "Am"]).codes == (1, 2, 7)
    with pytest.raises(ConfigError, match="unknown family"):
        FamilySelection(["Aq", "Zz"])
    with pytest.raises(ConfigError, match="SystemRef.system must be like 'ieee118' or 118"):
        SystemRef("big")
    assert OutageRef("b", ("a", "b"), (4, 7)).index == 7
    assert OutageRef(np.int64(4), ("a", "b"), (4, 7)).index == 4
    with pytest.raises(ConfigError, match="is ambiguous"):
        OutageRef("a", ("a", "a"), (4, 7))


def test_a_required_field_refuses_none_with_its_phrase():
    from fdia_graph.models.inputs import CsvSpec, StreamSystem

    with pytest.raises(ConfigError, match=r"^CsvSpec\.column is required$"):
        CsvSpec("x.csv", None)
    with pytest.raises(ConfigError, match="pass dataset=<fg.load"):
        StreamSystem(None)
    assert StreamSystem("ieee14").number == 14


def test_formulas_build_their_input_models():
    from fdia_graph.formulas.federated import fedavg, pool_moments
    from fdia_graph.formulas.metrics import perbus_counts

    with pytest.raises(ConfigError, match="ClientUpdates: client weights must be finite and positive"):
        fedavg([np.zeros(2), np.zeros(2)], [1.0, -1.0])
    with pytest.raises(ConfigError, match=r"LabelGrids: need two \[n, N\] boolean arrays"):
        perbus_counts(np.zeros((2, 3)), np.zeros((3, 2)))
    with pytest.raises(ConfigError, match="pool_moments needs at least one part"):
        pool_moments([])


def test_an_untyped_field_is_still_required_and_whole_numbers_are_integers():
    from fdia_graph.formulas.federated import block_diagonal_basis
    from fdia_graph.models.inputs import EdgeList, StatePool

    with pytest.raises(ConfigError, match=r"^StatePool\.X is required$"):
        StatePool(None)
    with pytest.raises(ConfigError, match=r"^EdgeList\.N must be an integer, got 3\.5$"):
        EdgeList(np.array([[0], [1]]), 3.5)
    with pytest.raises(ConfigError, match=r"StateBlocks\.d must be an integer"):
        block_diagonal_basis([(np.array([0]), np.eye(1))], 3.5)
