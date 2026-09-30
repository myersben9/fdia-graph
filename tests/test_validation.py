"""The validation engine (models.validation): rules declared on fields, one message shape, and the
public entry points that build their config models before any work."""

from dataclasses import dataclass
from typing import Annotated, Optional

import numpy as np
import pytest

from fdia_graph.models.choices import Choice, Iso
from fdia_graph.models.config import FederatedSettings, LoadOptions, PriorConfig, WindowSpec
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
    with pytest.raises(ConfigError, match="need integers 1 <= W <= 10"):
        WindowSpec(T=10, W=11)
    with pytest.raises(ConfigError, match="the partition has 3 clients but K=2"):
        FederatedSettings(K=2, partition_clients=3)
    with pytest.raises(ConfigError, match="pass those, not epochs"):
        FederatedSettings(epochs=5)


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: WindowSpec(T=10.5, W=2), "WindowSpec.T must be an integer"),
        (
            lambda: FederatedSettings(K=2, partition_clients=2.0),
            "FederatedSettings.partition_clients must be an integer",
        ),
        (lambda: FederatedSettings(epochs=1.5), "FederatedSettings.epochs must be an integer"),
    ],
)
def test_whole_number_settings_refuse_a_fraction(build, message):
    with pytest.raises(ConfigError, match=message):
        build()


def test_a_negative_bus_count_is_refused_even_without_branches():
    from fdia_graph.models.inputs import EdgeList

    with pytest.raises(ConfigError, match=r"EdgeList\.N must be >= 0"):
        EdgeList(np.zeros((2, 0), dtype=np.int64), -1)


@pytest.mark.parametrize(
    "assignment, edge_index, message",
    [
        (3, np.array([[0, 1], [1, 2]]), "assignment must be one integer client per bus"),
        ([0, 0, 1], np.array([[0.0, 1.0], [1.0, 2.0]]), "edge_index must be a non-negative integer"),
        ([0, 0, 1], np.array([["a", "b"], ["b", "c"]]), "edge_index must be a non-negative integer"),
    ],
)
def test_a_malformed_assignment_is_refused_before_the_grid_is_sized(assignment, edge_index, message):
    from fdia_graph.federated.partition import partition_from_assignment

    with pytest.raises(ConfigError, match=message):
        partition_from_assignment(assignment, edge_index)


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

    with pytest.raises(
        ConfigError, match=r"^Aggregation\.reduce must be one of \['sum', 'max'\], got 'mean'$"
    ):
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


def test_the_gate_refuses_a_type_dispatch_outside_the_models(tmp_path, monkeypatch):
    """Loose input is parsed by a model (models.inputs), so `isinstance` lives in the models only."""
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import readability

    (tmp_path / "models").mkdir()
    inside = tmp_path / "models" / "inputs.py"
    outside = tmp_path / "loader.py"
    for f in (inside, outside):
        f.write_text("def f(x):\n    return isinstance(x, str)\n")
    monkeypatch.setattr(readability, "_MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(readability, "ROOT", str(tmp_path))
    assert readability.hand_checks(str(inside)) == []
    assert [h[2] for h in readability.hand_checks(str(outside))] == ["isinstance"]


def test_the_gate_must_be_oracle_or_a_callable_localizer():
    from types import SimpleNamespace

    from fdia_graph.models.config import GateConfig

    assert (
        GateConfig("oracle").is_oracle and not GateConfig(SimpleNamespace(localize=lambda ds: None)).is_oracle
    )
    for bad in (SimpleNamespace(localize=None), np.zeros(3), "orcale"):
        with pytest.raises(ConfigError, match="pass gate=<fitted localizer> or gate='oracle'"):
            GateConfig(bad)


def test_parsers_are_models_that_keep_the_input_and_give_the_result():
    from fdia_graph.models.inputs import FamilySelection, ReleaseName, SystemRef

    assert SystemRef("IEEE118").number == 118 and ReleaseName("data-v0.8.3").numbers == (0, 8, 3)
    assert FamilySelection(["Ao", 2, "Am"]).codes == (1, 2, 7)
    with pytest.raises(ConfigError, match="unknown family"):
        FamilySelection(["Aq", "Zz"])
    with pytest.raises(ConfigError, match="SystemRef.system must be like 'ieee118' or 118"):
        SystemRef("big")


def test_a_required_field_refuses_none_with_its_phrase():
    from fdia_graph.models.inputs import CsvSpec

    with pytest.raises(ConfigError, match=r"^CsvSpec\.column is required$"):
        CsvSpec("x.csv", None)


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


def test_an_overflowing_value_fails_its_rule():
    from fdia_graph.models.config import FederatedSettings, TrainerConfig

    with pytest.raises(ConfigError, match=r"^TrainerConfig\.clip must be finite"):
        TrainerConfig(clip=10**1000)
    with pytest.raises(ConfigError, match=r"^FederatedSettings\.grad_clip must be finite"):
        FederatedSettings(grad_clip=10**1000)


def test_a_capability_name_and_a_fetch_are_checked_by_their_models(timeline):
    import fdia_graph as fg
    from fdia_graph.profiles import fetch_profile

    with pytest.raises(ConfigError, match=r"^Requirement\.capabilities must name capabilities"):
        fg.load(timeline, split="test").require("timelime", by="a typo")
    with pytest.raises(ConfigError, match=r"^ProfileFetch\.iso must be one of"):
        fetch_profile("pjm", "2024-01-01", "2024-01-02")
    with pytest.raises(ConfigError, match=r"^ProfileFetch\.resample_min must be >= 1"):
        fetch_profile("nyiso", "2024-01-01", "2024-01-02", resample_min=0)
    with pytest.raises(ConfigError, match=r"^ProfileFetch\.resample_min must be an integer"):
        fetch_profile("nyiso", "2024-01-01", "2024-01-02", resample_min=1.5)  # was truncated to 1
