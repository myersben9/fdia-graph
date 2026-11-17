"""The analysis toolkit's part of tests/test_validation.py:
the tests that run an estimator, a localizer, a trust selector or an analysis formula. Of that module:
The validation engine (models.validation): rules declared on fields, one message shape, and the
public entry points that build their config models before any work."""

import numpy as np
import pytest

from fdia_analysis.models import GateConfig, HuberConfig, LocalizerConfig
from fdia_analysis.models.config import FederatedSettings, PriorConfig
from fdia_graph.models.config import LoadOptions, WindowSpec
from fdia_graph.models.validation import (
    ConfigError,
)


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


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: WindowSpec(T=10.5, W=2), "WindowSpec.T must be an integer"),
        (
            lambda: FederatedSettings(K=2, partition_clients=2.0),
            "FederatedSettings.partition_clients must be an integer",
        ),
    ],
)
def test_whole_number_settings_refuse_a_fraction(build, message):
    with pytest.raises(ConfigError, match=message):
        build()


def test_a_negative_bus_count_is_refused_even_without_branches():
    from fdia_analysis.models.inputs import EdgeList

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
    from fdia_analysis.federated.partition import partition_from_assignment

    with pytest.raises(ConfigError, match=message):
        partition_from_assignment(assignment, edge_index)


def test_estimators_and_localizers_build_their_models():
    from fdia_analysis.localization import SwingThreshold
    from fdia_analysis.se import AdaptiveWeighting, GatedPrior, SubspacePrior

    with pytest.raises(ConfigError, match="HuberConfig.c must be > 0"):
        AdaptiveWeighting(huber=HuberConfig(c=0))
    with pytest.raises(ConfigError, match="PriorConfig.reweight must be one of"):
        SubspacePrior(prior=PriorConfig(reweight="tukey"))
    with pytest.raises(ConfigError, match="LocalizerConfig.fa_target must be in"):
        SwingThreshold(localizer=LocalizerConfig(fa_target=1.0))
    with pytest.raises(ConfigError, match="pass gate=<fitted localizer> or gate='oracle'"):
        GatedPrior(GateConfig(gate="orcale"))
    assert SubspacePrior(prior=PriorConfig(reweight="huber")).reweight == "huber"


def test_the_formula_options_share_the_message():
    from fdia_analysis.formulas.projection import meters_to_buses
    from fdia_analysis.localization.learned import FEATURE_SETS
    from fdia_analysis.models.choices import Features

    with pytest.raises(
        ConfigError, match=r"^Aggregation\.reduce must be one of \['sum', 'max'\], got 'mean'$"
    ):
        meters_to_buses(np.zeros((1, 2)), [np.array([0])], "mean")
    assert FEATURE_SETS["full14+prev"] == 16 and set(FEATURE_SETS) == set(Features)


def test_data_conditions_raise_named_errors(timeline):
    import fdia_graph as fg
    from fdia_analysis.localization import SwingThreshold
    from fdia_graph.errors import DataConditionError, NoBenignRecords

    attacked_only = fg.load(timeline, split="train", families=["Ad"])
    with pytest.raises(NoBenignRecords, match="fit needs benign records"):
        SwingThreshold().fit(attacked_only)
    assert issubclass(NoBenignRecords, DataConditionError) and issubclass(DataConditionError, ValueError)


def test_a_malformed_value_gets_the_engine_message_not_a_raw_type_error():
    from fdia_analysis.models.config import HuberConfig, SolveConfig, TrustConfig

    with pytest.raises(ConfigError, match=r"^HuberConfig\.c must be > 0, got 'bad'$"):
        HuberConfig(c="bad")
    for bad in (dict(npass=1.5), dict(iters=2.0)):
        with pytest.raises(ConfigError, match="must be an integer"):
            SolveConfig(**bad)
    with pytest.raises(ConfigError, match="TrustConfig.k must be an integer"):
        TrustConfig(k=1.5)
    assert SolveConfig(npass=np.int64(3)).npass == 3  # numpy integers are integers


def test_the_gate_must_be_oracle_or_a_callable_localizer():
    from types import SimpleNamespace

    from fdia_analysis.models.config import GateConfig

    assert (
        GateConfig("oracle").is_oracle and not GateConfig(SimpleNamespace(localize=lambda ds: None)).is_oracle
    )
    for bad in (SimpleNamespace(localize=None), np.zeros(3), "orcale"):
        with pytest.raises(ConfigError, match="pass gate=<fitted localizer> or gate='oracle'"):
            GateConfig(bad)


def test_formulas_build_their_input_models():
    from fdia_analysis.formulas.federated import fedavg, pool_moments
    from fdia_analysis.formulas.metrics import perbus_counts

    with pytest.raises(ConfigError, match="ClientUpdates: client weights must be finite and positive"):
        fedavg([np.zeros(2), np.zeros(2)], [1.0, -1.0])
    with pytest.raises(ConfigError, match=r"LabelGrids: need two \[n, N\] boolean arrays"):
        perbus_counts(np.zeros((2, 3)), np.zeros((3, 2)))
    with pytest.raises(ConfigError, match="pool_moments needs at least one part"):
        pool_moments([])


def test_an_untyped_field_is_still_required_and_whole_numbers_are_integers():
    from fdia_analysis.formulas.federated import block_diagonal_basis
    from fdia_analysis.models.inputs import EdgeList
    from fdia_graph.models.inputs import StatePool

    with pytest.raises(ConfigError, match=r"^StatePool\.X is required$"):
        StatePool(None)
    with pytest.raises(ConfigError, match=r"^EdgeList\.N must be an integer, got 3\.5$"):
        EdgeList(np.array([[0], [1]]), 3.5)
    with pytest.raises(ConfigError, match=r"StateBlocks\.d must be an integer"):
        block_diagonal_basis([(np.array([0]), np.eye(1))], 3.5)


def test_an_overflowing_value_fails_its_rule():
    from fdia_analysis.models.config import FederatedSettings, TrainerConfig

    with pytest.raises(ConfigError, match=r"^TrainerConfig\.clip must be finite"):
        TrainerConfig(clip=10**1000)
    with pytest.raises(ConfigError, match=r"^FederatedSettings\.grad_clip must be finite"):
        FederatedSettings(grad_clip=10**1000)
