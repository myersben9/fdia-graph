"""A malformed setting or argument of the analysis toolkit is a `ConfigError`, never a raw TypeError,
KeyError, ValueError or IndexError from deeper in the code (docs/reference/REVIEW_CHECKLIST.md, item 4).

The same two sweeps as the generation toolkit's tests/test_malformed_inputs.py, through the helpers both share (tests/helpers.py):
every settings, input and results model of `fdia_analysis.models` is built with each of its fields
replaced by a value of the wrong kind, and each entry point that takes settings is called with one
bad argument.
"""

import importlib

import numpy as np
import pytest

import fdia_analysis
from fdia_analysis.models import config, inputs
from fdia_analysis.models import results as result_models
from fdia_graph.errors import ConfigError
from fdia_graph.models import config as generation_config
from fdia_graph.models import inputs as generation_inputs
from tests.helpers import check_functions, converted_arguments, models_of, sweep

_EDGES = np.array([[0, 1], [1, 2]])
_ADJ = np.eye(3)
_GRAPH = inputs.ClientGraph(np.array([0, 0, 1]), _ADJ)
_PROVENANCE = dict(run_id="demo.x-1", experiment="demo.x", timestamp="2026-01-01T00:00:00Z", sdk_version="0")

# a valid value for every required field, so each bad value is the only thing wrong
VALID = {
    config.TrustConfig: dict(k=2),
    config.GateConfig: dict(gate="oracle"),
    inputs.ShapedArray: dict(values=np.zeros((2, 3)), shape=(2, 3)),
    inputs.EdgeList: dict(edge_index=_EDGES, N=3),
    inputs.ClientCount: dict(N=3, K=2),
    inputs.ClientGraph: dict(assignment=np.array([0, 0, 1]), adjacency=_ADJ),
    inputs.Halo: dict(graph=_GRAPH, k=0, depth=1),
    inputs.AssignmentSpec: dict(assignment=np.array([0, 0, 1]), edge_index=_EDGES),
    inputs.PartitionOnGrid: dict(assignment=np.array([0, 0, 1]), K=2, N=3),
    inputs.ClientUpdates: dict(tensors=(np.zeros(2), np.ones(2)), weights=np.array([1.0, 2.0])),
    inputs.MomentParts: dict(parts=((3, np.zeros(2), np.ones(2)),)),
    inputs.FeatureBlock: dict(X=np.zeros((2, 3, 4))),
    inputs.Affinity: dict(adjacency=_ADJ, attackable=np.array([True, False, True])),
    inputs.StateBlocks: dict(blocks=((np.array([0, 1]), np.eye(2)),), d=3),
    inputs.LabelGrids: dict(pred=np.zeros((2, 3), bool), truth=np.zeros((2, 3), bool)),
    inputs.TauSearch: dict(
        counts=(np.zeros((2, 3)),) * 3, active=np.array([True, False, True]), taus=np.array([0.1, 0.2])
    ),
    inputs.RankedLabels: dict(score=np.array([0.1, 0.9]), truth=np.array([False, True])),
    inputs.Aggregation: dict(reduce="sum"),
    result_models.Record: dict(experiment="demo.x", metric="angle_mae_deg", value=0.1),
    result_models.Provenance: _PROVENANCE,
    result_models.RunRecords: dict(provenance=result_models.Provenance(**_PROVENANCE), records=()),
}

MODELS = models_of(config, inputs, result_models)


def test_every_model_has_a_valid_example():
    for cls in MODELS:
        cls(**VALID.get(cls, {}))  # the base every sweep case starts from builds cleanly


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_a_wrong_kind_of_value_is_accepted_or_a_config_error(cls):
    sweep(cls, VALID.get(cls, {}))


# ---- the public entry points --------------------------------------------------------------------
def _build(module, name, **kw):
    return lambda: getattr(importlib.import_module(f"fdia_analysis.{module}"), name)(**kw)


ENTRY_POINTS = [
    ("SolveConfig npass", _build("models.config", "SolveConfig", npass="many")),
    ("SolveConfig iters", _build("models.config", "SolveConfig", iters=2.5)),
    ("HuberConfig c", _build("models.config", "HuberConfig", c=float("nan"))),
    ("HuberConfig tol", _build("models.config", "HuberConfig", tol="small")),
    ("RemovalConfig threshold", _build("models.config", "RemovalConfig", threshold="high")),
    ("RemovalConfig cond_mult", _build("models.config", "RemovalConfig", cond_mult=-1)),
    ("PriorConfig rank_frac", _build("models.config", "PriorConfig", rank_frac=10**1000)),
    ("PriorConfig reweight", _build("models.config", "PriorConfig", reweight="tukey")),
    (
        "JacobianWeightingConfig huber_c",
        _build("models.config", "JacobianWeightingConfig", huber_c=float("inf")),
    ),
    ("GateConfig gate_factor", _build("models.config", "GateConfig", gate="oracle", gate_factor="big")),
    ("LocalizerConfig fa_target", _build("models.config", "LocalizerConfig", fa_target=1.5)),
    ("LocalizerConfig fa_target kind", _build("models.config", "LocalizerConfig", fa_target="low")),
    ("LocalizerConfig fa_target", _build("models.config", "LocalizerConfig", fa_target=None)),
    ("TrustConfig k", _build("models.config", "TrustConfig", k=2.5)),
    ("TrustConfig fa_target", _build("models.config", "TrustConfig", k=2, fa_target=float("nan"))),
    ("LearnedConfig layers", _build("models.config", "LearnedConfig", layers=2.5)),
    ("LearnedConfig features", _build("models.config", "LearnedConfig", features="everything")),
    ("LearnedConfig batch_size", _build("models.config", "LearnedConfig", batch_size=2.5)),
    ("LearnedConfig epochs", _build("models.config", "LearnedConfig", epochs=0)),
    ("LearnedConfig lr", _build("models.config", "LearnedConfig", lr=float("nan"))),
    ("LearnedConfig dropout", _build("models.config", "LearnedConfig", dropout=1.5)),
    (
        "spectral_partition K",
        _build("federated.partition", "spectral_partition", edge_index=_EDGES, N=3, K=2.5),
    ),
    (
        "spectral_partition N",
        _build("federated.partition", "spectral_partition", edge_index=_EDGES, N="3", K=1),
    ),
]


@pytest.mark.parametrize("name, call", ENTRY_POINTS, ids=[n for n, _ in ENTRY_POINTS])
def test_an_entry_point_refuses_a_bad_setting_with_a_config_error(name, call):
    with pytest.raises(ConfigError):
        call()


@pytest.mark.parametrize(
    "assignment, edge_index",
    [
        ("abc", _EDGES),
        (None, _EDGES),
        (np.array([0, 0, 1]), "edges"),
        (np.array([0, 0, 1]), np.array([[0, -1], [1, 2]])),
        (np.array([0.5, 0, 1]), _EDGES),
        (np.array([0, 0, 5]), _EDGES),
    ],
)
def test_partition_from_assignment_refuses_a_malformed_input(assignment, edge_index):
    from fdia_analysis.federated.partition import partition_from_assignment

    with pytest.raises(ConfigError):
        partition_from_assignment(assignment, edge_index)


@pytest.mark.parametrize(
    "call",
    [
        lambda: _build("formulas.federated", "pool_moments", parts=None)(),
        lambda: _build("formulas.federated", "fedavg", arrays=None, weights=[1.0])(),
        lambda: _build("formulas.federated", "block_diagonal_basis", blocks=3, d=4)(),
        lambda: inputs.ShapedArray("bad", (2, 2), "scores"),
    ],
)
def test_a_raw_argument_reaches_its_model_unconverted(call):
    with pytest.raises(ConfigError):
        call()


def test_no_function_outside_the_models_is_a_check():
    found = check_functions(fdia_analysis)
    assert not found, f"declare these checks on a model in models/: {found}"


def test_no_function_converts_its_own_argument_before_the_model_sees_it():
    """The analysis code builds its own models and the generation toolkit's; neither gets an argument
    the caller converted first."""
    hits = converted_arguments(fdia_analysis, config, inputs, generation_config, generation_inputs)
    assert not hits, "convert inside the model instead: " + "; ".join(hits)
