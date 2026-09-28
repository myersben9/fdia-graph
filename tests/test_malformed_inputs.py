"""A malformed setting or argument is a `ConfigError`, never a raw TypeError, KeyError, ValueError
or IndexError from deeper in the code (docs/reference/REVIEW_CHECKLIST.md, item 4).

Two sweeps. Every config and input model is built with each of its fields replaced by a value of
the wrong kind, and must either accept it or refuse it with a `ConfigError`. Then each public
entry point that takes settings is called with one bad argument, and must refuse it with a
`ConfigError` before it downloads, reads a file or starts work.
"""

import dataclasses
import inspect

import numpy as np
import pytest

from fdia_graph.errors import ConfigError
from fdia_graph.models import config, inputs
from fdia_graph.models.validation import Validated

BAD = ("bogus", 1.5, -1, float("nan"), float("inf"), 10**1000, None, object(), [1, 2], {"a": 1})

_EDGES = np.array([[0, 1], [1, 2]])
_ADJ = np.eye(3)
_GRAPH = inputs.ClientGraph(np.array([0, 0, 1]), _ADJ)

# a valid value for every required field, so each bad value is the only thing wrong
VALID = {
    config.WindowSpec: dict(T=10, W=2),
    config.TrustConfig: dict(k=2),
    config.ProfileFetch: dict(iso="nyiso"),
    config.GateConfig: dict(gate="oracle"),
    inputs.AdmissibleTargets: dict(families=("Aq", "Ad"), targets={1: 2, 2: 3}),
    inputs.FamilySelection: dict(families=("Aq",)),
    inputs.SystemRef: dict(system="ieee14"),
    inputs.SupportedSystem: dict(system="ieee14", supported=frozenset({14})),
    inputs.StreamSystem: dict(system="ieee14"),
    inputs.ReleaseName: dict(release="v0.8.0"),
    inputs.OutageRef: dict(outage=0, names=("a", "b"), indices=(0, 1)),
    inputs.StatePool: dict(X=np.zeros((2, 3, 4))),
    inputs.ShapedArray: dict(values=np.zeros((2, 3)), shape=(2, 3)),
    inputs.FieldRequest: dict(fields=("a",), offered=("a", "b"), known=("a", "b")),
    inputs.CsvSpec: dict(path="x.csv", column="load"),
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
    inputs.Requirement: dict(capabilities=("timeline",), by="a test"),
}


def _models():
    for mod in (config, inputs):
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if issubclass(cls, Validated) and cls is not Validated and cls.__module__ == mod.__name__:
                yield cls


MODELS = list(_models())


def test_every_model_has_a_valid_example():
    for cls in MODELS:
        cls(**VALID.get(cls, {}))  # the base every sweep case starts from builds cleanly


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_a_wrong_kind_of_value_is_accepted_or_a_config_error(cls):
    base = VALID.get(cls, {})
    for field in dataclasses.fields(cls):
        for bad in BAD:
            try:
                cls(**{**base, field.name: bad})
            except (ConfigError, cls.error):  # an input error, or the data condition the model names
                pass
            except Exception as e:  # noqa: BLE001  (any other type is the finding)
                pytest.fail(f"{cls.__name__}({field.name}={bad!r}) raised {type(e).__name__}: {e}")


# ---- the public entry points --------------------------------------------------------------------
def _no_work(*a, **k):
    raise AssertionError("a bad argument got past the checks")


def _build(module, cls_name, **kw):
    import importlib

    return lambda: getattr(importlib.import_module(f"fdia_graph.{module}"), cls_name)(**kw)


def _learned(**kw):
    def call():
        pytest.importorskip("torch")
        from fdia_graph.localization import BusMLP

        BusMLP(**kw)

    return call


ENTRY_POINTS = [
    ("WLS npass", _build("se", "WLS", npass="many")),
    ("WLS iters", _build("se", "WLS", iters=2.5)),
    ("AdaptiveWeighting c", _build("se", "AdaptiveWeighting", c=float("nan"))),
    ("AdaptiveWeighting tol", _build("se", "AdaptiveWeighting", tol="small")),
    ("ResidualRemoval threshold", _build("se", "ResidualRemoval", threshold="high")),
    ("ResidualRemoval cond_mult", _build("se", "ResidualRemoval", cond_mult=-1)),
    ("SubspacePrior rank_frac", _build("se", "SubspacePrior", rank_frac=10**1000)),
    ("SubspacePrior reweight", _build("se", "SubspacePrior", reweight="tukey")),
    ("JacobianWeighting huber_c", _build("se", "JacobianWeighting", huber_c=float("inf"))),
    ("GatedPrior gate_factor", _build("se", "GatedPrior", gate_factor="big")),
    ("SwingThreshold fa_target", _build("localization", "SwingThreshold", fa_target=1.5)),
    ("SwingThreshold fa_target kind", _build("localization", "SwingThreshold", fa_target="low")),
    ("ResidualLocalizer fa_target", _build("localization", "ResidualLocalizer", fa_target=None)),
    ("TrustedMeters k", _build("trust", "TrustedMeters", k=2.5)),
    ("TrustedMeters fa_target", _build("trust", "TrustedMeters", k=2, fa_target=float("nan"))),
    ("BusMLP layers", _learned(layers=2.5)),
    ("BusMLP features", _learned(features="everything")),
    ("BusMLP batch_size", _learned(batch_size=2.5)),
    ("BusMLP epochs", _learned(epochs=0)),
    ("BusMLP lr", _learned(lr=float("nan"))),
    ("BusMLP dropout", _learned(dropout=1.5)),
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
    "kwargs",
    [
        dict(split=3),
        dict(units=None),
        dict(order=["time"]),
        dict(format=1.5),
        dict(families="Aq"),
        dict(families=[1.5]),
        dict(families=[object()]),
        dict(release="latest"),
        dict(release=8),
    ],
    ids=str,
)
def test_load_refuses_a_bad_argument_before_any_download(kwargs, monkeypatch):
    import fdia_graph as fg

    monkeypatch.setattr(fg, "ensure_local", _no_work)
    with pytest.raises(ConfigError):
        fg.load("ieee14", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(layer="bogus"),
        dict(layer=None),
        dict(train_frac="most"),
        dict(train_frac=float("nan")),
        dict(val_frac=-0.1),
        dict(max_test=2.5),
    ],
    ids=str,
)
def test_pyg_stream_refuses_a_bad_argument_before_loading(kwargs, monkeypatch):
    from fdia_graph import torch_data

    monkeypatch.setattr(torch_data, "_resolve_stream", _no_work)
    with pytest.warns(DeprecationWarning), pytest.raises(ConfigError):
        torch_data.pyg_stream("ieee14", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [dict(layer="bogus"), dict(train_frac=1.5), dict(val_frac="some")],
    ids=str,
)
def test_torch_windows_refuses_a_bad_argument_before_loading(kwargs, monkeypatch):
    from fdia_graph import torch_data

    monkeypatch.setattr(torch_data, "_resolve_stream", _no_work)
    with pytest.warns(DeprecationWarning), pytest.raises(ConfigError):
        torch_data.torch_windows("ieee14", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [dict(iso="bogus"), dict(iso=None), dict(resample_min="five"), dict(resample_min=-5)],
    ids=str,
)
def test_fetch_profile_refuses_a_bad_argument_before_the_network(kwargs, monkeypatch):
    from fdia_graph import profiles

    monkeypatch.setattr(profiles, "requests", None, raising=False)
    args = {"iso": "nyiso", **kwargs}
    with pytest.raises(ConfigError):
        profiles.fetch_profile(args.pop("iso"), "2024-01-01", "2024-01-02", **args)


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
    from fdia_graph.federated.partition import partition_from_assignment

    with pytest.raises(ConfigError):
        partition_from_assignment(assignment, edge_index)


@pytest.mark.parametrize(
    "knobs",
    [
        dict(attacked_frac=1.5),
        dict(attacked_frac="half"),
        dict(ramp_len=2.5),
        dict(hops=0),
        dict(am_direction="sideways"),
        dict(corrupt_len=-1),
        dict(am_rate=float("inf")),
    ],
    ids=str,
)
def test_timeline_knobs_refuse_a_bad_value_before_any_work(knobs):
    with pytest.raises(ConfigError):
        config.TimelineKnobs(**knobs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(families=["Zz"]),
        dict(families="Aq"),
        dict(am_len=2.5),
        dict(ramp_len=0),
        dict(attacked_frac=2.0),
        dict(am_direction="sideways"),
    ],
    ids=str,
)
def test_generate_timeline_refuses_a_bad_knob_before_building_the_case(kwargs, monkeypatch):
    from fdia_graph import timeline

    monkeypatch.setattr(timeline, "FdiaGenerator", _no_work)
    with pytest.raises(ConfigError):
        timeline.generate_timeline(14, **kwargs)


# the deprecated shims kept for one minor version; every other check lives on a model
_SHIMS = {"check_split", "check_units", "check_order", "check_partition"}


def test_no_function_outside_the_models_is_a_check():
    import ast
    import pathlib

    import fdia_graph

    root = pathlib.Path(fdia_graph.__file__).parent
    found = [
        f"{path.relative_to(root)}:{node.lineno} {node.name}"
        for path in root.rglob("*.py")
        if "models" not in path.relative_to(root).parts
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("check_") and node.name not in _SHIMS
    ]
    assert not found, f"declare these checks on a model in models/: {found}"
