"""A malformed setting or argument is a `ConfigError`, never a raw TypeError, KeyError, ValueError
or IndexError from deeper in the code (docs/reference/REVIEW_CHECKLIST.md, item 4).

Two sweeps. Every config and input model is built with each of its fields replaced by a value of
the wrong kind, and must either accept it or refuse it with a `ConfigError`. Then each public
entry point that takes settings is called with one bad argument, and must refuse it with a
`ConfigError` before it downloads, reads a file or starts work.
"""

import dataclasses
import inspect
import re

import numpy as np
import pytest

from fdia_graph.errors import ConfigError
from fdia_graph.models import config, inputs
from fdia_graph.models.frames import OperatingLimits
from fdia_graph.models.validation import Validated


def _id(value: object) -> str:
    """A test id without memory addresses, so parallel workers (pytest-xdist) collect the same ids."""
    return re.sub(r" at 0x[0-9a-fA-F]+", "", str(value))


BAD = ("bogus", 1.5, -1, float("nan"), float("inf"), 10**1000, None, object(), [1, 2], {"a": 1})

_EDGES = np.array([[0, 1], [1, 2]])
_ADJ = np.eye(3)
_GRAPH = inputs.ClientGraph(np.array([0, 0, 1]), _ADJ)
_LIMITS = OperatingLimits(*(np.full(3, v) for v in (0.94, 1.06, -np.inf, np.inf, -np.inf, np.inf)))

# a valid value for every required field, so each bad value is the only thing wrong
VALID = {
    config.WindowSpec: dict(T=10, W=2),
    config.TrustConfig: dict(k=2),
    config.TrustSchedule: dict(buses=[0], slots=[1]),
    config.WuDefenseConfig: dict(pmus=[0], slots=[1]),
    config.WuDqnConfig: dict(),
    config.ProfileFetch: dict(iso="nyiso"),
    config.GateConfig: dict(gate="oracle"),
    config.IsoExport: dict(iso="nyiso"),
    inputs.DateSpan: dict(start="2024-01-01", end="2024-01-02"),
    inputs.ProfileSource: dict(source=[1.0, 2.0]),
    inputs.DatasetName: dict(name="ieee14", local=frozenset(), builtin=frozenset({"ieee14"})),
    inputs.AdmissibleTargets: dict(families=("Aq", "Ad"), targets={1: 2, 2: 3}),
    inputs.CertifiableLimits: dict(limits=_LIMITS),
    inputs.TrustablePmus: dict(buses=[0], pmu=frozenset({0}), n_bus=3),
    inputs.WindowSlots: dict(slots=[1], snapshots=3),
    inputs.SameDefense: dict(actions=[4, 4], steps=[2, 2]),
    inputs.ChosenAction: dict(action=0, valid=[True, False]),
    inputs.FamilySelection: dict(families=("Aq",)),
    inputs.GeneratedFamilies: dict(families=("At",)),
    inputs.SystemRef: dict(system="ieee14"),
    inputs.SupportedSystem: dict(system="ieee14", supported=frozenset({14})),
    inputs.ReleaseName: dict(release="v0.8.0"),
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
    inputs.LoadValues: dict(values=[1.0, 2.0, 3.0]),
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
    ids=_id,
)
def test_load_refuses_a_bad_argument_before_any_download(kwargs, monkeypatch):
    import fdia_graph as fg

    monkeypatch.setattr(fg, "ensure_local", _no_work)
    with pytest.raises(ConfigError):
        fg.load("ieee14", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [dict(iso="bogus"), dict(iso=None), dict(resample_min="five"), dict(resample_min=-5)],
    ids=_id,
)
def test_fetch_profile_refuses_a_bad_argument_before_the_network(kwargs, monkeypatch):
    from fdia_graph import profiles

    monkeypatch.setattr(profiles, "requests", None, raising=False)
    args = {"iso": "nyiso", **kwargs}
    with pytest.raises(ConfigError):
        profiles.fetch_profile(args.pop("iso"), "2024-01-01", "2024-01-02", **args)


@pytest.mark.parametrize(
    "start, end", [("bogus", "2024-01-02"), ("2024-01-01", None), (20240101, "2024-01-02")]
)
def test_fetch_profile_refuses_a_bad_date_before_the_network(start, end, monkeypatch):
    from fdia_graph import profiles

    monkeypatch.setattr(profiles, "_FEEDS", {})  # reaching a feed is the failure
    with pytest.raises(ConfigError, match="DateSpan"):
        profiles.fetch_profile("nyiso", start, end)


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
    ],
    ids=_id,
)
def test_timeline_knobs_refuse_a_bad_value_before_any_work(knobs):
    with pytest.raises(ConfigError):
        config.TimelineKnobs(**knobs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(families=["Zz"]),
        dict(families="Aq"),
        dict(families=["Ad"]),
        dict(am_len=2.5),
        dict(ramp_len=0),
        dict(attacked_frac=2.0),
    ],
    ids=_id,
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


@pytest.mark.parametrize(
    "build",
    [
        lambda: inputs.ProfileSource(3),  # a scalar is not a load series
        lambda: inputs.ProfileSource({"a": 1.0}),  # nor is a mapping
        lambda: inputs.StateSource(3),  # a pool is an array or a path
        lambda: inputs.DateSpan(20240101, "2024-01-02"),  # '20240131' reads as a date from Python 3.11
        lambda: inputs.DateSpan("20240101", "2024-01-02"),
        lambda: config.ExportRequest("numpy", 3),
        lambda: config.ExportRequest("numpy", "node_x"),  # one string, not split into letters
        lambda: config.ExportRequest("numpy", {"node_x": 1}),  # a mapping, not read for its keys
        lambda: config.ExportRequest("numpy", {"node_x"}),  # a set has no order
        lambda: inputs.LoadValues(3),  # a scalar is not a series
        lambda: inputs.CsvSpec("x.csv", None),  # a CSV source names its column
        lambda: inputs.DateSpan("2024-01-02", "2024-01-01"),  # reversed
        lambda: config.IsoExport("nyiso", None),
        lambda: config.IsoExport("nyiso", 3),
        lambda: inputs.CsvSpec(3, "load_mw"),
        lambda: inputs.DatasetName(["ieee14"], frozenset(), frozenset({"ieee14"})),  # unhashable
        lambda: inputs.DatasetName(True, frozenset(), frozenset({"ieee14"})),
        lambda: inputs.LoadValues([]),
        lambda: inputs.LoadValues([[1.0, 2.0], [3.0, 4.0]]),
    ],
)
def test_a_source_or_field_list_of_the_wrong_kind_is_refused(build):
    with pytest.raises(ConfigError):
        build()


def test_a_csv_source_and_a_dataset_name_are_checked_when_given():
    import fdia_graph as fg
    from fdia_graph.profiles import CsvColumn
    from fdia_graph.registry import resolve

    with pytest.raises(ConfigError, match="CsvSpec"):
        CsvColumn("x.csv", None)
    with pytest.raises(ConfigError, match="DatasetName"):
        resolve(["ieee14"])
    assert resolve(118).name == resolve("118").name == resolve(" IEEE118 ").name == "ieee118"
    with pytest.raises(ConfigError, match="DatasetName"):
        fg.load(["ieee14"])


def test_a_state_pool_path_may_be_a_pathlib_path(tmp_path):
    import pathlib

    assert inputs.StateSource(pathlib.Path(tmp_path) / "pool.npz").path == str(tmp_path / "pool.npz")


def test_export_refuses_a_malformed_field_list(timeline):
    import fdia_graph as fg

    with pytest.raises(ConfigError, match="ExportRequest.fields"):
        fg.load(timeline, split="test").export(format="numpy", fields=3)


@pytest.mark.parametrize(
    "call",
    [
        lambda: __import__("fdia_graph.dataset.base", fromlist=["x"]).family_ids(3),
        lambda: __import__("fdia_graph.formulas.federated", fromlist=["x"]).pool_moments(None),
        lambda: __import__("fdia_graph.formulas.federated", fromlist=["x"]).fedavg(None, [1.0]),
        lambda: __import__("fdia_graph.formulas.federated", fromlist=["x"]).block_diagonal_basis(3, 4),
        lambda: inputs.ShapedArray("bad", (2, 2), "scores"),
        lambda: inputs.StateSource(np.array(["bad"])),
    ],
)
def test_a_raw_argument_reaches_its_model_unconverted(call):
    with pytest.raises(ConfigError):
        call()


def test_no_function_converts_its_own_argument_before_the_model_sees_it():
    """`Model(tuple(arg))` or `Model(np.asarray(arg))` turns a malformed argument into a raw
    TypeError or ValueError before the model can refuse it; the model's rules do the conversion."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "fdia_graph"
    names = {
        n
        for m in (config, inputs)
        for n, c in vars(m).items()
        if inspect.isclass(c) and issubclass(c, Validated) and c is not Validated
    }
    conversions = {"tuple", "list", "float", "int", "np.asarray", "np.array", "np.ascontiguousarray"}
    hits = []
    for path in sorted(src.rglob("*.py")):
        if "models" in path.parts:
            continue
        for fn in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            for node in ast.walk(fn):
                if not (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names
                ):
                    continue
                for arg in list(node.args) + [k.value for k in node.keywords]:
                    for c in ast.walk(arg):
                        if (
                            isinstance(c, ast.Call)
                            and ast.unparse(c.func) in conversions
                            and c.args
                            and isinstance(c.args[0], ast.Name)
                            and c.args[0].id in params
                        ):
                            hits.append(f"{path.relative_to(src)}:{node.lineno} {ast.unparse(c)}")
    assert not hits, "convert inside the model instead: " + "; ".join(hits)
