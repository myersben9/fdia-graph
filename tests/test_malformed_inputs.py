"""A malformed setting or argument is a `ConfigError`, never a raw TypeError, KeyError, ValueError
or IndexError from deeper in the code (docs/reference/REVIEW_CHECKLIST.md, item 4).

Two sweeps. Every config and input model is built with each of its fields replaced by a value of
the wrong kind, and must either accept it or refuse it with a `ConfigError`. Then each public
entry point that takes settings is called with one bad argument, and must refuse it with a
`ConfigError` before it downloads, reads a file or starts work. The analysis toolkit's models and
entry points get the same sweeps in analysis/tests/test_malformed_inputs.py, through the same helpers (tests/helpers.py).
"""

import pathlib
import re

import numpy as np
import pytest

from fdia_graph.errors import ConfigError
from fdia_graph.models import config, inputs
from fdia_graph.models.frames import OperatingLimits
from tests.helpers import check_functions, converted_arguments, models_of, sweep


def _id(value: object) -> str:
    """A test id without memory addresses, so parallel workers (pytest-xdist) collect the same ids."""
    return re.sub(r" at 0x[0-9a-fA-F]+", "", str(value))


_LIMITS = OperatingLimits(*(np.full(3, v) for v in (0.94, 1.06, -np.inf, np.inf, -np.inf, np.inf)))

# a valid value for every required field, so each bad value is the only thing wrong
VALID = {
    config.WindowSpec: dict(T=10, W=2),
    config.TrustSchedule: dict(buses=[0], slots=[1]),
    config.ProfileFetch: dict(iso="nyiso"),
    config.SolveSettings: dict(method="opf", engine="ipopt"),
    config.NewYorkSolve: dict(),
    config.IsoExport: dict(iso="nyiso"),
    inputs.DateSpan: dict(start="2024-01-01", end="2024-01-02"),
    inputs.ProfileSource: dict(source=type("Series", (), {"loads": lambda self: np.ones(2)})()),
    inputs.DatasetName: dict(name="ieee14", local=frozenset(), builtin=frozenset({"ieee14"})),
    inputs.AdmissibleTargets: dict(families=("Aq", "Ad"), targets={1: 2, 2: 3}),
    inputs.OwnFamilies: dict(names=("At",), placed=("At", "Am")),
    inputs.OwnedBuses: dict(buses=3, owned=3, names=("a", "b")),
    inputs.ReadableLayout: dict(path="t.h5", edge_feat="P_from,Q_from", current_form="rectangular"),
    inputs.SplitRule: dict(split_by="time", frames=3),
    inputs.ModelledNet: dict(
        out_of_service=(), scaled=(), not_read=(), open_branches=0, buses=14, solved_buses=14
    ),
    inputs.TrustablePmus: dict(buses=[0], pmu=frozenset({0}), n_bus=3),
    inputs.FamilySelection: dict(families=("Aq",)),
    inputs.GeneratedFamilies: dict(families=("At",)),
    inputs.SystemRef: dict(system="ieee14"),
    inputs.SupportedSystem: dict(system="ieee14", supported=frozenset({14})),
    inputs.ReleaseName: dict(release="v0.8.0"),
    inputs.StatePool: dict(X=np.zeros((2, 3, 4))),
    inputs.UnitCosts: dict(linear=np.ones(2), quadratic=np.zeros(2)),
    inputs.OpfPool: dict(system=14, steps=3, solved=2),
    inputs.FieldRequest: dict(fields=("a",), offered=("a", "b"), known=("a", "b")),
    inputs.CsvSpec: dict(path="x.csv", column="load"),
    inputs.Requirement: dict(capabilities=("time_order",), by="a test"),
    inputs.LoadValues: dict(values=[1.0, 2.0, 3.0]),
}


MODELS = models_of(config, inputs)


def test_every_model_has_a_valid_example():
    for cls in MODELS:
        cls(**VALID.get(cls, {}))  # the base every sweep case starts from builds cleanly


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_a_wrong_kind_of_value_is_accepted_or_a_config_error(cls):
    sweep(cls, VALID.get(cls, {}))


# ---- the public entry points --------------------------------------------------------------------
def _no_work(*a, **k):
    raise AssertionError("a bad argument got past the checks")


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
    from fdia_graph.data import profiles

    monkeypatch.setattr(profiles, "requests", None, raising=False)
    args = {"iso": "nyiso", **kwargs}
    with pytest.raises(ConfigError):
        profiles.fetch_profile(args.pop("iso"), "2024-01-01", "2024-01-02", **args)


@pytest.mark.parametrize(
    "start, end", [("bogus", "2024-01-02"), ("2024-01-01", None), (20240101, "2024-01-02")]
)
def test_fetch_profile_refuses_a_bad_date_before_the_network(start, end, monkeypatch):
    from fdia_graph.data import profiles

    monkeypatch.setattr(profiles, "_FEEDS", {})  # reaching a feed is the failure
    with pytest.raises(ConfigError, match="DateSpan"):
        profiles.fetch_profile("nyiso", start, end)


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
        config.TimelineSettings.of(**knobs)


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
    from fdia_graph.systems.network import Network

    monkeypatch.setattr(Network, "from_ieee", _no_work)
    with pytest.raises(ConfigError):
        timeline.generate_timeline(14, **kwargs)


def test_no_function_outside_the_models_is_a_check():
    import fdia_graph

    found = check_functions(fdia_graph)
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
    from fdia_graph.data.profiles import CsvColumn
    from fdia_graph.registry import resolve

    with pytest.raises(ConfigError, match="CsvSpec"):
        CsvColumn("x.csv", None)
    with pytest.raises(ConfigError, match="DatasetName"):
        resolve(["ieee14"])
    assert resolve(118).name == resolve("118").name == resolve(" IEEE118 ").name == "ieee118"
    with pytest.raises(ConfigError, match="DatasetName"):
        fg.load(["ieee14"])


def test_a_state_pool_path_may_be_a_pathlib_path(tmp_path):

    assert inputs.StateSource(pathlib.Path(tmp_path) / "pool.npz").path == str(tmp_path / "pool.npz")


def test_export_refuses_a_malformed_field_list(timeline):
    import fdia_graph as fg

    with pytest.raises(ConfigError, match="ExportRequest.fields"):
        fg.load(timeline, split="test").export(format="numpy", fields=3)


@pytest.mark.parametrize(
    "call",
    [
        lambda: __import__("fdia_graph.dataset.base", fromlist=["x"]).family_ids(3),
        lambda: inputs.StateSource(np.array(["bad"])),
    ],
)
def test_a_raw_argument_reaches_its_model_unconverted(call):
    with pytest.raises(ConfigError):
        call()


def test_no_function_converts_its_own_argument_before_the_model_sees_it():
    import fdia_graph

    hits = converted_arguments(fdia_graph, config, inputs)
    assert not hits, "convert inside the model instead: " + "; ".join(hits)
