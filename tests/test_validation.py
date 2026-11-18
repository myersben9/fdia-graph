"""The validation engine (models.validation): rules declared on fields, one message shape, and the
public entry points that build their config models before any work."""

from dataclasses import dataclass
from typing import Annotated, Optional

import numpy as np
import pytest

from fdia_graph.errors import ConfigError
from fdia_graph.models.choices import Choice, Iso
from fdia_graph.models.validation import AtLeast, Boolean, InRange, Integer, OneOf, Positive, Validated


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


def test_a_view_is_refused_through_the_capability_table(timeline):
    import fdia_graph as fg
    from fdia_graph.dataset.base import CAPABILITIES
    from fdia_graph.errors import MissingCapability
    from fdia_graph.models.choices import Capability

    assert set(CAPABILITIES) == set(Capability)  # every capability has its test and its phrase
    pu = fg.load(timeline, split="test", units="pu")
    pu.require("time_order", "benign_layer", by="anything")  # what a timeline offers passes
    with pytest.raises(MissingCapability, match="^the estimator needs units='physical'"):
        pu.require("physical_units", by="the estimator")
    with pytest.raises(MissingCapability, match="needs order='time'"):
        fg.load(timeline, split="test", order="random").windows(4)


def test_the_gate_exempts_the_models_package_only(tmp_path, monkeypatch):
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import readability

    sibling = tmp_path / "models_extra.py"
    sibling.write_text("def f():\n    raise ValueError('x')\n")
    monkeypatch.setattr(readability, "ROOTS", {"package": str(tmp_path)})
    monkeypatch.setattr(readability, "_MODELS", ("package/models/",))
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
    monkeypatch.setattr(readability, "ROOTS", {"package": str(tmp_path)})
    monkeypatch.setattr(readability, "_MODELS", ("package/models/",))
    assert readability.hand_checks(str(inside)) == []
    assert [h[2] for h in readability.hand_checks(str(outside))] == ["isinstance"]


def test_parsers_are_models_that_keep_the_input_and_give_the_result():
    from fdia_graph.models.inputs import FamilySelection, ReleaseName, SystemRef

    assert SystemRef("IEEE118").number == 118 and ReleaseName("data-v0.8.3").numbers == (0, 8, 3)
    assert FamilySelection(["Aq", 2, "Am"]).codes == (1, 2, 7)
    with pytest.raises(ConfigError, match="unknown family"):
        FamilySelection(["Aq", "Zz"])
    with pytest.raises(ConfigError, match="SystemRef.system must be like 'ieee118' or 118"):
        SystemRef("big")


def test_a_required_field_refuses_none_with_its_phrase():
    from fdia_graph.models.inputs import CsvSpec

    with pytest.raises(ConfigError, match=r"^CsvSpec\.column is required$"):
        CsvSpec("x.csv", None)


def test_a_capability_name_and_a_fetch_are_checked_by_their_models(timeline):
    import fdia_graph as fg
    from fdia_graph.data.profiles import fetch_profile

    with pytest.raises(ConfigError, match=r"^Requirement\.capabilities must name capabilities"):
        fg.load(timeline, split="test").require("timelime", by="a typo")
    with pytest.raises(ConfigError, match=r"^ProfileFetch\.iso must be one of"):
        fetch_profile("pjm", "2024-01-01", "2024-01-02")
    with pytest.raises(ConfigError, match=r"^ProfileFetch\.resample_min must be >= 1"):
        fetch_profile("nyiso", "2024-01-01", "2024-01-02", resample_min=0)
    with pytest.raises(ConfigError, match=r"^ProfileFetch\.resample_min must be an integer"):
        fetch_profile("nyiso", "2024-01-01", "2024-01-02", resample_min=1.5)  # was truncated to 1


def test_a_requirement_names_who_asks(timeline):
    import fdia_graph as fg

    with pytest.raises(ConfigError, match=r"^Requirement\.by is required$"):
        fg.load(timeline, split="test").require("time_order", by=None)  # type: ignore[arg-type]


def test_the_generator_refuses_a_bad_option():
    """The generator is built on a network the caller made, so its own options are what it checks; a
    timeline's knobs are refused before the case is built (test_malformed_inputs)."""
    pytest.importorskip("pandapower")
    from fdia_graph.engine.core import FdiaGenerator
    from fdia_graph.models.config import MeterSettings
    from fdia_graph.systems.network import Network

    with pytest.raises(ConfigError, match=r"^MeterSettings\.meter_model must be one of \['hybrid'\]"):
        MeterSettings(meter_model="scada")
    with pytest.raises(ConfigError, match=r"^GeneratorOptions\.max_load_mw must be > 0, got -1\.0$"):
        FdiaGenerator(Network.from_ieee(14), max_load_mw=-1.0)


def test_generate_states_refuses_a_system_it_has_no_case_for():
    from fdia_graph.data.ieee_pool import generate_states

    with pytest.raises(ConfigError, match=r"^SupportedSystem: unknown system 15; supported: \["):
        generate_states(15, np.ones(4))
    with pytest.raises(ConfigError, match=r"^SupportedSystem\.system must be like 'ieee118' or 118"):
        generate_states("grid", np.ones(4))


def test_a_boolean_field_refuses_a_truthy_string():
    @dataclass(frozen=True)
    class Switch(Validated):
        on: Annotated[bool, Boolean()] = False

    assert Switch(True).on is True and Switch(np.bool_(False)).on == False  # noqa: E712
    with pytest.raises(ConfigError, match="Switch.on must be True or False"):
        Switch("False")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="must be True or False"):
        Switch(1)  # type: ignore[arg-type]
