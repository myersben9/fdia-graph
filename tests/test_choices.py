"""The `Choice` enums: one declaration per option, one error message, plain strings still accepted."""

import numpy as np
import pytest

from fdia_graph.choices import Choice


class AmDirection(Choice):  # the parameter name comes from the class name
    MASK = "mask"
    INDUCE = "induce"


def test_a_member_is_its_string_value():
    assert AmDirection("mask") is AmDirection.MASK
    assert AmDirection.MASK == "mask" and f"{AmDirection.MASK}" == "mask" and str(AmDirection.MASK) == "mask"
    assert AmDirection(AmDirection.INDUCE) is AmDirection.INDUCE  # a member passes through


def test_an_unknown_value_names_the_parameter_and_every_allowed_value():
    with pytest.raises(ValueError, match=r"am_direction must be one of \['mask', 'induce'\], got 'up'"):
        AmDirection("up")
    assert AmDirection.param() == "am_direction" and AmDirection.values() == ["mask", "induce"]


def test_the_iso_matches_without_regard_to_case():
    from fdia_graph.profiles import Iso

    assert Iso("NYISO") is Iso.NYISO and Iso("Caiso") is Iso.CAISO
    with pytest.raises(ValueError, match="iso must be one of"):
        Iso("pjm")


def test_the_split_codes_are_keyed_by_the_enum_and_read_by_name():
    from fdia_graph.schema import SPLIT_CODE, Split

    assert [SPLIT_CODE[s] for s in ("train", "val", "test")] == [0, 1, 2]
    assert list(SPLIT_CODE) == list(Split)


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda fg: fg.load("ieee14", units="kw"), "units must be one of"),
        (lambda fg: fg.load("ieee14", split="holdout"), "split must be one of"),
        (lambda fg: fg.load("ieee14", order="shuffled"), "order must be one of"),
    ],
)
def test_load_rejects_a_bad_argument_before_any_download(call, message, monkeypatch):
    import fdia_graph as fg

    def no_download(*a, **k):
        raise AssertionError("a bad argument reached the download")

    monkeypatch.setattr(fg, "ensure_local", no_download)  # the name load() calls
    with pytest.raises(ValueError, match=message):
        call(fg)


def test_the_formula_and_model_options_share_the_message():
    from fdia_graph.formulas.projection import meters_to_buses
    from fdia_graph.localization.learned import FEATURE_SETS, Features

    with pytest.raises(ValueError, match="reduce must be one of"):
        meters_to_buses(np.zeros((1, 2)), [np.array([0])], "mean")
    assert FEATURE_SETS["full14+prev"] == 16 and set(FEATURE_SETS) == set(Features)
