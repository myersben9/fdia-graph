"""`Case.from_matpower` on a three-bus case file written here: what it reads and what it refuses. Needs no input
data."""

import dataclasses

import pytest

from fdia_graph.errors import DataConditionError
from fdia_graph.generation.nygrid.case import Case

TINY = """function mpc = tiny
% a three-bus case
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [
	1	3	0	0	0	0	1	1	0	345	1	1.1	0.9;
	2	1	50	10	0	0	1	1	0	345	11	1.1	0.9;
	7	2	0	0	0	0	2	1	0	115	12	1.1	0.9;
];
mpc.gen = [
	1	0	0	9999	-9999	1	100	1	200	0;
	7	0	0	50	-50	1.02	100	1	80	10;
];
mpc.branch = [
	1	2	0.01	0.1	0.02	0	0	0	0	0	1	-360	360;
	2	7	0.02	0.2	0	150	0	0	0.975	0	1	-360	360;
];
mpc.bus_name = {
	'ONE';
	'TWO';
	'SEVEN';
};
mpc.bus_latlon = [
	42.1	-75.2;
	42.5	-74.9;
	41	-72.5;
];
mpc.genfuel = {
	'Hydro';
	'Import';
};
mpc.gen_ramp = [
	NaN;
	2.5;
];
"""
BUS_TWO = "\t2\t1\t50\t10\t"
GEN_TWO = "\t7\t0\t0\t50\t"
ZONE_TWO = "345\t11\t"


def differing(a: Case, b: Case) -> list[str]:
    """The fields of two cases that are not the same bits: dtype, shape and every value, NaN included."""
    out = []
    for field in dataclasses.fields(Case):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if field.name == "base_mva":
            same = type(x) is type(y) and x == y
        else:
            same = x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        if not same:
            out.append(field.name)
    return out


def read(tmp_path, text: str) -> Case:
    (tmp_path / "tiny.m").write_text(text, encoding="utf-8")
    return Case.from_matpower(tmp_path / "tiny.m")


def edited(old: str, new: str) -> str:
    assert TINY.count(old) == 1, old
    return TINY.replace(old, new)


def test_the_tiny_case(tmp_path):
    case = read(tmp_path, TINY)
    assert case.base_mva == 100.0 and case.bus_id.tolist() == [1, 2, 7]
    assert case.name.tolist() == ["ONE", "TWO", "SEVEN"] and case.fuel.tolist() == ["Hydro", "Import"]
    assert case.f.tolist() == [0, 1] and case.t.tolist() == [1, 2] and case.unit_bus.tolist() == [0, 2]
    assert case.tap.tolist() == [1.0, 0.975] and case.rating.tolist() == [0.0, 150.0]
    assert case.lat.tolist() == [42.1, 42.5, 41.0] and str(case.ramp.tolist()) == "[nan, 2.5]"
    assert len(case.dc_f) == 0 and case.unit_on.all()


def test_the_zone_column_is_the_load_zone(tmp_path):
    case = read(tmp_path, TINY)  # zone 1 is A, 11 is K, 12 is a bus outside New York
    assert case.zone.tolist() == ["A", "K", ""] and case.is_ny.tolist() == [True, True, False]


# Hand edits that must read to the same case: (what it is, the canonical text, the text in its place).
SAME = [
    ("a comment after a name", "'ONE';", "'ONE'; % edited, it's fine"),
    ("a comment after a fuel", "'Hydro';", "'Hydro'; % run of river"),
    ("a space before a semicolon", "'ONE';", "'ONE' ;"),
    ("a comma after a name", "'ONE';", "'ONE',"),
    ("two names on one line", "\t'ONE';\n\t'TWO';", "\t'ONE'; 'TWO';"),
    ("a comment after a row", "\t42.5\t-74.9;", "\t42.5\t-74.9; % moved"),
    ("a comment line in a matrix", "mpc.gen = [\n", "mpc.gen = [\n% the units\n"),
    (
        "a standard field that is passed over",
        "mpc.genfuel",
        "mpc.gencost = [\n\t2 0 0 3 0 1 0;\n];\nmpc.genfuel",
    ),
]


@pytest.mark.parametrize("old, new", [edit[1:] for edit in SAME], ids=[edit[0] for edit in SAME])
def test_a_harmless_edit_reads_to_the_same_case(tmp_path, old, new):
    assert differing(read(tmp_path, TINY), read(tmp_path, edited(old, new))) == []


def test_a_branch_out_of_service_is_left_out(tmp_path):
    case = read(tmp_path, edited("0.975\t0\t1\t-360", "0.975\t0\t0\t-360"))
    assert case.f.tolist() == [0] and case.t.tolist() == [1]


# Hand edits that must be refused: (what it is, the canonical text, the text in its place, what the message says).
REFUSED = [
    ("a name missing", "\t'SEVEN';\n", "", r"mpc\.bus_name has 2 rows, the case has 3"),
    (
        "a fuel too many",
        "\t'Import';\n",
        "\t'Import';\n\t'Coal';\n",
        r"mpc\.genfuel has 3 rows, the case has 2",
    ),
    ("a ramp too many", "\t2.5;\n", "\t2.5;\n\t3;\n", r"mpc\.gen_ramp has 3 rows, the case has 2"),
    ("a place missing", "\t41\t-72.5;\n", "", r"mpc\.bus_latlon has 2 rows, the case has 3"),
    (
        "a bus number that is not whole",
        BUS_TWO,
        "\t2.5\t1\t50\t10\t",
        r"mpc\.bus: a bus number \(BUS_I\) must be a whole",
    ),
    (
        "a bus number on two rows",
        BUS_TWO,
        "\t1\t1\t50\t10\t",
        r"mpc\.bus: bus number 1 is on more than one row",
    ),
    ("a bus type outside 1, 2, 3", BUS_TWO, "\t2\t4\t50\t10\t", r"mpc\.bus: a bus type must be 1, 2 or 3"),
    (
        "no reference bus",
        "\t1\t3\t0\t0\t0\t0\t1\t1",
        "\t1\t2\t0\t0\t0\t0\t1\t1",
        r"mpc\.bus has 0 reference buses",
    ),
    ("two reference buses", BUS_TWO, "\t2\t3\t50\t10\t", r"mpc\.bus has 2 reference buses"),
    (
        "a unit on a bus that is not there",
        GEN_TWO,
        "\t9\t0\t0\t50\t",
        r"mpc\.gen: bus 9 is not a bus of mpc\.bus",
    ),
    (
        "a branch to a bus that is not there",
        "\t2\t7\t0.02",
        "\t2\t8\t0.02",
        r"mpc\.branch: bus 8 is not a bus",
    ),
    (
        "a zone number past 12",
        ZONE_TWO,
        "345\t13\t",
        r"mpc\.bus: a zone \(ZONE\) must be a whole number, 1 to 11",
    ),
    ("a zone number that is not whole", ZONE_TWO, "345\t1.5\t", r"mpc\.bus: a zone \(ZONE\) must be"),
    (
        "an indexed assignment",
        "mpc.genfuel",
        "mpc.bus(1, 3) = 500;\nmpc.genfuel",
        r"line 28: `mpc\.bus\(1, 3\) = 500;` is an indexed assignment",
    ),
    (
        "a field the reader does not know",
        "mpc.genfuel",
        "mpc.bus_zone = {'A'};\nmpc.genfuel",
        r"line 28: .*mpc.bus_zone is not a field",
    ),
    (
        "a field assigned twice",
        "mpc.genfuel",
        "mpc.gen_ramp = [1; 2];\nmpc.genfuel",
        r"line 33: .*mpc.gen_ramp is assigned twice",
    ),
    ("a needed field left out", "mpc.gen_ramp = [\n\tNaN;\n\t2.5;\n];\n", "", r"has no mpc\.gen_ramp"),
    (
        "a row one value short",
        "\t42.5\t-74.9;",
        "\t42.5;",
        r"cannot be read as a MATPOWER case file: .*inhomogeneous",
    ),
    ("a word among the numbers", "\t2.5;\n", "\tfast;\n", r"cannot be read as a MATPOWER case file: .*fast"),
]


@pytest.mark.parametrize(
    "old, new, message", [edit[1:] for edit in REFUSED], ids=[edit[0] for edit in REFUSED]
)
def test_an_edit_that_breaks_the_case_is_refused(tmp_path, old, new, message):
    with pytest.raises(DataConditionError, match=message) as refusal:
        read(tmp_path, edited(old, new))
    assert "tiny.m" in str(refusal.value)  # every refusal names the file
