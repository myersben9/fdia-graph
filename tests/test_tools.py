"""Two repository tools on small samples: the constants rule of `tools/layout.py` on a temporary tree,
and `tools/count_lines.py` on a file whose lines are easy to miscount."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import count_lines  # noqa: E402
import layout  # noqa: E402

TREE = {
    "pkg/constants/facts.py": "SPEED = 3.0\n",  # a home of constants: any name may live here
    "pkg/schema.py": "NODE_X = 'data/node_x'\n",  # the other home
    "pkg/stray.py": "import os\n\nLIMIT = 5\n\n\nclass Table:\n    ROWS = (1, 2)\n",
    "pkg/guarded.py": "try:\n    import torch\n    HAS_TORCH = True\nexcept ImportError:\n    HAS_TORCH = False\n"
    "if HAS_TORCH:\n    _DEVICE = 'cpu'\n",
    "pkg/exempt.py": "TRIES = 8\nlower = 1\n",
    "pkg/typed.py": "from typing import TypeVar\nimport typing\n\nT = TypeVar('T')\nKEY = typing.TypeVar('KEY')\n",
}


def constants_rule(tmp_path, exempt):
    for rel, text in TREE.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    read = lambda rel: (tmp_path / rel).read_text(encoding="utf-8")  # noqa: E731
    return [str(p) for p in layout.stray_constants("pkg", exempt, sorted(TREE), read)]


def test_a_constant_outside_the_homes_fails_and_an_exempt_one_passes(tmp_path):
    problems = constants_rule(tmp_path, {"exempt.py": {"TRIES": "knob"}})
    named = sorted(p.split(": ")[2].split()[0] for p in problems)
    # the module-level name, and the names assigned under `try`, `except` and `if`; not the class table,
    # the lower-case name, the two TypeVars, the names in the homes or the exempt one
    assert named == ["HAS_TORCH", "LIMIT", "_DEVICE"]
    assert any(p.startswith("constants: pkg/stray.py:3: LIMIT belongs in constants/") for p in problems)


def test_an_exemption_of_a_name_that_is_gone_fails(tmp_path):
    exempt = {
        "exempt.py": {"TRIES": "knob", "GONE": "knob"},
        "stray.py": {"LIMIT": "knob"},
        "guarded.py": {"HAS_TORCH": "environment", "_DEVICE": "environment"},
        "removed.py": {"OLD": "knob"},
    }
    assert constants_rule(tmp_path, exempt) == [
        "constants: pkg/exempt.py: exempt is GONE, which the file does not assign",
        "constants: pkg/removed.py: exempt is OLD, which the file does not assign",
    ]


def test_the_two_toolkits_pass_the_rule_and_the_analysis_table_lists_only_what_exists():
    assert layout.constants(layout.tree()) == []


SAMPLE = '''"""A module docstring
over two lines."""

import os  # code with a comment after it is code

# a comment alone is not code


def f(x):
    """A docstring is not code."""
    y = (
        x
        + 1
    )
    text = """a string that is a value,
    not a docstring: both lines are code"""
    "a string statement that is not first is code"
    return y, text


class A:
    """Not code."""

    B = 1
'''


def test_code_lines_leaves_out_docstrings_comments_and_blank_lines():
    # import, def, the four lines of y, the two of text, the string statement, return, class, B
    assert count_lines.LineCount.code_lines(SAMPLE) == 12
    assert count_lines.LineCount.code_lines("") == 0
    assert count_lines.LineCount.code_lines('"""Only a docstring."""\n') == 0


def test_a_package_count_separates_the_homes_of_constants(tmp_path):
    for rel, text in {"constants/a.py": "A = 1\nB = 2\n", "schema.py": "C = 3\n", "m.py": SAMPLE}.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert count_lines.LineCount.package(tmp_path) == (3 + SAMPLE.count("\n"), 15, 3)
