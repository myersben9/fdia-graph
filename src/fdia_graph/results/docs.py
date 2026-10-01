"""Measured numbers in markdown come from the store, never from a keyboard.

A document holds a results block where a number or a table belongs:

    The proposed estimator's angle error is <!-- results: value experiment=se.estimators
    system=ieee14 method=prior+huber family=geo metric=angle_mae_deg --><!-- /results --> degrees.

    <!-- results: se.estimators.angle -->
    <!-- /results -->

`fill` replaces each block's body with what its query renders now; `stale` lists the blocks whose
body differs, which `tools/results_docs.py --check` turns into a failing gate. A query is a function
registered under a name with `@query("name")`: it gets the store and the block's arguments
(strings: the bare words positionally, the `key=value` ones by name) and returns markdown. Three
are built in: `value` (one cell), `table` (a pivot of one metric by two keys) and `reduction` (the
percent error reduction of one method over another).
A block whose body starts on a new line renders a block of markdown, one inline renders inline.
"""

from __future__ import annotations

import functools
import re
import shlex
from collections.abc import Callable, Mapping
from typing import Optional

from ..models.errors import NoSuchResult
from . import render
from .store import Store

Query = Callable[..., str]
QUERIES: dict[str, Query] = {}

BLOCK = re.compile(r"<!-- results: (?P<spec>.*?) -->(?P<body>.*?)<!-- /results -->", re.DOTALL)
_FORMAT = ("fmt", "scale", "sd")
_OPEN, _CLOSE = "<!-- results:", "<!-- /results -->"
_INLINE_CODE = re.compile(r"`[^`\n]*`")
# fenced code (``` or ~~~, indented under a list too), kept as one split part up to its closing fence
_FENCE = re.compile(r"(^[ \t]*(?P<f>`{3,}|~{3,}).*?^[ \t]*(?P=f)[ \t]*$)", re.DOTALL | re.MULTILINE)


def query(name: str) -> Callable[[Query], Query]:
    """Register a function as the query `name` of results blocks."""
    return functools.partial(_register, name)


def _register(name: str, fn: Query) -> Query:
    QUERIES[name] = fn
    return fn


def parse(spec: str) -> tuple[str, list[str], dict[str, str]]:
    """A block's spec as (query name, its positional arguments, its `key=value` arguments)."""
    name, *args = shlex.split(spec)
    return name, [a for a in args if "=" not in a], dict(a.split("=", 1) for a in args if "=" in a)


def render_spec(spec: str, store: Store) -> str:
    """What the block `spec` renders from `store` now."""
    name, pos, args = parse(spec)
    return QUERIES[name](store, *pos, **args)


def fill(text: str, store: Store) -> str:
    """`text` with every results block's body rendered afresh; a block inside a fenced code block is
    an example, not a block, and stays as written."""
    sub = functools.partial(_rendered, store=store)
    return "".join(part if code else BLOCK.sub(sub, part) for part, code in _parts(text))


def _parts(text: str) -> list[tuple[str, bool]]:
    """`text` split into (part, is fenced code) in order."""
    out, at = [], 0
    for m in _FENCE.finditer(text):
        out += [(text[at : m.start()], False), (m.group(0), True)]
        at = m.end()
    return [*out, (text[at:], False)]


def _rendered(m: re.Match[str], store: Store) -> str:
    """One block with its body rendered now; a block on its own lines keeps them."""
    body = render_spec(m.group("spec"), store)
    if m.group("body").startswith("\n"):
        body = "\n" + body + "\n"
    return f"<!-- results: {m.group('spec')} -->{body}<!-- /results -->"


def stale(text: str, store: Store) -> list[str]:
    """The specs of the blocks of `text` whose body is not what they render now."""
    live = "".join(part for part, code in _parts(text) if not code)
    blocks = list(BLOCK.finditer(live))
    prose = _INLINE_CODE.sub("", BLOCK.sub("", live))  # markers quoted in `code` are not markers
    opened, closed = prose.count(_OPEN) + len(blocks), prose.count(_CLOSE) + len(blocks)
    if not opened == closed == len(blocks):  # an unterminated or stray marker
        return [f"malformed results markers: {opened} opened, {closed} closed, {len(blocks)} blocks"]
    pairs = zip(blocks, BLOCK.finditer(fill(live, store)))  # each block against itself, by position
    return [old.group("spec") for old, new in pairs if old.group("body") != new.group("body")]


# ---- built-in queries
def _filters(args: dict[str, str]) -> dict[str, object]:
    """The block arguments that select records: the keys and tags, a comma meaning "any of"."""
    skip = set(_FORMAT) | {"experiment", "rows", "cols", "base", "new", "order"}
    return {k: v.split(",") if "," in v else v for k, v in args.items() if k not in skip}


def _fmt(args: dict[str, str]) -> tuple[Optional[str], float, bool]:
    return args.get("fmt"), float(args.get("scale", "1")), args.get("sd", "") in ("1", "true", "yes")


@query("value")
def _value(store: Store, experiment: str, **args: str) -> str:
    """One number: the single newest record the arguments select."""
    spec, scale, sd = _fmt(args)
    return render.cell(store.one(experiment, **_filters(args)), spec, scale, sd)


@query("table")
def _table(store: Store, experiment: str, rows: str, cols: str, **args: str) -> str:
    """A pivot of the selected records: one line per `rows` key, one column per `cols` key, in the
    order given by `order` ("a,b,c" for the rows) or the store's order."""
    spec, scale, sd = _fmt(args)
    grid = render.pivot(store.latest(experiment, **_filters(args)), rows, cols)
    row_keys = _row_keys(grid, args.get("order"), f"table {experiment} {args}")
    col_keys = list(dict.fromkeys(c for _, c in grid))
    body = [[r, *(render.cell(grid.get((r, c)), spec, scale, sd) for c in col_keys)] for r in row_keys]
    return render.table([rows, *col_keys], body)


def _row_keys(grid: Mapping[tuple[str, str], object], order: Optional[str], what: str) -> list[str]:
    """The rows of a pivot: `order`'s ("a,b,c") or the store's; an empty grid or an ordered row with
    no record is `NoSuchResult`, so a typo in a block never renders an empty table."""
    have = list(dict.fromkeys(r for r, _ in grid))
    keys = order.split(",") if order else have
    missing = [r for r in keys if r not in have]
    if not grid or missing:
        raise NoSuchResult(f"{what}: {'no records' if not grid else f'no rows {missing}'}")
    return keys


@query("reduction")
def _reduction(store: Store, experiment: str, base: str, new: str, **args: str) -> str:
    """The percent error reduction of method `new` over method `base` (negative when it is worse),
    a whole number by default."""
    keys = _filters(args)
    b = store.one(experiment, **{**keys, "method": base}).value
    n = store.one(experiment, **{**keys, "method": new}).value
    return render.number(100.0 * (1.0 - n / b), args.get("fmt", ".0f"))
