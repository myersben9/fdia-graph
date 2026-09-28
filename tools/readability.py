"""Readability measures for src/fdia_graph, the limits from docs/plans/READABILITY_PLAN.md rule 1,
and the file-protocol rule: a dataset path ("data/...", "graph/...", any group of `schema.Group`)
or a group name used as one (`f.create_group("data")`, `f["attack"]`, `"episodes" in f`) may be
spelled only in src/fdia_graph/schema.py; every other module goes through `schema`. And the
validation rule (docs/plans/VALIDATION_PLAN.md): `raise ValueError` / `raise TypeError` and
`isinstance(...)` only inside `fdia_graph.models`, where the one engine checks every input and the
parser models read loose input by its type; everywhere else an input is checked or parsed by
building its model, and a condition only the data reveals raises a named error from `errors`. And
the typing rule: an annotation says `Any` only where ANY_ALLOWED lists it with the reason the value
can be anything; every other annotation carries its real type.

    python tools/readability.py --report                 # every function outside a limit, whole package
    python tools/readability.py --check --base origin/main   # gate: functions touched since base must pass

A function is measured on five things: cyclomatic complexity (radon), nesting depth of loops and
branches, nested functions that read variables of the enclosing function, parameter count on
private functions, and positional record indexing (a bare name indexed by two or more distinct
integer constants of 4 or more, the `r[10]`, `r[11]` pattern). The sixth rule of the plan, one copy of every formula, is a review
rule and is not measured here.

Exceptions: a function listed in EXCEPTIONS with a one-line reason passes the gate. Adding one
is a reviewed decision; the reason is the whole point.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "fdia_graph")

LIMITS = {"complexity": 10, "nesting": 3, "captures": 0, "params": 7, "positional": 0}
_GROUPS = ("data", "benign", "clean", "graph", "episodes", "attack")  # schema.Group, kept in step by a test
# a lone group name is a protocol literal when it is created as a group, or when it is subscripted or
# tested with `in`; "benign" and "clean" are also record fields, so only the four that are never
# fields are checked that way
_GROUPS_NEVER_FIELDS = ("data", "graph", "episodes", "attack")
_SCHEMA = os.path.join(ROOT, "schema.py")  # the one module allowed to spell a dataset path

# "module.qualname": reason. Keep every entry justified; the report still lists them, marked.
EXCEPTIONS: dict[str, str] = {}

_NEST = (ast.For, ast.While, ast.If, ast.With, ast.Try)
_FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass
class Measure:
    file: str
    name: str
    line: int
    end: int
    complexity: int
    nesting: int
    captures: int
    params: int
    positional: int
    private: bool

    def failures(self) -> list[str]:
        out = []
        if self.complexity > LIMITS["complexity"]:
            out.append(f"complexity {self.complexity}")
        if self.nesting > LIMITS["nesting"]:
            out.append(f"nesting {self.nesting}")
        if self.captures > LIMITS["captures"]:
            out.append(f"closure reads {self.captures} outer variables")
        if self.private and self.params > LIMITS["params"]:
            out.append(f"{self.params} parameters")
        if self.positional > LIMITS["positional"]:
            out.append(f"{self.positional} positional record index(es)")
        return out

    @property
    def key(self) -> str:
        mod = os.path.splitext(self.file.replace(os.sep, "."))[0]
        return f"{mod}.{self.name}"


def _nesting(node: ast.AST) -> int:
    best = 0

    def walk(n: ast.AST, d: int) -> None:
        nonlocal best
        best = max(best, d)
        for c in ast.iter_child_nodes(n):
            if isinstance(c, _FUNC + (ast.Lambda,)):
                continue
            walk(c, d + 1 if isinstance(c, _NEST) else d)

    walk(node, 0)
    return best


def _captures(inner: ast.FunctionDef, outer_names: set[str]) -> int:
    params = {a.arg for a in inner.args.args + inner.args.kwonlyargs}
    stored = {n.id for n in ast.walk(inner) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    loaded = {n.id for n in ast.walk(inner) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return len((loaded - params - stored) & outer_names - {"self", "cls"})


def _positional(fn: ast.AST) -> int:
    """Sites where a bare name is indexed by an integer constant of 4 or more, counted only for
    names indexed at two or more distinct such positions: that is a record tuple read by position
    (r[10], r[11]); a single site is usually a dict keyed by an id."""
    sites: dict[str, set[int]] = {}
    for s in ast.walk(fn):
        if (
            isinstance(s, ast.Subscript)
            and isinstance(s.value, ast.Name)
            and isinstance(s.slice, ast.Constant)
            and isinstance(s.slice.value, int)
            and s.slice.value >= 4
        ):
            sites.setdefault(s.value.id, set()).add(s.slice.value)
    return sum(len(v) for v in sites.values() if len(v) >= 2)


def _complexities(path: str) -> dict[tuple[str, int], int]:
    from radon.complexity import cc_visit

    out = {}
    for block in cc_visit(open(path, encoding="utf8").read()):
        out[(block.name, block.lineno)] = block.complexity
        for m in getattr(block, "methods", []):
            out[(m.name, m.lineno)] = m.complexity
    return out


def _functions_with_qualnames(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """Every function in the module with its qualified name (Class.method, outer.inner), so two
    methods called __init__ never share a key."""
    found: list[tuple[str, ast.AST]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _FUNC):
                found.append((prefix + child.name, child))
                walk(child, prefix + child.name + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
            else:
                walk(child, prefix)

    walk(tree, "")
    return found


def _measure_function(rel: str, qualname: str, node: ast.AST, cc: dict[tuple[str, int], int]) -> Measure:
    outer = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    outer |= {a.arg for a in node.args.args + node.args.kwonlyargs}
    caps = 0
    for inner in ast.walk(node):
        if isinstance(inner, ast.FunctionDef) and inner is not node:
            caps = max(caps, _captures(inner, outer))
    params = len(node.args.args) + len(node.args.kwonlyargs)
    if params and node.args.args and node.args.args[0].arg in ("self", "cls"):
        params -= 1
    return Measure(
        file=rel,
        name=qualname,
        line=node.lineno,
        end=node.end_lineno or node.lineno,
        complexity=cc.get((node.name, node.lineno), 1),
        nesting=_nesting(node),
        captures=caps,
        params=params,
        positional=_positional(node),
        private=node.name.startswith("_") and not node.name.startswith("__"),  # dunders are public API
    )


def _rel(path: str) -> str:
    """The path relative to the repository, or as given when it lies on another drive (Windows
    cannot express that relatively, and a CI runner's temp dir sits on D:)."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def protocol_literals(path: str) -> list[tuple[str, int, str]]:
    """Path-shaped string literals ("<group>/..." or a lone group name used as a path prefix in an
    f-string) outside the schema module: (file, line, literal). Docstrings are not literals."""
    rel = _rel(path)
    if os.path.normcase(os.path.normpath(path)) == os.path.normcase(os.path.normpath(_SCHEMA)):
        return []
    tree = ast.parse(open(path, encoding="utf8").read())
    docs = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.ClassDef, *_FUNC))
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
    }
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            head = n.value.split("/", 1)[0]
            if "/" in n.value and head in _GROUPS:
                out.append((rel, n.lineno, n.value))
        lit = _group_used_as_group(n)
        if lit is not None:
            out.append((rel, n.lineno, lit))
    return sorted(out, key=lambda t: t[1])  # ast.walk is breadth-first; report in line order


def _group_used_as_group(n: ast.AST) -> Optional[str]:
    """The group name when `n` creates, subscripts or tests membership of a group by literal."""
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "create_group":
        args = list(n.args[:1]) + [kw.value for kw in n.keywords if kw.arg == "name"]
        for arg in args:  # positional or `name=`
            if isinstance(arg, ast.Constant) and arg.value in _GROUPS:
                return str(arg.value)
    if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant):
        if n.slice.value in _GROUPS_NEVER_FIELDS:
            return str(n.slice.value)
    if (
        isinstance(n, ast.Compare)
        and isinstance(n.left, ast.Constant)
        and n.left.value in _GROUPS_NEVER_FIELDS
    ):
        if any(isinstance(op, (ast.In, ast.NotIn)) for op in n.ops):
            return str(n.left.value)
    return None


_MODELS = os.path.join(ROOT, "models")  # the one package allowed to raise a bare ValueError / TypeError
_BARE = {"ValueError", "TypeError"}


def hand_checks(path: str) -> list[tuple[str, int, str]]:
    """`raise ValueError(...)` / `raise TypeError(...)` / `isinstance(...)` outside `fdia_graph.models`:
    (file, line, "ValueError" | "TypeError" | "isinstance")."""
    here, models = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(_MODELS))
    if os.path.commonpath([here, models]) == models:  # inside the package, not a sibling named models_*
        return []
    tree = ast.parse(open(path, encoding="utf8").read())
    raises = [
        (_rel(path), n.lineno, name)
        for n in ast.walk(tree)
        if isinstance(n, ast.Raise) and (name := _raised_name(n)) in _BARE
    ]
    types = [
        (_rel(path), n.lineno, "isinstance")
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "isinstance"
    ]
    return sorted(raises + types, key=lambda h: h[1])


# Where `Any` stays, and why: the value there genuinely can be anything. Keyed by
# "file.py:qualname" plus ":param" or ":return"; a field is "file.py:Class.field". Every other
# annotation carries its real type (an optional dependency's type under `if TYPE_CHECKING:`).
_RAW = "the validation engine's raw input: a rule runs on whatever the caller passed, and a failure becomes a ConfigError"
_KW = "keyword arguments forwarded unchanged to another signature, each its own type"
_STAGE = "a record's staging dict: tensors and scalar provenance, each key its own type, unpacked into RecordBundle"
_BUNDLE = "a bundle's fields as a dict: arrays, tensors, scalars and lists, each key its own type"
ANY_ALLOWED: dict[str, str] = {
    "__init__.py:__getattr__:return": "a lazily imported public name: any of the package's functions or classes",
    "generation.py:generate:**knobs": _KW,
    "streams.py:generate_stream:**knobs": _KW,
    "streams.py:load_stream.out": "a stream file's contents: arrays and the pickled episode list, each key its own type",
    "timeline.py:_timeline_attrs:knobs": "the recorded generation knobs, each key its own type, written as file attributes",
    "dataset/records.py:RecordsMixin._add_benign:item": _STAGE,
    "dataset/records.py:RecordsMixin._base_item:return": _STAGE,
    "dataset/records.py:RecordsMixin._add_optional_layers:item": _STAGE,
    "dataset/records.py:RecordsMixin.loader:**kw": _KW,
    "federated/localizer.py:FederatedLocalizer.__init__:**kw": _KW,
    "federated/se.py:RegionalPrior.__init__:**kw": _KW,
    "localization/base.py:LocalizerBase.score.out": "the score table's rows: overall, benign and per-family metrics, one row type per key",
    "localization/learned.py:BusCNN.__init__:**kw": _KW,
    "se/methods.py:GatedPrior.__init__:**kw": _KW,
    "models/assets.py:AssetSpec.meta": "a local dataset's JSON metadata, as fg.generate wrote it",
    "models/base.py:Bundle.to_dict:return": _BUNDLE,
    "models/base.py:Bundle.ordered:mapping": _BUNDLE,
    "models/base.py:_rebuild:field_values": _BUNDLE,
    "models/validation.py:Validated.__dataclass_fields__": "the dataclasses protocol's own field map, Field[Any]",
    "models/validation.py:Rule.holds:value": _RAW,
    "models/validation.py:Rule.apply:value": _RAW,
    "models/validation.py:Rule.apply:return": "what a rule stores: the value itself, or its conversion (a string, an array, a parse)",
    "models/validation.py:OneOf.apply:value": _RAW,
    "models/validation.py:Positive.holds:value": _RAW,
    "models/validation.py:Finite.holds:value": _RAW,
    "models/validation.py:Integer.holds:value": _RAW,
    "models/validation.py:AtLeast.holds:value": _RAW,
    "models/validation.py:InRange.holds:value": _RAW,
    "models/validation.py:Required.holds:value": _RAW,
    "models/validation.py:AsArray.apply:value": _RAW,
    "models/validation.py:AsTuple.apply:value": _RAW,
    "models/validation.py:Dims.holds:value": _RAW,
    "models/validation.py:IntegerDtype.holds:value": _RAW,
    "models/validation.py:NonEmpty.holds:value": _RAW,
    "models/validation.py:AllFinite.holds:value": _RAW,
    "models/validation.py:AllPositive.holds:value": _RAW,
    "models/validation.py:Parses.apply:value": _RAW,
    "models/validation.py:_checked:value": _RAW,
    "models/validation.py:_checked:return": "a field's value after its rules: whatever the field's type is",
}


def _mentions_any(node: ast.AST) -> bool:
    return any(
        (isinstance(n, ast.Name) and n.id == "Any") or (isinstance(n, ast.Attribute) and n.attr == "Any")
        for n in ast.walk(node)
    )


def _annotated(tree: ast.AST) -> list[tuple[str, int, ast.AST]]:
    """(qualified place, line, annotation) for every annotation in a module: each parameter
    ("f:x", "f:*args", "f:**kw"), each return ("f:return"), each annotated name ("C.field",
    "f.var"), and each module-level type alias whose value names a type ("Alias")."""
    out: list[tuple[str, int, ast.AST]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _FUNC):
                q = prefix + child.name
                a = child.args
                for arg in a.posonlyargs + a.args + a.kwonlyargs:
                    if arg.annotation is not None:
                        out.append((f"{q}:{arg.arg}", arg.lineno, arg.annotation))
                for star, arg in (("*", a.vararg), ("**", a.kwarg)):
                    if arg is not None and arg.annotation is not None:
                        out.append((f"{q}:{star}{arg.arg}", arg.lineno, arg.annotation))
                if child.returns is not None:
                    out.append((f"{q}:return", child.lineno, child.returns))
                walk(child, q + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                out.append((prefix + child.target.id, child.lineno, child.annotation))
            elif (
                isinstance(child, ast.Assign)
                and not prefix
                and isinstance(child.value, ast.Subscript)
                and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Name)
            ):
                out.append((child.targets[0].id, child.lineno, child.value))
            else:
                walk(child, prefix)

    walk(tree, "")
    return out


def any_annotations(path: str) -> list[tuple[str, int]]:
    """(key, line) of every annotation in the file that uses `Any`; the key is what ANY_ALLOWED lists."""
    tree = ast.parse(open(path, encoding="utf8").read())
    rel = _rel(path).replace(os.sep, "/")
    return [(f"{rel}:{place}", line) for place, line, ann in _annotated(tree) if _mentions_any(ann)]


def any_annotations_all() -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for dp, _, fs in os.walk(ROOT):
        for f in sorted(fs):
            if f.endswith(".py"):
                out += any_annotations(os.path.join(dp, f))
    return out


def any_unlisted() -> list[tuple[str, int]]:
    """`Any` where ANY_ALLOWED does not list it."""
    return [(k, line) for k, line in any_annotations_all() if k not in ANY_ALLOWED]


def any_stale() -> list[str]:
    """ANY_ALLOWED entries that name no `Any` any more: a typed place, or one renamed or removed."""
    found = {k for k, _ in any_annotations_all()}
    return sorted(k for k in ANY_ALLOWED if k not in found)


def _what(name: str) -> str:
    return f"raise {name}" if name in _BARE else f"{name}(...)"


def _raised_name(n: ast.Raise) -> str:
    exc = n.exc.func if isinstance(n.exc, ast.Call) else n.exc
    return exc.id if isinstance(exc, ast.Name) else ""


def hand_checks_all() -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for dp, _, fs in os.walk(ROOT):
        for f in sorted(fs):
            if f.endswith(".py"):
                out += hand_checks(os.path.join(dp, f))
    return out


def protocol_literals_all() -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for dp, _, fs in os.walk(ROOT):
        for f in sorted(fs):
            if f.endswith(".py"):
                out += protocol_literals(os.path.join(dp, f))
    return out


def measure_file(path: str) -> list[Measure]:
    rel = _rel(path)
    tree = ast.parse(open(path, encoding="utf8").read())
    cc = _complexities(path)
    return [_measure_function(rel, qualname, node, cc) for qualname, node in _functions_with_qualnames(tree)]


def measure_all() -> list[Measure]:
    out: list[Measure] = []
    for dp, _, fs in os.walk(ROOT):
        for f in sorted(fs):
            if f.endswith(".py"):
                out += measure_file(os.path.join(dp, f))
    return out


def report(ms: list[Measure]) -> int:
    bad = [m for m in ms if m.failures()]
    print(
        f"{len(ms)} functions measured, {len(bad)} outside a limit "
        f"(limits: {', '.join(f'{k} {v}' for k, v in LIMITS.items())})"
    )
    for m in sorted(bad, key=lambda m: (m.file, m.line)):
        mark = "  [excepted: " + EXCEPTIONS[m.key] + "]" if m.key in EXCEPTIONS else ""
        print(f"  {m.file}:{m.line} {m.name}: " + "; ".join(m.failures()) + mark)
    lits = protocol_literals_all()
    print(f"{len(lits)} dataset-path literals outside {os.path.relpath(_SCHEMA, ROOT)}")
    for rel, line, lit in lits:
        print(f"  {rel}:{line} {lit!r}")
    hands = hand_checks_all()
    print(
        f"{len(hands)} hand-written ValueError/TypeError raises or isinstance calls outside fdia_graph.models"
    )
    for rel, line, name in hands:
        print(f"  {rel}:{line} {_what(name)}")
    unlisted, stale = any_unlisted(), any_stale()
    print(f"{len(unlisted)} Any annotations not in ANY_ALLOWED, {len(stale)} stale ANY_ALLOWED entries")
    for key, line in unlisted:
        print(f"  line {line}: {key}")
    for key in stale:
        print(f"  stale: {key}")
    return 0


def _changed_lines(base: str) -> dict[str, set[int]]:
    """Lines added or modified since `base`, per file under src/fdia_graph, from `git diff -U0`."""
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "-U0", f"{base}...HEAD", "--", "src/fdia_graph"],
        capture_output=True,
        text=True,
        encoding="utf-8",  # docstrings carry equation symbols; the Windows default codepage cannot decode them
        errors="replace",
        check=True,
        cwd=top,
    ).stdout
    out: dict[str, set[int]] = {}
    cur: Optional[str] = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            cur = os.path.normpath(os.path.join(top, line[6:]))
            out.setdefault(cur, set())
        elif line.startswith("@@") and cur is not None:
            new = line.split("+")[1].split(" ")[0]
            start, _, count = new.partition(",")
            n = int(count) if count else 1
            out[cur].update(range(int(start), int(start) + max(n, 1)))
    return out


def check(base: str) -> int:
    changed = _changed_lines(base)
    failing = []
    for path, lines in changed.items():
        if not os.path.exists(path) or not path.endswith(".py"):
            continue
        for m in measure_file(path):
            if any(m.line <= ln <= m.end for ln in lines) and m.failures() and m.key not in EXCEPTIONS:
                failing.append(m)
    if failing:
        print("functions touched by this change that are outside a readability limit:")
        for m in failing:
            print(f"  {m.file}:{m.line} {m.name}: " + "; ".join(m.failures()))
        print("split the function, or add it to EXCEPTIONS in tools/readability.py with a reason.")
        return 1
    lits = [
        (rel, line, lit)
        for path, lines in changed.items()
        if path.endswith(".py") and os.path.exists(path)
        for rel, line, lit in protocol_literals(path)
        if line in lines
    ]
    if lits:
        print("dataset-path literals added by this change (spell the path through fdia_graph.schema):")
        for rel, line, lit in lits:
            print(f"  {rel}:{line} {lit!r}")
        return 1
    hands = [
        (rel, line, name)
        for path, lines in changed.items()
        if path.endswith(".py") and os.path.exists(path)
        for rel, line, name in hand_checks(path)
        if line in lines
    ]
    if hands:
        print("hand-written input checks added by this change (declare the rule or the parse on a model")
        print("in fdia_graph.models, or raise a named error from fdia_graph.errors):")
        for rel, line, name in hands:
            print(f"  {rel}:{line} {_what(name)}")
        return 1
    unlisted, stale = any_unlisted(), any_stale()
    if unlisted or stale:
        print("Any annotations: give each its real type (an optional dependency's type under")
        print("`if TYPE_CHECKING:`), or list it in ANY_ALLOWED in tools/readability.py with the reason:")
        for key, line in unlisted:
            print(f"  line {line}: {key}")
        for key in stale:
            print(f"  stale ANY_ALLOWED entry (no Any there now): {key}")
        return 1
    n_lines = sum(len(v) for v in changed.values())
    print(f"readability gate: {n_lines} changed lines in {len(changed)} file(s), all touched functions pass")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--base", default="origin/main")
    a = ap.parse_args()
    rc = 0
    if a.report or not a.check:
        rc |= report(measure_all())
    if a.check:
        rc |= check(a.base)
    sys.exit(rc)
