"""Write the class and module diagrams of the package from its source, as Mermaid sources under
docs/figures/diagrams/, so the pictures in docs/reference/CLASS_MAP.md never drift from the code.

    python tools/class_diagrams.py            # write classes_<group>.mmd and modules.mmd
    python tools/class_diagrams.py --check    # exit 1 when a committed source is out of date (CI)

Then `python tools/render_mermaid.py docs/figures/diagrams/classes_*.mmd docs/figures/diagrams/modules.mmd`
renders them. Everything comes from the AST: a class's public methods, its bases (inheritance), the
classes of other modules it names (a dashed "uses" edge), and the imports between modules (the
module diagram). Private classes are left out except where the walker's are the point.
"""

from __future__ import annotations

import ast
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "fdia_graph")
OUT = os.path.join(ROOT, "docs", "figures", "diagrams")

# The class diagrams: one per group of modules, the classes of those modules plus any class of another
# group they name (drawn as an outside participant), so every line connects to something on the page.
GROUPS = {
    "dataset": [
        "dataset/base",
        "dataset/graph",
        "dataset/physics",
        "dataset/records",
        "dataset/export",
        "dataset/sequence",
        "dataset/__init__",
    ],
    "engine": ["engine/base", "engine/measurement", "engine/physics", "engine/attacks", "engine/core"],
    "estimation": ["se/base", "se/methods", "se/jacobian"],
    "localization_trust": [
        "localization/base",
        "localization/methods",
        "localization/learned",
        "trust/base",
        "trust/dqn",
    ],
    "models": ["models/base", "models/fields", "models/data"],
    "scores": ["models/base", "models/scores"],
}
# classes drawn as one node: the seven field-group mixins every record bundle inherits, which as
# 28 separate inheritance lines said nothing
COLLAPSE = {
    "models": (
        "FieldGroups",
        [
            "StreamLayers",
            "GraphFields",
            "CleanFields",
            "TemporalFields",
            "RecordIds",
            "LabelFields",
            "ScanFields",
        ],
    ),
}
MAX_METHODS = 8  # methods listed per class before "... and n more"
NOISE_MODULES = {"dataset/__init__"}  # FdiaGraph itself: every class takes one, the edge says nothing
NOISE_CLASSES = {"NodeColumns", "EdgeColumns", "BranchColumns"}  # the column-index tuples, likewise
# collaborators a class takes by duck typing, so no import names them: (other class, edge label)
EXTRA = {
    "GatedPrior": [("LocalizerBase", "gate")],
    "ResidualLocalizer": [("SEBase", "estimator")],
}


def modules() -> dict[str, ast.Module]:
    out = {}
    for dirpath, _, files in os.walk(SRC):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(dirpath, f)
                name = os.path.relpath(p, SRC)[:-3].replace("\\", "/")
                with open(p, encoding="utf8") as fh:
                    out[name] = ast.parse(fh.read())
    return out


def resolve(module: str, node: ast.ImportFrom) -> str | None:
    """The package-relative module an `from .x import y` names, or None for an outside import."""
    if node.level == 0:
        return node.module if node.module and node.module.startswith("fdia_graph") else None
    base = module.split("/")[:-1]
    base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
    target = base + (node.module.split(".") if node.module else [])
    return "/".join(target)


def class_index(mods: dict[str, ast.Module]) -> dict[str, str]:
    """class name -> module, over the whole package (public classes and the walker's private ones)."""
    idx = {}
    for m, t in mods.items():
        for n in t.body:
            if isinstance(n, ast.ClassDef):
                idx[n.name] = m
    return idx


def class_diagram(group: str, mods: dict[str, ast.Module], idx: dict[str, str]) -> str:
    members = GROUPS[group]
    lines = ["classDiagram"]
    seen: set[str] = set()
    inherit, uses = [], []
    collapsed_name, collapsed = COLLAPSE.get(group, (None, []))
    if collapsed_name:
        lines.append(f"    class {collapsed_name} {{")
        lines.append(f"        <<{len(collapsed)} mixins>>")
        lines.extend(f"        {c}" for c in collapsed)
        lines.append("    }")
        seen.add(collapsed_name)
    for m in members:
        t = mods[m]
        imported: dict[str, str] = {}  # local name -> module, for the "uses" edges
        for n in ast.walk(t):  # module level and inside methods, the lazy imports included
            if isinstance(n, ast.ImportFrom):
                tgt = resolve(m, n)
                if tgt and tgt in mods or (tgt and tgt + "/__init__" in mods):
                    for a in n.names:
                        imported[a.asname or a.name] = tgt
        for n in t.body:
            if not isinstance(n, ast.ClassDef) or n.name.startswith("_") or n.name in collapsed:
                continue
            seen.add(n.name)
            methods = [
                x.name
                for x in n.body
                if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and not x.name.startswith("_")
            ]
            props = [
                x.name
                for x in n.body
                if isinstance(x, ast.FunctionDef)
                and any(getattr(d, "id", "") == "property" for d in x.decorator_list)
            ]
            body = []
            for name in methods[:MAX_METHODS]:
                body.append(f"        +{name}()" if name not in props else f"        +{name}")
            if len(methods) > MAX_METHODS:
                body.append(f"        ... and {len(methods) - MAX_METHODS} more")
            lines.append(f"    class {n.name} {{")
            lines.extend(body)
            lines.append("    }")
            for b in n.bases:
                bname = b.id if isinstance(b, ast.Name) else getattr(b, "attr", None)
                if bname in collapsed:
                    bname = collapsed_name
                if bname and (bname in idx or bname == collapsed_name):
                    inherit.append((bname, n.name))
            # names of other modules' classes used inside the class body
            named = {x.id for x in ast.walk(n) if isinstance(x, ast.Name)} | {
                x.attr for x in ast.walk(n) if isinstance(x, ast.Attribute)
            }
            for local, tgt in imported.items():
                if (
                    local in named
                    and local in idx
                    and local != n.name
                    and not local.startswith("_")
                    and idx[local] not in NOISE_MODULES
                    and local not in NOISE_CLASSES
                    and local not in collapsed
                ):
                    uses.append((n.name, local))
            for other, label in EXTRA.get(n.name, []):  # duck-typed collaborators no import names
                uses.append((n.name, other, label))
    outside = {b for b, _ in inherit} | {u[1] for u in uses}
    for name in sorted(outside - seen):
        lines.append(f"    class {name} {{\n        <<{idx.get(name, '?').split('/')[0]}>>\n    }}")
    for b, c in sorted(set(inherit)):
        lines.append(f"    {b} <|-- {c}")
    # one edge per pair of endpoints, a labeled (EXTRA) edge winning over the plain "uses" the import
    # scan may also have found, and a full sort so the source is the same on every run
    labeled: dict[tuple[str, str], str] = {}
    for edge in uses:
        key = (edge[0], edge[1])
        label = edge[2] if len(edge) > 2 else "uses"
        if key not in labeled or label != "uses":
            labeled[key] = label
    for (c, u), label in sorted(labeled.items()):
        if (u, c) not in set(inherit) and u != c:
            lines.append(f"    {c} ..> {u} : {label}")
    return "\n".join(lines) + "\n"


# read by every module, so their edges are said once in the caption instead of drawn; and the public
# API imports everything, which is not a dependency worth a line each
FOUNDATION = {"models", "formulas", "schema"}
OMIT = {"fdia_graph"}


def module_diagram(mods: dict[str, ast.Module]) -> str:
    """Top-level modules and packages of fdia_graph, one node each, an edge per import between them
    (the foundation modules and the public API's fan-out left out, see FOUNDATION and OMIT)."""

    def top(m: str) -> str:
        head = m.split("/")[0]
        return head if head != "__init__" else "fdia_graph"

    edges: dict[tuple[str, str], int] = defaultdict(int)
    nodes = set()
    for m, t in mods.items():
        src = top(m)
        nodes.add(src)
        for n in t.body:
            if isinstance(n, ast.ImportFrom):
                tgt = resolve(m, n)
                if tgt and (tgt in mods or tgt + "/__init__" in mods):
                    dst = top(tgt)
                    if dst != src:
                        edges[(src, dst)] += 1
            # imports inside functions count too: they are the lazy edges (torch, the engine behind generate)
            for f in ast.walk(n):
                if isinstance(f, ast.ImportFrom) and f is not n:
                    tgt = resolve(m, f)
                    if tgt and (tgt in mods or tgt + "/__init__" in mods):
                        dst = top(tgt)
                        if dst != src:
                            edges[(src, dst)] += 1
    label = {
        "fdia_graph": "__init__.py<br/>fg.* public API",
        "dataset": "dataset/",
        "engine": "engine/",
        "se": "se/",
        "localization": "localization/",
        "trust": "trust/",
        "models": "models/",
        "formulas": "formulas/",
        "registry": "registry.py",
        "download": "download.py",
        "schema": "schema.py",
        "timeline": "timeline.py",
        "generation": "generation.py",
        "profiles": "profiles.py",
        "streams": "streams.py<br/>(deprecated)",
        "torch_data": "torch_data.py<br/>(deprecated)",
    }
    drawn = {(a, b) for (a, b) in edges if a not in OMIT | FOUNDATION and b not in FOUNDATION}
    lines = ["flowchart TB"]
    for n in sorted(nodes - OMIT - FOUNDATION):
        lines.append(f'    {n}["{label.get(n, n)}"]')
    for a, b in sorted(drawn):
        lines.append(f"    {a} --> {b}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    mods = modules()
    idx = class_index(mods)
    wanted = {f"classes_{g}.mmd": class_diagram(g, mods, idx) for g in GROUPS}
    wanted["modules.mmd"] = module_diagram(mods)
    stale = []
    for name, text in wanted.items():
        path = os.path.join(OUT, name)
        current = open(path, encoding="utf8").read() if os.path.exists(path) else None
        if current != text:
            stale.append(name)
            if "--check" not in argv:
                with open(path, "w", encoding="utf8", newline="\n") as fh:
                    fh.write(text)
    if "--check" in argv:
        if stale:
            print("out of date: " + ", ".join(stale) + "  (run tools/class_diagrams.py and re-render)")
            return 1
        print("class and module diagram sources match the code")
        return 0
    print("wrote " + ", ".join(stale) if stale else "sources unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
