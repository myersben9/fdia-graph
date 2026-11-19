"""Write the class and module diagrams of the two toolkits (`fdia_graph` and `fdia_analysis`) from
their source, as Mermaid sources under docs/figures/diagrams/, so the pictures in
docs/reference/CLASS_MAP.md never drift from the code.

    python tools/class_diagrams.py            # write classes_<group>.mmd and modules.mmd
    python tools/class_diagrams.py --check    # exit 1 when a committed source is out of date (CI)

Then `python tools/render_mermaid.py docs/figures/diagrams/classes_*.mmd docs/figures/diagrams/modules.mmd`
renders them. Everything comes from the AST: a class's public methods, its bases (inheritance), the
classes of other modules it names (a dashed "uses" edge), and the imports between modules (the
module diagram). Private classes are left out except where the walker's are the point. The module
diagram is one picture with a box per toolkit, so the one direction of the dependency between them
(analysis reads generation, never the reverse) is what the reader sees.
"""

from __future__ import annotations

import ast
import os
import sys
from collections import defaultdict

from readability import DEFERRED  # tools/ is the script's own directory, so on sys.path
from sources import ANALYSIS, GENERATION, PACKAGES, ROOT, package_dir

OUT = os.path.join(ROOT, "docs", "figures", "diagrams")

# The class diagrams: one per group of modules, the classes of those modules plus any class of another
# group they name (drawn as an outside participant), so every line connects to something on the page.
# A module is named by its package and its path in it.
GROUPS = {
    "dataset": [
        f"{GENERATION}/dataset/base",
        f"{GENERATION}/dataset/graph",
        f"{GENERATION}/dataset/physics",
        f"{GENERATION}/dataset/records",
        f"{GENERATION}/dataset/export",
        f"{GENERATION}/dataset/sequence",
        f"{GENERATION}/dataset/__init__",
    ],
    "engine": [
        f"{GENERATION}/engine/base",
        f"{GENERATION}/engine/measurement",
        f"{GENERATION}/engine/physics",
        f"{GENERATION}/engine/attacks/__init__",
        f"{GENERATION}/engine/core",
    ],
    "attacks": [
        f"{GENERATION}/engine/attacks/area",
        f"{GENERATION}/engine/attacks/false_state",
        f"{GENERATION}/engine/attacks/minimize",
        f"{GENERATION}/engine/attacks/overload",
        f"{GENERATION}/engine/attacks/ramp_frames",
        f"{GENERATION}/engine/attacks/episodes",
        f"{GENERATION}/engine/attacks/__init__",
    ],
    "estimation": [f"{ANALYSIS}/se/base", f"{ANALYSIS}/se/methods", f"{ANALYSIS}/se/jacobian"],
    "localization_trust": [
        f"{ANALYSIS}/localization/base",
        f"{ANALYSIS}/localization/methods",
        f"{ANALYSIS}/localization/learned",
        f"{ANALYSIS}/trust/base",
        f"{ANALYSIS}/trust/meter_dqn",
    ],
    "federated": [
        f"{ANALYSIS}/federated/partition",
        f"{ANALYSIS}/federated/aggregate",
        f"{ANALYSIS}/federated/localizer",
        f"{ANALYSIS}/federated/se",
    ],
    "models": [f"{GENERATION}/models/base", f"{GENERATION}/models/fields", f"{GENERATION}/models/data"],
    "scores": [f"{GENERATION}/models/base", f"{ANALYSIS}/models/scores", f"{ANALYSIS}/models/data"],
}
MAX_METHODS = 8  # methods listed per class before "... and n more"
# FdiaGraph itself: every class takes one, the edge says nothing
NOISE_MODULES = {f"{GENERATION}/dataset/__init__"}
NOISE_CLASSES = {"NodeChannel", "EdgeChannel", "CurrentChannel"}  # the column enums, likewise
# collaborators a class takes by duck typing, so no import names them: (other class, edge label)
EXTRA = {
    "GatedPrior": [("LocalizerBase", "gate")],
    "ResidualLocalizer": [("SEBase", "estimator")],
}


def modules() -> dict[str, ast.Module]:
    """Every module of the two packages, by `<package>/<path in it>` (`fdia_analysis/se/base`)."""
    out = {}
    for package in PACKAGES:
        src = package_dir(package)
        for dirpath, dirs, files in os.walk(src):
            # a subpackage not yet under the SDK's rules (DEFERRED in tools/readability.py) keeps its own
            # architecture page, and its class names would shadow the package's in these diagrams
            dirs[:] = [d for d in dirs if not (dirpath == src and f"{package}/{d}" in DEFERRED)]
            for f in files:
                if f.endswith(".py"):
                    p = os.path.join(dirpath, f)
                    name = os.path.relpath(p, src)[:-3].replace("\\", "/")
                    with open(p, encoding="utf8") as fh:
                        out[f"{package}/{name}"] = ast.parse(fh.read())
    return out


def resolve(module: str, node: ast.ImportFrom) -> str | None:
    """The module (`<package>/<path>`) a `from .x import y` or a `from fdia_graph.x import y` names, or
    None for an outside import."""
    if node.level == 0:
        parts = (node.module or "").split(".")
        return "/".join(parts) if parts[0] in PACKAGES and len(parts) > 1 else None
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
            if not isinstance(n, ast.ClassDef) or n.name.startswith("_"):
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
                if bname and bname in idx:
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
                ):
                    uses.append((n.name, local))
            for other, label in EXTRA.get(n.name, []):  # duck-typed collaborators no import names
                uses.append((n.name, other, label))
    outside = {b for b, _ in inherit} | {u[1] for u in uses}
    for name in sorted(outside - seen):
        # the package it comes from, under its toolkit: `fdia_analysis/se/base` reads as "se"
        home = idx[name].split("/")[1] if name in idx else "?"
        lines.append(f"    class {name} {{\n        <<{home}>>\n    }}")
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
# API imports everything, which is not a dependency worth a line each.
FOUNDATION = {"models", "formulas", "constants", "schema", "errors"}
OMIT = set(PACKAGES)  # each package's own `__init__.py`
# the caption of each toolkit's box, and the node label of each top-level module or package
TOOLKIT = {
    GENERATION: "fdia-graph, the generation toolkit: src/fdia_graph",
    ANALYSIS: "fdia-analysis, the analysis toolkit: analysis/src/fdia_analysis",
}
LABEL = {
    GENERATION: "__init__.py<br/>fg.* public API",
    "dataset": "dataset/",
    "data": "data/",
    "systems": "systems/",
    "sensing": "sensing/",
    "engine": "engine/",
    "generation": "generation/",
    "ratings": "ratings/",
    "registry": "registry.py",
    "download": "download.py",
    "timeline": "timeline.py",
    "se": "se/",
    "localization": "localization/",
    "federated": "federated/",
    "trust": "trust/",
    "results": "results/",
}


def module_diagram(mods: dict[str, ast.Module]) -> str:
    """Top-level modules and packages of the two toolkits, one node each in its toolkit's box, an edge
    per import between them (the foundation modules and each package's own `__init__.py` left out, see
    FOUNDATION and OMIT). No top-level name is in both toolkits but the foundation's, so a node is its
    name."""

    def top(m: str) -> str:
        parts = m.split("/")
        return parts[1] if len(parts) > 1 and parts[1] != "__init__" else parts[0]

    edges: dict[tuple[str, str], int] = defaultdict(int)
    nodes = set()
    home: dict[str, str] = {}
    for m, t in mods.items():
        src = top(m)
        nodes.add(src)
        home[src] = m.split("/")[0]
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
    drawn = {(a, b) for (a, b) in edges if a not in OMIT | FOUNDATION and b not in OMIT | FOUNDATION}
    lines = ["flowchart TB"]
    for package, caption in TOOLKIT.items():
        lines.append(f'    subgraph {package}["{caption}"]')
        for n in sorted(n for n in nodes - OMIT - FOUNDATION if home[n] == package):
            lines.append(f'        {n}["{LABEL.get(n, n)}"]')
        lines.append("    end")
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
