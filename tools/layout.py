"""The repository's layout rules, and the file map they keep current (docs/reference/MODULES.md).

    python tools/layout.py            # write the file map
    python tools/layout.py --check    # the gate: a broken rule or a stale file map fails

Each rule is a kind of drift that once made the tree hard to read (CONTRIBUTING.md, "The rules"):

- **names**: a Python file's name says what the file does, in plain words. It never repeats its folder's
  name (tests/nygrid/test_nygrid_areas.py), never differs from a sibling's by a short ending alone
  (reduce.py beside reduced.py), and carries no digit and no citation key or first author's surname
  from docs/reference/REFERENCES.md (wu26.py, solution1.py).
- **docstrings**: every Python file opens with a docstring. Its first paragraph is the file's line in the
  file map.
- **package pages**: every package of the two toolkits (`fdia_graph` under `src/`, `fdia_analysis` under
  `analysis/src/`) has one hand-written page (`PAGES`), and that page names each module of the package by
  file name (`base.py`) or dotted path (`se.base`), so a new module cannot land without its page saying
  where it fits. Where both toolkits have a module of one name (`models/config.py`), the analysis one is
  named with its package (`fdia_analysis.models.config`).
- **cited names**: a code span in a Markdown file or a Python file that names a module, or a name in one
  (`fdia_analysis.trust.base`, `trust.base.TrustSelector`, `ramp.py`), names something that exists, in the
  package it says or, with no package said, in either; a span that is an experiment of the results store
  (`se.estimators`, from results/<experiment>.csv) is one too. The CHANGELOG (whose entries name what a
  change removed) is not read.
- **repository map**: every folder at the root of the tree and every folder directly under `docs/` has its role in `REPO` or
  `DOCS`, and every folder named there exists, so a new place for files cannot appear without saying what it
  holds. The map is the first section of the file map.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass

from sources import ANALYSIS, GENERATION, PACKAGES, ROOT, TESTS

MAP = "docs/reference/MODULES.md"

# Each package's hand-written page, by toolkit and by its folder under the toolkit's package ("" is the
# top level).
PAGES = {
    GENERATION: {
        "": "docs/ROADMAP.md",
        "dataset": "docs/ROADMAP.md",
        "models": "docs/ROADMAP.md",
        "ratings": "docs/ROADMAP.md",
        "formulas": "docs/reference/FORMULAS.md",
        "engine": "docs/guides/generation.md",
        "engine/attacks": "docs/guides/generation.md",
        "generation": "docs/guides/generation.md",
        "engine/sensing": "docs/concepts/new_york_grid.md",
        "generation/nygrid": "docs/concepts/new_york_grid.md",
    },
    ANALYSIS: {
        "": "analysis/README.md",
        "models": "docs/ROADMAP.md",
        "formulas": "docs/reference/FORMULAS.md",
        "results": "results/README.md",
        "se": "docs/analysis/estimation/README.md",
        "se/detect": "docs/concepts/new_york_grid.md",
        "localization": "docs/analysis/localization/README.md",
        "trust": "docs/analysis/trust/README.md",
        "federated": "docs/analysis/federated/README.md",
    },
}

# The repository map: each folder at the root of the tree with what it holds. The order is the reading order.
REPO = {
    "src": "the generation toolkit, the package `fdia_graph` (the distribution fdia-graph): generating a "
    "timeline and loading it",
    "analysis": "the analysis toolkit, the package `fdia_analysis` (the distribution fdia-analysis, which "
    "depends on fdia-graph): state estimation, localization, federated training, trust selection and the "
    "results store's code, with its own pyproject.toml, `src/` and `tests/`",
    "tests": "the generation toolkit's test suite, and the helpers both suites share (their shared fixtures are in the root `conftest.py`)",
    "examples": "scripts a user copies: the quickstart, the training examples, the New York extension points",
    "results": "the results store: data only (one CSV per experiment and `runs.csv`), with the queries the docs "
    "read it through",
    "pipelines": "the builds of released data: the New York pool and timeline, and the MATLAB reference they are "
    "checked against",
    "tools": "repository tooling only: the gates, the generated docs, pull requests and releases",
    "docs": "the documentation, one folder per kind of page",
    ".github": "the CI workflows and the dependency updates",
}
# Each folder of docs/ (the Diataxis kinds, and the images they share).
DOCS = {
    "getting-started": "tutorials: a first run, start to finish",
    "guides": "how to do one task",
    "concepts": "explanations: how a part is built and why",
    "analysis": "one page per study with its figures: what was measured, on what, and the result",
    "reference": "facts to look up: the file map, the data dictionary, the glossary, the formulas, the references",
    "figures": "the images pages share, each with its data or its diagram source",
}

# The file map's sections: (heading, the folder its files live under); a file under none goes in the last.
SECTIONS = (
    ("The generation toolkit, `fdia_graph`", PACKAGES[GENERATION]),
    ("The analysis toolkit, `fdia_analysis`", PACKAGES[ANALYSIS]),
    ("Tools", "tools"),
    ("The generation toolkit's tests", TESTS[0]),
    ("The analysis toolkit's tests", TESTS[1]),
    ("Examples", "examples"),
    ("Pipelines", "pipelines"),
    ("The results store's queries", "results"),
)
OTHER = "Other files"
BEGIN, END = "<!-- file-map: generated by tools/layout.py -->", "<!-- /file-map -->"
HISTORY = ("CHANGELOG.md",)  # history: its entries may cite what a later change removed
SHORT_ENDING = 2  # `reduce` and `reduced`, `load` and `loads`: endings this short read as the same word


@dataclass(frozen=True)
class Problem:
    rule: str
    where: str
    what: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.where}: {self.what}"


def _git(*args: str) -> set[str]:
    out = subprocess.run(["git", *args, "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return {n for n in out.split(chr(0)) if n}


def tree() -> list[str]:
    """Every file of the checkout, tracked or new (not ignored), that exists in the working tree: a new file is in
    the file map before it is committed, so a stray file left in the checkout shows up there (delete it or ignore
    it)."""
    names = _git("ls-files") | _git("ls-files", "--others", "--exclude-standard")
    return sorted(f for f in names if os.path.exists(os.path.join(ROOT, f)))


def _read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def summary(rel: str) -> str:
    """The first paragraph of a Python file's docstring, on one line ("" when it has none)."""
    doc = ast.get_docstring(ast.parse(_read(rel))) or ""
    return " ".join(doc.split("\n\n", 1)[0].split())


# ---- names ----------------------------------------------------------------------------------------


def _reference_words() -> set[str]:
    """Each citation key of REFERENCES.md, lower-cased (`wu26`), and its first author's surname (`wu`)."""
    words: set[str] = set()
    for key, text in re.findall(
        r"^\| \[([A-Z0-9]+)\] \| (.+?) \|$", _read("docs/reference/REFERENCES.md"), re.M
    ):
        words.add(key.lower())
        # initials, then the surname, which may open with a particle ("U. von Luxburg"): its last word counts
        m = re.match(r"(?:[A-Z]\.(?:-[A-Z]\.)?\s*)+((?:[a-z]+\s+)*[A-Z][\w'-]+)", text)
        if m:
            words.add(m.group(1).split()[-1].lower())
    return words


def _words(stem: str) -> list[str]:
    parts = stem.split("_")
    return parts[1:] if parts[0] == "test" else parts


def names(files: list[str]) -> list[Problem]:
    py = [f for f in files if f.endswith(".py")]
    cited = _reference_words()
    out = []
    by_folder: dict[str, list[str]] = defaultdict(list)
    for f in py:
        folder, stem = os.path.dirname(f), os.path.basename(f)[:-3]
        by_folder[folder].append(stem)
        words = _words(stem)
        parent = os.path.basename(folder)
        if parent and f"_{parent}_" in "_" + "_".join(words) + "_":  # a folder name may hold an underscore
            out.append(Problem("names", f, f"repeats its folder's name {parent!r}"))
        if re.search(r"\d", stem):
            out.append(Problem("names", f, "carries a digit; say what the file does"))
        hits = sorted(set(words) & cited)
        if hits:
            out.append(
                Problem("names", f, f"carries a citation key or author's name {hits}; say what the file does")
            )
    for folder, stems in by_folder.items():
        for a in stems:
            for b in stems:
                tail = b[len(a) :]
                if a != b and b.startswith(a) and len(tail) <= SHORT_ENDING and "_" not in tail:
                    out.append(Problem("names", f"{folder}/{b}.py", f"differs from {a}.py by {tail!r} alone"))
    return out


# ---- docstrings and the file map -------------------------------------------------------------------


def docstrings(files: list[str]) -> list[Problem]:
    return [
        Problem("docstrings", f, "has no module docstring")
        for f in files
        if f.endswith(".py") and not summary(f)
    ]


def _cell(text: str) -> str:
    return text.replace("|", "/")


def _package_of(rel: str) -> str:
    """The toolkit package a file belongs to ("" for a file of neither)."""
    return next((p for p, folder in PACKAGES.items() if rel.startswith(folder + "/")), "")


def _module(rel: str) -> str:
    """The dotted module of a package file under its package: `analysis/src/fdia_analysis/se/base.py` ->
    `se.base`, its `__init__.py` -> `se`, the package's own `__init__.py` -> the package's name."""
    package = _package_of(rel)
    inside = rel[len(PACKAGES[package]) + 1 : -len(".py")]
    return inside.replace("/", ".").removesuffix("__init__").rstrip(".") or package


def file_map(files: list[str]) -> str:
    """Every Python file, by section, with the first paragraph of its docstring: a package file by its module
    (`se.base`), any other by its path. A test folder's `__init__.py` is left out."""
    py = [f for f in files if f.endswith(".py")]
    placed: set[str] = set()
    blocks = []
    for heading, folder in (*SECTIONS, (OTHER, "")):
        rows = [f for f in py if f not in placed and (not folder or f.startswith(folder + "/"))]
        placed |= set(rows)
        packaged = folder in PACKAGES.values()
        rows = [f for f in rows if packaged or not f.endswith("__init__.py")]
        if not rows:
            continue
        lines = [f"## {heading}", "", "| file | what it does |", "|---|---|"]
        for f in rows:
            shown = _module(f) if packaged else f
            lines.append(f"| `{shown}` | {_cell(summary(f))} |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def repo_map() -> str:
    """The repository map as the file map's first section: each folder of `REPO`, and under `docs/` each of `DOCS`."""
    lines = ["## The repository", "", "| folder | what it holds |", "|---|---|"]
    for folder, role in REPO.items():
        lines.append(f"| `{folder}/` | {_cell(role)} |")
        if folder == "docs":
            lines += [f"| `docs/{sub}/` | {_cell(role)} |" for sub, role in DOCS.items()]
    return "\n".join(lines)


def repository(files: list[str]) -> list[Problem]:
    """Every folder at the root of the tree is in `REPO` and every folder of docs/ in `DOCS`, and both name only
    folders that exist."""
    roots = {f.split("/", 1)[0] for f in files if "/" in f}
    docs = {f.split("/")[1] for f in files if f.startswith("docs/") and f.count("/") >= 2}
    out = []
    for where, found, mapped in (("", roots, REPO), ("docs/", docs, DOCS)):
        out += [
            Problem("repository map", where + name, f"has no role: add it to {'DOCS' if where else 'REPO'}")
            for name in sorted(found - set(mapped))
        ]
        out += [
            Problem("repository map", where + name, "is mapped but holds no file")
            for name in sorted(set(mapped) - found)
        ]
    return out


def rendered_map(files: list[str]) -> str:
    head = (
        "# Modules\n\n"
        "What every folder of the repository holds, then what every Python file does, in one line: the first "
        "paragraph of its docstring. Generated by `tools/layout.py`; edit the docstring (or `REPO` and `DOCS` in the "
        "tool), not this page. A toolkit's section names each file by its module under the toolkit's package: "
        "`dataset.base` is `src/fdia_graph/dataset/base.py`, `se.base` is `analysis/src/fdia_analysis/se/base.py`.\n\n"
    )
    return f"{head}{BEGIN}\n{repo_map()}\n\n{file_map(files)}\n{END}\n"


# ---- package pages ---------------------------------------------------------------------------------


Package = tuple[str, str]  # (toolkit package, folder under it): ("fdia_analysis", "se/detect")


def _package_modules(files: list[str]) -> dict[Package, list[str]]:
    out: dict[Package, list[str]] = defaultdict(list)
    for f in files:
        package = _package_of(f)
        if package and f.endswith(".py"):
            folder = os.path.dirname(f)[len(PACKAGES[package]) + 1 :]
            out[(package, folder)]  # a package with only an __init__ still needs its page
            if not f.endswith("__init__.py"):
                out[(package, folder)].append(os.path.basename(f)[:-3])
    return out


def _explicit(text: str, where: Package, m: str, shared: bool) -> bool:
    """`text` names module `m` of the package `where` unambiguously: with its toolkit, by path
    (`fdia_analysis/se/base.py`, optionally under its source root) or dotted (`fdia_analysis.se.base`); or,
    when the other toolkit has no module of that folder and name (`shared` false), by its path or dotted
    path from the toolkit's package (`se/base.py`, `se.base`)."""
    package, folder = where
    inside = f"{folder}/{m}" if folder else m
    root = os.path.dirname(PACKAGES[package])
    forms = [
        rf"(?<![\w/])(?:{re.escape(root)}/)?{re.escape(f'{package}/{inside}.py')}\b",
        rf"(?<![\w.]){re.escape(f'{package}.{inside}'.replace('/', '.'))}\b",
    ]
    if folder and not shared:
        forms.append(rf"(?<![\w/]){re.escape(f'{inside}.py')}\b")
        forms.append(rf"(?<![\w.]){re.escape(inside.replace('/', '.'))}\b")
    return any(re.search(f, text) for f in forms)


def pages(files: list[str]) -> list[Problem]:
    """Each package's page names each of its modules, explicitly (`_explicit`) or by the bare `<module>.py` once every
    other package on the same page with a module of that name is named explicitly (so a bare `base.py` on a page of
    two packages with a `base.py` counts for neither until one of them is written out)."""
    out = []
    packages = _package_modules(files)

    def shared(where: Package, m: str) -> bool:
        return any(o != where and o[1] == where[1] and m in ms for o, ms in packages.items())

    for where, mods in sorted(packages.items()):
        package, folder = where
        path = f"{PACKAGES[package]}/{folder}".rstrip("/")
        page = PAGES[package].get(folder)
        if page is None:
            out.append(Problem("pages", path, "has no page: add one to PAGES in tools/layout.py"))
            continue
        text = _read(page)
        for m in mods:
            others = [
                o for o, ms in packages.items() if o != where and PAGES[o[0]].get(o[1]) == page and m in ms
            ]
            bare = re.search(rf"(?<![\w/.]){m}\.py\b", text) and all(
                _explicit(text, o, m, shared(o, m)) for o in others
            )
            if not (_explicit(text, where, m, shared(where, m)) or bare):
                out.append(Problem("pages", f"{path}/{m}.py", f"is not named on its page {page}"))
    return out


# ---- cited names -----------------------------------------------------------------------------------


def _defined(rel: str) -> tuple[set[str], bool]:
    """The names a module defines or imports at its top level, and whether it has a module `__getattr__`
    (lazy or moved names: anything may resolve)."""
    tree_ = ast.parse(_read(rel))
    found: set[str] = set()
    lazy = False
    for n in tree_.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(n.name)
            lazy |= n.name == "__getattr__"
        elif isinstance(n, ast.Assign):
            found |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            found.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            found |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, (ast.If, ast.Try)):  # TYPE_CHECKING blocks and optional imports
            for sub in ast.walk(n):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    found |= {(a.asname or a.name).split(".")[0] for a in sub.names}
    return found, lazy


def resolves(dotted: str, files: set[str], package: str) -> bool:
    """`dotted` (under the toolkit `package`) names a package, a module, or a top-level name of one. Past the
    first name inside a module nothing is checked: an attribute of a class may be inherited or set at run time."""
    folder = PACKAGES[package]
    parts = dotted.split(".")
    for i, part in enumerate(parts):
        if f"{folder}/{part}/__init__.py" in files:
            folder = f"{folder}/{part}"
            continue
        if f"{folder}/{part}.py" in files:
            if i + 1 == len(parts):
                return True
            found, lazy = _defined(f"{folder}/{part}.py")
            return lazy or parts[i + 1] in found
        init = f"{folder}/__init__.py"
        found, lazy = _defined(init)
        return lazy or part in found
    return True


_SPAN = re.compile(r"`([A-Za-z_][\w.]*)`")
DATA = (
    "npz",
    "npy",
    "csv",
    "h5",
    "json",
    "mat",
    "txt",
    "yml",
    "yaml",
    "toml",
    "png",
    "svg",
    "pdf",
    "pkl",
    "md",
    "py",
)
_PY_FILE = re.compile(r"`([\w-]+\.py)`")


def _tops(files: list[str], package: str) -> set[str]:
    """The modules and subpackages at the top of a toolkit's package (`schema`, `dataset`)."""
    folder = PACKAGES[package]
    depth = folder.count("/") + 1
    tops = {os.path.basename(f)[:-3] for f in files if os.path.dirname(f) == folder and f.endswith(".py")}
    tops |= {f.split("/")[depth] for f in files if f.startswith(folder + "/") and f.count("/") > depth}
    return tops - {"__init__"}


def _names_something(span: str, present: set[str], tops: dict[str, set[str]]) -> bool:
    """Whether a code span that reads as a module path names something. A span that says its toolkit
    (`fdia_graph.schema.Attr`) must resolve in that toolkit; one that does not (`se.base`) is a module path
    when its first name is a top-level module of either toolkit, and then must resolve in one of them."""
    said = next((p for p in PACKAGES if span.startswith(p + ".")), None)
    name = span[len(said) + 1 :] if said else span
    first = name.split(".")[0]
    if said:
        return ("." not in name and first in tops[said]) or resolves(name, present, said)
    homes = [p for p in PACKAGES if first in tops[p]]
    if "." not in name or not homes:
        return True  # not a module path
    return any(resolves(name, present, p) for p in homes)


def cited(files: list[str]) -> list[Problem]:
    present = set(files)
    tops = {package: _tops(files, package) for package in PACKAGES}
    basenames = {os.path.basename(f) for f in files}
    experiments = {
        os.path.basename(f)[: -len(".csv")] for f in files if re.fullmatch(r"results/[\w.]+\.csv", f)
    }
    out = []
    for rel in files:
        if not rel.endswith((".md", ".py")) or rel.startswith(HISTORY) or rel == MAP:
            continue
        text = _read(rel)
        for span in sorted(set(_SPAN.findall(text))):
            if span.rsplit(".", 1)[-1] in DATA or span in experiments:  # a file name or a results experiment
                continue
            if not _names_something(span, present, tops):
                out.append(Problem("cited names", rel, f"`{span}` names nothing in {' or '.join(PACKAGES)}"))
        for span in sorted(set(_PY_FILE.findall(text))):
            if span not in basenames:
                out.append(Problem("cited names", rel, f"`{span}` is no file of the repository"))
    return out


# ---- the gate --------------------------------------------------------------------------------------


def check(files: list[str]) -> list[Problem]:
    problems = names(files) + docstrings(files) + pages(files) + cited(files) + repository(files)
    current = _read(MAP) if os.path.exists(os.path.join(ROOT, MAP)) else ""
    if current != rendered_map(files):
        problems.append(Problem("file map", MAP, "is stale: run python tools/layout.py"))
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="fail on a broken rule or a stale file map")
    args = ap.parse_args()
    files = tree()
    if not args.check:
        with open(os.path.join(ROOT, MAP), "w", encoding="utf-8", newline="\n") as f:
            f.write(rendered_map(files))
        files = tree()  # the map itself is now part of the tree
    problems = check(files)
    for p in problems:
        print(p)
    by_rule = defaultdict(int)
    for p in problems:
        by_rule[p.rule] += 1
    print(f"{len(problems)} problem(s)" + (f": {dict(by_rule)}" if problems else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
