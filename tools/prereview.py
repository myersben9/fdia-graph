"""Every check a pull request must pass, for both toolkits of the repository (`fdia_graph` and
`fdia_analysis`, `sources.PACKAGES`), run locally in one command before the review is requested
(CONTRIBUTING.md, "The pull request").

    python tools/prereview.py                  # every gate, against origin/main
    python tools/prereview.py --base main      # a different base for the changed-lines gates
    python tools/prereview.py --fast           # skip the test suite (the gates only)
    python tools/prereview.py --also-python C:/envs/py312/python.exe   # the suite again on another Python

CI runs the suite on Python 3.9 and 3.12 and on Windows; this gate runs one interpreter, the one it is
started with. A parser of loose input can behave differently between versions, so an interpreter
with the test extras installed can be added with `--also-python` (repeatable) or, set once per
machine, `FDIA_PREREVIEW_PYTHONS` (paths joined by the platform's path separator).

It runs the gates CI runs (formatting, lint, types, readability, the layout rules and the file map,
the generated diagrams and data dictionary) and the two test suites (`tests`, `analysis/tests`) on this checkout's
own source (`PYTHONPATH` is set to `src` and `analysis/src`, so an
editable install of another checkout cannot stand in for it), plus checks CI does not run, each one
a kind of finding reviews kept raising (`docs/reference/REVIEW_CHECKLIST.md`). The changed files are
the branch's commits against the base, the uncommitted changes and the untracked files.

- **changelog**: never more than one `## Unreleased`; a change under `src/` or `analysis/src/` needs
  exactly one, with at least one entry, and CHANGELOG.md changed.
- **rendered diagrams**: a changed `docs/figures/diagrams/*.mmd` has its `.png` and `.svg` changed too;
  and every hand-written flow diagram reads on a phone: `flowchart TB`, at most 8 nodes, a PNG at most
  900 px wide (every run; the generated class and module diagrams are exempt, and so are the diagrams
  listed in `NOT_YET_REDRAWN`, which the site's stylesheet must name too).
- **plain pages**: no page under `docs/`, nor the README, holds a fenced Mermaid block or a decision
  code of the planning notes (a capital N or D and one to three digits, as a word of its own); every
  run. The pages not yet cleaned are listed in `NOT_YET_PLAIN`; CHANGELOG.md is not read.
- **cited paths**: every repository path a Markdown file cites exists (every file, every run: a
  page nobody edits goes stale too) (a file or a folder,
  with or without an extension; in CHANGELOG.md the `## Unreleased` section only, the released
  sections being history), in a code span whose first part is a root entry of the tree (or a
  dotfile or upper-case Markdown name), or as a link or image destination (relative to the file; anchors, queries and web links are ignored).
  A code span is read from the root (`docs/analysis/estimation/README.md`), or from the citing file's folder when it starts with `./` or `../`.
- **vacuous tests**: no `or True` / `assert True` in `tests/`.
- **integer fields**: every `int` field of a model in either toolkit's `models/config.py` or
  `models/inputs.py` carries the `Integer()` rule (SKIP where the checkout has no such models).

Exit status 0 only when nothing fails. The review checklist in
`docs/reference/REVIEW_CHECKLIST.md` covers what no tool can check.
"""

from __future__ import annotations

import argparse
import ast
import os
import posixpath
import re
import subprocess
import sys
import time
from dataclasses import dataclass

from render_mermaid import flow_faults, is_generated
from sources import CONFTEST, PACKAGES, ROOT, SOURCE_ROOTS, TESTS, python_path

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
SCRIPTS = ("tools", "pipelines")  # the folders of scripts, formatted and linted with the packages


@dataclass(frozen=True)
class Gate:
    name: str
    cmd: list[str]
    slow: bool = False


def _suite(py: str) -> list[str]:
    """The two test suites on interpreter `py`, spread over worker processes when it has pytest-xdist (the
    test extra installs it). Every worker builds its own tiny timeline in its own cache, so more
    workers than a few repeat that build more than they save; 8 at most."""
    has_xdist = subprocess.run([py, "-c", "import xdist"], capture_output=True).returncode == 0
    workers = ["-n", str(min(8, os.cpu_count() or 1))] if has_xdist else []
    strict = [flag for package in PACKAGES for flag in ("-W", f"error::DeprecationWarning:{package}")]
    return [py, "-m", "pytest", "-q", "-x", *workers, *strict, *TESTS]


def _version(py: str) -> str:
    out = subprocess.run(
        [py, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], capture_output=True, text=True
    )
    return out.stdout.strip() or "?"


def gates(base: str, also: list[str]) -> list[Gate]:
    py = sys.executable
    return [
        Gate(
            "format", [py, "-m", "ruff", "format", "--check", *PACKAGES.values(), CONFTEST, *TESTS, *SCRIPTS]
        ),
        Gate("lint", [py, "-m", "ruff", "check", *SOURCE_ROOTS, CONFTEST, *TESTS, *SCRIPTS]),
        Gate("types", [py, "-m", "pyright", *PACKAGES.values()]),
        Gate("readability", [py, "tools/readability.py", "--report", "--check", "--base", base]),
        Gate("layout", [py, "tools/layout.py", "--check"]),
        Gate("class diagrams", [py, "tools/class_diagrams.py", "--check"]),
        Gate("data dictionary", [py, "tools/models_doc.py", "--check"]),
        Gate("results docs", [py, "tools/results_docs.py", "--check"]),
        Gate(f"tests ({_version(py)})", _suite(py), slow=True),
        *(Gate(f"tests ({_version(other)})", _suite(other), slow=True) for other in also),
    ]


def run(gate: Gate) -> tuple[bool, str, float]:
    env = {**os.environ, "PYTHONPATH": python_path()}
    t0 = time.time()
    p = subprocess.run(
        gate.cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return p.returncode == 0, (p.stdout + p.stderr).strip(), time.time() - t0


def _git_names(*args: str) -> set[str]:
    """The paths a git listing prints, read NUL-separated (`-z`), so a path with a space is one path."""
    out = subprocess.run(["git", *args, "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return {name for name in out.split(chr(0)) if name}


def changed(base: str) -> list[str]:
    """The branch's commits against `base`, the uncommitted changes and the untracked files."""
    return sorted(
        _git_names("diff", "--name-only", f"{base}...HEAD")
        | _git_names("diff", "--name-only", "HEAD")
        | _git_names("ls-files", "--others", "--exclude-standard")
    )


def _tree() -> set[str]:
    """Every file of the checkout, tracked or new (not ignored); a tracked file deleted in the working
    tree is not part of it."""
    names = _git_names("ls-files") | _git_names("ls-files", "--others", "--exclude-standard")
    return {f for f in names if os.path.exists(os.path.join(ROOT, f))}


def _read(rel: str) -> str:
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


def _unreleased_entries(text: str) -> int:
    """The bullets under `## Unreleased`, up to the next `## ` heading."""
    m = re.search(r"^## Unreleased[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return len(re.findall(r"^- ", m.group(1), re.M)) if m else 0


def _changelog(files: list[str]) -> tuple[str, str]:
    text = _read("CHANGELOG.md")
    heads = len(re.findall(r"^## Unreleased", text, re.M))
    if heads > 1:
        return FAIL, f"CHANGELOG.md has {heads} '## Unreleased' sections"
    src = [f for f in files if f.startswith(tuple(root + "/" for root in SOURCE_ROOTS))]
    if src and heads != 1:
        return FAIL, "a change to a package's source needs a '## Unreleased' section in CHANGELOG.md"
    if src and _unreleased_entries(text) == 0:
        return FAIL, "the '## Unreleased' section of CHANGELOG.md has no entry"
    if src and "CHANGELOG.md" not in files:
        return FAIL, f"{len(src)} source file(s) of the packages changed but CHANGELOG.md did not"
    return PASS, ""


_SPAN = re.compile(r"`([\w.][\w./-]*)`")  # a code span with no space: a candidate path


def _is_path(span: str, top: set[str]) -> bool:
    """A code span names a repository path when its first part is an entry at the root of the tree
    (`tools/pr.py`, `.github/workflows/docs.yml`, `CHANGELOG.md`), or it is a dotfile or an upper-case
    Markdown name, which a deleted root file would otherwise slip past. `fg.load` and `np.ndarray`
    are not: no root entry is called `fg` or `np`."""
    first = span.split("/", 1)[0]
    return (
        first in top
        or (span.startswith(".") and "." in span[1:])
        or bool(re.fullmatch(r"[A-Z][A-Z_]*\.md", span))
    )


_LINKED = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")


_REFERENCE = re.compile(r"^ {0,3}\[[^\]]+\]:\s*<?([^\s>]+)>?", re.M)  # a reference link: [name]: target


_FENCED = re.compile(  # a fenced block: the whole opening run, closed by a run of the same character
    r"^ {0,3}(`{3,}).*?^ {0,3}\1`*[ \t]*$|^ {0,3}(~{3,}).*?^ {0,3}\2~*[ \t]*$",
    re.M | re.S,
)


def _targets(md: str, text: str) -> list[str]:
    """The repository paths a Markdown file cites: backticked repository-rooted paths, and link and
    image destinations resolved against the file's folder (web links, mail and anchors skipped)."""
    top = {f.split("/", 1)[0] for f in _tree()}
    text = _FENCED.sub("", text)  # a fenced code block cites nothing: `[x](t)` there is code
    spans = _SPAN.findall(_LINKED.sub("", text))  # a link's text is not a citation; its target is
    out = []
    for s in spans:
        if s.startswith(("./", "../")):  # relative to the citing file, as a link is
            out.append(posixpath.normpath(posixpath.join(posixpath.dirname(md), s)))
        elif _is_path(s, top):
            out.append(s)
    prose = re.sub(r"`[^`\n]*`", "", text)  # a code span holds no link
    for dest in _LINKED.findall(prose) + _REFERENCE.findall(prose):
        # a web or mail link (`https:`, `mailto:`, protocol-relative `//host`) or an anchor
        if re.match(r"[a-z][a-z0-9+.-]*:", dest, re.I) or dest.startswith(("#", "//")):
            continue
        path = dest.split("#", 1)[0].split("?", 1)[0]
        if path:
            out.append(posixpath.normpath(posixpath.join(posixpath.dirname(md), path)))
    return out


def _in_tree(path: str) -> bool:
    """The path exists inside this checkout: an absolute path, or one climbing out with `..`, names
    something that is not part of the repository even when it exists on this machine."""
    full = os.path.realpath(os.path.join(ROOT, path))
    return os.path.commonpath([full, os.path.realpath(ROOT)]) == os.path.realpath(ROOT) and os.path.exists(
        full
    )


def _current_text(md: str) -> str:
    """The part of a Markdown file whose citations must hold today: the whole file, except the
    changelog, whose released sections are history and may cite files a later release removed."""
    text = _read(md)
    if md != "CHANGELOG.md":
        return text
    if "## Unreleased" not in text:  # nothing unreleased: every section is history
        return ""
    head, rest = text.split("## Unreleased", 1)
    return head + "## Unreleased" + rest.split("\n## ", 1)[0]  # up to the next release's heading


def _cited_paths(files: list[str]) -> tuple[str, str]:
    # a bare name such as `REFERENCES.md` is shorthand for a file of that name somewhere in the tree
    names = {posixpath.basename(f) for f in _tree()}
    missing = [
        f"{f}: {path}"
        for f in files
        if f.endswith(".md") and os.path.exists(os.path.join(ROOT, f))
        for path in _targets(f, _current_text(f))
        if not _in_tree(path) and not ("/" not in path and path in names)
    ]
    return (FAIL if missing else PASS), "missing: " + "; ".join(missing[:10])


def _vacuous_tests() -> tuple[str, str]:
    hits = [
        f"{rel}:{i}"
        for rel in sorted(f for f in _tree() if f.startswith("tests/") and f.endswith(".py"))
        for i, line in enumerate(_read(rel).splitlines(), 1)
        if re.search(r"\bor True\b|\bassert True\b", line)
    ]
    return (FAIL if hits else PASS), "vacuous assertions: " + ", ".join(hits)


_MODELS = tuple(
    f"{folder}/models/{module}.py" for folder in PACKAGES.values() for module in ("config", "inputs")
)


def _expand(ann: str, aliases: dict[str, str]) -> str:
    """An annotation with every module-level alias in it replaced by its definition, to any depth:
    `Optional[Count]` reads as `Optional[Annotated[int, Integer(), AtLeast(1)]]`."""
    for _ in range(10):  # aliases of aliases; ten levels is far more than any module uses
        new = (
            re.sub(r"\b(" + "|".join(map(re.escape, aliases)) + r")\b", lambda m: aliases[m.group(1)], ann)
            if aliases
            else ann
        )
        if new == ann:
            break
        ann = new
    return ann


def _integer_fields() -> tuple[str, str]:
    """Every `int` field of a model in a toolkit's models/config.py or models/inputs.py carries
    `Integer()`, directly or through a module-level alias (`Count = Annotated[int, Integer(), AtLeast(1)]`)."""
    present = [rel for rel in _MODELS if os.path.exists(os.path.join(ROOT, rel))]
    if not present:
        return SKIP, "no config or input models in this checkout"
    bare = []
    for rel in present:
        tree = ast.parse(_read(rel))
        aliases = {
            t.id: ast.unparse(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Name)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                ann = _expand(ast.unparse(node.annotation), aliases)
                # a scalar int setting, bare or wrapped in Optional/Annotated in any order; a
                # collection of ints (tuple[int, ...]) is not one
                if re.match(r"((Annotated|Optional)\[)*int\b", ann) and "Integer()" not in ann:
                    bare.append(f"{rel}:{node.lineno} {node.target.id}: {ann}")
    return (FAIL if bare else PASS), "int fields without Integer(): " + "; ".join(bare)


DIAGRAMS = "docs/figures/diagrams/"
STYLESHEET = "docs/stylesheets/extra.css"  # limits the display width of the phone diagrams on the site
# Hand-written flow diagrams exempt from the phone rules, each with its reason: wide renders of the old
# renderer, left as they are. Redrawing one top to bottom and re-rendering it, or deleting it, takes
# its name off this list (and off the stylesheet's).
NOT_YET_REDRAWN = {
    "concepts_one_record": "not yet redrawn",
    "localization_flow": "not yet redrawn",
    "se_estimators": "not yet redrawn",
    "se_gated_prior": "not yet redrawn",
    "se_jacobian_weighting": "not yet redrawn",
    "trust_flow": "not yet redrawn",
    "pipeline": "kept until the next release: the published PyPI page loads it",
}


def _rendered_diagrams(files: list[str]) -> tuple[str, str]:
    """A changed diagram source has its two images changed with it, and every hand-written flow
    diagram reads on a phone (`render_mermaid.flow_faults`), the generated class and module diagrams
    and the `NOT_YET_REDRAWN` ones aside."""
    stale = [
        f
        for f in files
        if f.startswith(DIAGRAMS)
        and f.endswith(".mmd")
        and os.path.exists(os.path.join(ROOT, f))
        and not {f[:-4] + ".png", f[:-4] + ".svg"} <= set(files)
    ]
    sources = sorted(f for f in _tree() if f.startswith(DIAGRAMS) and f.endswith(".mmd"))
    names = {posixpath.basename(f)[:-4] for f in sources}
    faults = [
        f"{f} {fault}"
        for f in sources
        if not is_generated(f) and posixpath.basename(f)[:-4] not in NOT_YET_REDRAWN
        for fault in flow_faults(os.path.join(ROOT, f))
    ]
    faults += [
        f"NOT_YET_REDRAWN names {name}, which is no diagram" for name in NOT_YET_REDRAWN if name not in names
    ]
    # the site shows a phone diagram at half its pixels; the stylesheet leaves the wide ones alone by name
    css = _read(STYLESHEET) if os.path.exists(os.path.join(ROOT, STYLESHEET)) else ""
    spared = set(re.findall(r'\[src\$="/(\w+)\.png"\]', css)) - {"modules"}
    if spared != set(NOT_YET_REDRAWN):
        odd = sorted(spared ^ set(NOT_YET_REDRAWN))
        faults.append(f"{STYLESHEET} and NOT_YET_REDRAWN disagree on " + ", ".join(odd))
    detail = "; ".join((["re-render: " + ", ".join(stale)] if stale else []) + faults)
    return (FAIL if stale or faults else PASS), detail


_MERMAID = re.compile(r"^ {0,3}(?:`{3,}|~{3,})[ \t]*\{?\.?mermaid\b", re.M)  # the opening of a Mermaid block
# A decision code of the project's planning notes: a capital N or D and one to three digits, a word of its
# own, or a range of them ("N27-N37"). A hyphen to anything else makes it part of another name
# ("Sec. IV-D1" of a paper, "N-1"), which is not one.
_DECISION_CODE = re.compile(r"(?<![\w-])[ND]\d{1,3}(?:-[ND]\d{1,3})*(?![\w-])")
# Pages that still carry decision codes, each with its reason (`{path: reason}`). Cleaning a page, or
# deleting it, takes it off this list; no page is on it.
NOT_YET_PLAIN: dict[str, str] = {}


def page_faults(text: str) -> list[str]:
    """What a reader of a docs page cannot use: a Mermaid block, which the GitHub mobile app and PyPI show as
    code, and the decision codes of the planning notes, which mean nothing outside them."""
    faults = ["has a Mermaid block (render the diagram to an image)"] if _MERMAID.search(text) else []
    codes = sorted(set(_DECISION_CODE.findall(text)))
    if codes:
        faults.append("has decision codes (say the reason in plain words): " + ", ".join(codes))
    return faults


def _plain_pages() -> tuple[str, str]:
    """Every page under `docs/`, and the README, is free of `page_faults`, the `NOT_YET_PLAIN` ones aside;
    the list names only pages that still have a decision code."""
    pages = sorted(f for f in _tree() if f.endswith(".md") and (f.startswith("docs/") or f == "README.md"))
    found = {f: page_faults(_read(f)) for f in pages}
    faults = [
        f"{f} {fault}"
        for f, of_page in found.items()
        for fault in of_page
        if f not in NOT_YET_PLAIN or fault.startswith("has a Mermaid")
    ]
    faults += [
        f"NOT_YET_PLAIN names {f}, which has no decision code"
        for f in NOT_YET_PLAIN
        if not any(fault.startswith("has decision") for fault in found.get(f, []))
    ]
    return (FAIL if faults else PASS), "; ".join(faults)


def repo_checks(files: list[str]) -> list[tuple[str, str, str]]:
    """The checks CI does not run: (name, PASS/FAIL/SKIP, detail)."""
    return [
        ("plain pages", *_plain_pages()),
        ("changelog", *_changelog(files)),
        (
            "cited paths",
            *_cited_paths(sorted(f for f in _tree() if f.endswith(".md"))),
        ),
        ("vacuous tests", *_vacuous_tests()),
        ("integer fields", *_integer_fields()),
        ("rendered diagrams", *_rendered_diagrams(files)),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--fast", action="store_true", help="skip the test suite")
    ap.add_argument(
        "--also-python", action="append", default=[], help="run the suite on this interpreter too"
    )
    a = ap.parse_args()
    also = a.also_python + [p for p in os.environ.get("FDIA_PREREVIEW_PYTHONS", "").split(os.pathsep) if p]
    results: list[tuple[str, str, str]] = []
    for gate in gates(a.base, also):
        if gate.slow and a.fast:
            continue
        ok, out, secs = run(gate)
        print(f"{PASS if ok else FAIL}  {gate.name:16s} {secs:6.1f}s")
        results.append((gate.name, PASS if ok else FAIL, out))
    for name, status, detail in repo_checks(changed(a.base)):
        print(f"{status}  {name:16s} {detail if status == SKIP else ''}")
        results.append((name, status, detail))
    failed = [(n, out) for n, status, out in results if status == FAIL]
    for name, out in failed:
        print(f"\n---- {name} ----\n" + "\n".join(out.splitlines()[-25:]))
    print(
        "\nno check failed: now the checklist review, then request the review"
        if not failed
        else f"\n{len(failed)} check(s) failed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
