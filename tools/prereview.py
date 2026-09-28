"""Every check a pull request must pass, run locally in one command before the review is requested
(CONTRIBUTING.md, "The pull request").

    python tools/prereview.py                  # every gate, against origin/main
    python tools/prereview.py --base main      # a different base for the changed-lines gates
    python tools/prereview.py --fast           # skip the test suite (the gates only)

It runs the gates CI runs (formatting, lint, types, readability, the generated diagrams and data
dictionary) and the strict test suite on this checkout's own source (`PYTHONPATH=src`, so an
editable install of another checkout cannot stand in for it), plus checks CI does not run, each one
a kind of finding reviews kept raising (`docs/reference/REVIEW_CHECKLIST.md`). The changed files are
the branch's commits against the base, the uncommitted changes and the untracked files.

- **changelog**: never more than one `## Unreleased`; a change under `src/` needs exactly one, with
  at least one entry, and CHANGELOG.md changed.
- **rendered diagrams**: a changed `docs/figures/diagrams/*.mmd` has its `.png` and `.svg` changed too.
- **cited paths**: every repository path a changed Markdown file cites exists (a file or a folder,
  with or without an extension), in backticks or as a link or image destination (relative to the file; anchors, queries and web links are ignored).
- **vacuous tests**: no `or True` / `assert True` in `tests/`.
- **integer fields**: every `int` field of a model in `models/config.py` or `models/inputs.py`
  carries the `Integer()` rule (SKIP where the checkout has no such models).

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass(frozen=True)
class Gate:
    name: str
    cmd: list[str]
    slow: bool = False


def gates(base: str) -> list[Gate]:
    py = sys.executable
    return [
        Gate("format", [py, "-m", "ruff", "format", "--check", "src/fdia_graph", "tests", "tools"]),
        Gate("lint", [py, "-m", "ruff", "check", "src", "tests", "tools"]),
        Gate("types", [py, "-m", "pyright", "src/fdia_graph"]),
        Gate("readability", [py, "tools/readability.py", "--report", "--check", "--base", base]),
        Gate("class diagrams", [py, "tools/class_diagrams.py", "--check"]),
        Gate("data dictionary", [py, "tools/models_doc.py", "--check"]),
        Gate(
            "tests (strict)",
            [py, "-m", "pytest", "-q", "-x", "-W", "error::DeprecationWarning:fdia_graph", "tests"],
            slow=True,
        ),
    ]


def run(gate: Gate) -> tuple[bool, str, float]:
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src"), "FDIA_FROZEN_STRICT": "1"}
    t0 = time.time()
    p = subprocess.run(
        gate.cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return p.returncode == 0, (p.stdout + p.stderr).strip(), time.time() - t0


def _git_names(*args: str) -> set[str]:
    return set(
        subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    )


def changed(base: str) -> list[str]:
    """The branch's commits against `base`, the uncommitted changes and the untracked files."""
    return sorted(
        _git_names("diff", "--name-only", f"{base}...HEAD")
        | _git_names("diff", "--name-only", "HEAD")
        | _git_names("ls-files", "--others", "--exclude-standard")
    )


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
    src = [f for f in files if f.startswith("src/")]
    if src and heads != 1:
        return FAIL, "a change under src/ needs a '## Unreleased' section in CHANGELOG.md"
    if src and _unreleased_entries(text) == 0:
        return FAIL, "the '## Unreleased' section of CHANGELOG.md has no entry"
    if src and "CHANGELOG.md" not in files:
        return FAIL, f"{len(src)} file(s) under src/ changed but CHANGELOG.md did not"
    return PASS, ""


_BACKTICKED = re.compile(
    r"`((?:src|docs|tools|tests|examples|scripts)/[\w./-]+"
    r"|[A-Z][A-Z_]*\.md|pyproject\.toml|mkdocs\.yml)`"  # and the root files: README.md, pyproject.toml
)
_LINKED = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")


def _targets(md: str, text: str) -> list[str]:
    """The repository paths a Markdown file cites: backticked repository-rooted paths, and link and
    image destinations resolved against the file's folder (web links, mail and anchors skipped)."""
    out = list(_BACKTICKED.findall(_LINKED.sub("", text)))  # a link's text is not a citation; its target is
    for dest in _LINKED.findall(re.sub(r"`[^`\n]*`", "", text)):  # a code span holds no link
        if re.match(r"[a-z][a-z0-9+.-]*:", dest, re.I) or dest.startswith("#"):
            continue
        path = dest.split("#", 1)[0].split("?", 1)[0]
        if path:
            out.append(posixpath.normpath(posixpath.join(posixpath.dirname(md), path)))
    return out


def _cited_paths(files: list[str]) -> tuple[str, str]:
    # a bare name such as `REFERENCES.md` is shorthand for a file of that name somewhere in the tree
    names = {posixpath.basename(f) for f in _git_names("ls-files")}
    missing = [
        f"{f}: {path}"
        for f in files
        if f.endswith(".md") and os.path.exists(os.path.join(ROOT, f))
        for path in _targets(f, _read(f))
        if not os.path.exists(os.path.join(ROOT, path)) and not ("/" not in path and path in names)
    ]
    return (FAIL if missing else PASS), "missing: " + "; ".join(missing[:10])


def _vacuous_tests() -> tuple[str, str]:
    hits = [
        f"tests/{name}:{i}"
        for name in sorted(os.listdir(os.path.join(ROOT, "tests")))
        if name.endswith(".py")
        for i, line in enumerate(_read(f"tests/{name}").splitlines(), 1)
        if re.search(r"\bor True\b|\bassert True\b", line)
    ]
    return (FAIL if hits else PASS), "vacuous assertions: " + ", ".join(hits)


_MODELS = ("src/fdia_graph/models/config.py", "src/fdia_graph/models/inputs.py")


def _integer_fields() -> tuple[str, str]:
    """Every `int` field of a model in models/config.py or models/inputs.py carries `Integer()`,
    directly or through a module-level alias (`Count = Annotated[int, Integer(), AtLeast(1)]`)."""
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
                ann = ast.unparse(node.annotation)
                ann = aliases.get(ann, ann)  # a field typed by an alias is checked as the alias
                # a scalar int setting, bare or in Optional/Annotated; a collection of ints is not one
                if re.match(r"(Annotated\[)?(Optional\[)?int\b", ann) and "Integer()" not in ann:
                    bare.append(f"{rel}:{node.lineno} {node.target.id}: {ann}")
    return (FAIL if bare else PASS), "int fields without Integer(): " + "; ".join(bare)


def _rendered_diagrams(files: list[str]) -> tuple[str, str]:
    stale = [
        f
        for f in files
        if f.startswith("docs/figures/diagrams/")
        and f.endswith(".mmd")
        and not {f[:-4] + ".png", f[:-4] + ".svg"} <= set(files)
    ]
    return (FAIL if stale else PASS), "re-render: " + ", ".join(stale)


def repo_checks(files: list[str]) -> list[tuple[str, str, str]]:
    """The checks CI does not run: (name, PASS/FAIL/SKIP, detail)."""
    return [
        ("changelog", *_changelog(files)),
        ("cited paths", *_cited_paths(files)),
        ("vacuous tests", *_vacuous_tests()),
        ("integer fields", *_integer_fields()),
        ("rendered diagrams", *_rendered_diagrams(files)),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--fast", action="store_true", help="skip the test suite")
    a = ap.parse_args()
    results: list[tuple[str, str, str]] = []
    for gate in gates(a.base):
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
