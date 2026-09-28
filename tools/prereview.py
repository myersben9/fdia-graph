"""Every check a pull request must pass, run locally in one command before the review is requested
(CONTRIBUTING.md, "The pull request").

    python tools/prereview.py                  # every gate, against origin/main
    python tools/prereview.py --base main      # a different base for the changed-lines gates
    python tools/prereview.py --fast           # skip the test suite (the gates only)

It runs the gates CI runs (formatting, lint, types, readability, the generated diagrams and data
dictionary) and the strict test suite on this checkout's own source (`PYTHONPATH=src`, so an
editable install of another checkout cannot stand in for it), plus checks CI does not run, each one a kind of finding reviews kept raising
(`docs/reference/REVIEW_CHECKLIST.md`):

- **changelog**: a change under `src/` adds an entry, and there is exactly one `## Unreleased`.
- **rendered diagrams**: a changed `docs/figures/diagrams/*.mmd` has its `.png` and `.svg` changed too.
- **cited paths**: every repository path a changed Markdown file cites exists.
- **vacuous tests**: no `or True` / `assert True` in `tests/`.
- **integer fields**: every `int` field of a model in `models/config.py` or `models/inputs.py`
  carries the `Integer()` rule.

Exit status 0 only when everything passes. The review checklist in
`docs/reference/REVIEW_CHECKLIST.md` covers what no tool can check.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def changed(base: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    staged = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout
    return sorted(set(out.split()) | set(staged.split()))


def _read(rel: str) -> str:
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


def _changelog(files: list[str]) -> tuple[bool, str]:
    heads = len(re.findall(r"^## Unreleased", _read("CHANGELOG.md"), re.M))
    if heads > 1:
        return False, f"CHANGELOG.md has {heads} '## Unreleased' sections"
    src = [f for f in files if f.startswith("src/")]
    if src and "CHANGELOG.md" not in files:
        return False, f"{len(src)} file(s) under src/ changed but CHANGELOG.md did not"
    return True, ""


_CITED = re.compile(
    r"`((?:src|docs|tools|tests|examples|scripts)/[\w./-]+\.(?:py|md|json|png|svg|npz|csv|mmd|yml))`"
)


def _cited_paths(files: list[str]) -> tuple[bool, str]:
    missing = [
        f"{f}: {path}"
        for f in files
        if f.endswith(".md") and os.path.exists(os.path.join(ROOT, f))
        for path in _CITED.findall(_read(f))
        if not os.path.exists(os.path.join(ROOT, path))
    ]
    return not missing, "missing: " + "; ".join(missing[:10])


def _vacuous_tests() -> tuple[bool, str]:
    hits = [
        f"tests/{name}:{i}"
        for name in sorted(os.listdir(os.path.join(ROOT, "tests")))
        if name.endswith(".py")
        for i, line in enumerate(_read(f"tests/{name}").splitlines(), 1)
        if re.search(r"\bor True\b|\bassert True\b", line)
    ]
    return not hits, "vacuous assertions: " + ", ".join(hits)


def _integer_fields() -> tuple[bool, str]:
    """Every `int` field of a model in models/config.py or models/inputs.py carries `Integer()`,
    directly or through a module-level alias (`Count = Annotated[int, Integer(), AtLeast(1)]`)."""
    bare = []
    for rel in ("src/fdia_graph/models/config.py", "src/fdia_graph/models/inputs.py"):
        if not os.path.exists(os.path.join(ROOT, rel)):
            continue
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
    return not bare, "int fields without Integer(): " + "; ".join(bare)


def repo_checks(files: list[str]) -> list[tuple[str, bool, str]]:
    """The checks CI does not run."""
    out = []
    ok, detail = _changelog(files)
    out.append(("changelog", ok, detail))
    for name, check in (
        ("cited paths", lambda: _cited_paths(files)),
        ("vacuous tests", _vacuous_tests),
        ("integer fields", _integer_fields),
    ):
        ok, detail = check()
        out.append((name, ok, detail))
    stale = [
        f
        for f in files
        if f.startswith("docs/figures/diagrams/")
        and f.endswith(".mmd")
        and not {f[:-4] + ".png", f[:-4] + ".svg"} <= set(files)
    ]
    out.append(("rendered diagrams", not stale, "" if not stale else "re-render: " + ", ".join(stale)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--fast", action="store_true", help="skip the test suite")
    a = ap.parse_args()
    results: list[tuple[str, bool, str]] = []
    for gate in gates(a.base):
        if gate.slow and a.fast:
            continue
        ok, out, secs = run(gate)
        print(f"{'PASS' if ok else 'FAIL'}  {gate.name:16s} {secs:6.1f}s")
        results.append((gate.name, ok, out))
    for name, ok, detail in repo_checks(changed(a.base)):
        print(f"{'PASS' if ok else 'FAIL'}  {name:16s}")
        results.append((name, ok, detail))
    failed = [(n, out) for n, ok, out in results if not ok]
    for name, out in failed:
        print(f"\n---- {name} ----\n" + "\n".join(out.splitlines()[-25:]))
    print(
        "\nall checks pass: now the checklist review, then request the review"
        if not failed
        else f"\n{len(failed)} check(s) failed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
