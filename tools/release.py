"""Cut a package release after the version-bump PR has merged (CONTRIBUTING.md, "Releasing").

    python tools/release.py vX.Y.Z notes.md

Checks that main is checked out, clean and at origin/main, and that both toolkits say X.Y.Z, then
tags the commit, pushes the tag, creates the GitHub release with the notes, and waits for
`publish.yml` to upload to PyPI. Refuses to move an existing tag: a released version is never
re-cut, bump again instead.

The two toolkits are released together at one version: the tag publishes `fdia-graph` and
`fdia-analysis`, so `pyproject.toml`, `fdia_graph.__version__`, `analysis/pyproject.toml` and
`fdia_analysis.__version__` must all say X.Y.Z, and `fdia-analysis` must require exactly
`fdia-graph==X.Y.Z` (the same rule as `tests/test_versions.py`).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr import api  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()


def _find(pattern: str, path: str) -> str:
    found = re.search(pattern, open(f"{ROOT}/{path}", encoding="utf8").read(), re.M)
    if not found:
        raise SystemExit(f"{path} has no line matching {pattern}")
    return found.group(1)


def versions() -> dict[str, str]:
    """Where a version is written -> what it says there: the four version lines of the two toolkits."""
    line, dunder = r'^version = "([^"]+)"', r'^__version__ = "([^"]+)"'
    return {
        "pyproject.toml": _find(line, "pyproject.toml"),
        "fdia_graph.__version__": _find(dunder, "src/fdia_graph/__init__.py"),
        "analysis/pyproject.toml": _find(line, "analysis/pyproject.toml"),
        "fdia_analysis.__version__": _find(dunder, "analysis/src/fdia_analysis/__init__.py"),
    }


def main(tag: str, notes_file: str) -> None:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise SystemExit(f"tag must look like v0.17.0, got {tag}")
    want = tag[1:]
    said = versions()
    if set(said.values()) != {want}:
        raise SystemExit(f"the tag says {want}, the checkout says {said}: merge the bump PR first")
    requirement = _find(r'^\s*"(fdia-graph[^"]*)"', "analysis/pyproject.toml")
    if requirement != f"fdia-graph=={want}":
        raise SystemExit(f"analysis/pyproject.toml requires {requirement}, not fdia-graph=={want}")
    print(f"{tag} publishes fdia-graph and fdia-analysis")
    if git("branch", "--show-current") != "main" or git("status", "--porcelain"):
        raise SystemExit("check out a clean main first")
    git("fetch", "origin")
    if git("rev-parse", "HEAD") != git("rev-parse", "origin/main"):
        raise SystemExit("local main is not at origin/main; pull first")
    if git("tag", "-l", tag):
        raise SystemExit(f"{tag} already exists; a released version is never re-cut")
    notes = (
        open(notes_file, encoding="utf8").read().strip()
    )  # before the tag, so a bad file leaves nothing behind
    if not notes:
        raise SystemExit(f"{notes_file} is empty; write the release notes first")
    git("tag", "-a", tag, "-m", f"{want}: see CHANGELOG.md")
    git("push", "origin", tag)
    rel = api(
        "POST",
        "/releases",
        json={"tag_name": tag, "name": tag, "body": notes, "draft": False, "prerelease": False},
    )
    print("release:", rel["html_url"])
    for _ in range(40):  # publish.yml usually finishes within two minutes
        runs = api("GET", "/actions/workflows/publish.yml/runs?per_page=1")["workflow_runs"]
        if runs and runs[0]["head_branch"] == tag and runs[0]["status"] == "completed":
            print("publish workflow:", runs[0]["conclusion"], runs[0]["html_url"])
            if runs[0]["conclusion"] != "success":
                raise SystemExit(
                    f"publish workflow {runs[0]['conclusion']}: the tag and release exist, PyPI has nothing"
                )
            break
        time.sleep(15)
    else:
        raise SystemExit("publish workflow did not finish in ten minutes; check the Actions tab")
    for _ in range(20):
        try:
            d = json.load(
                urllib.request.urlopen(f"https://pypi.org/pypi/fdia-graph/{want}/json?t={int(time.time())}")
            )
            print("on PyPI:", [f["filename"] for f in d["urls"]])
            return
        except Exception:
            time.sleep(15)
    raise SystemExit(f"{want} is not on PyPI five minutes after a successful publish run; check pypi.org")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
