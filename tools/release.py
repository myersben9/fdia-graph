"""Cut a package release after the version-bump PR has merged (CONTRIBUTING.md, "Releasing").

    python tools/release.py vX.Y.Z notes.md

Checks that main is checked out, clean and at origin/main, that `pyproject.toml` and
`fdia_graph.__version__` both say X.Y.Z, then tags the commit, pushes the tag, creates the GitHub
release with the notes, and waits for `publish.yml` to upload to PyPI. Refuses to move an
existing tag: a released version is never re-cut, bump again instead.
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


def versions() -> tuple:
    py = re.search(
        r'^version = "([^"]+)"', open(f"{ROOT}/pyproject.toml", encoding="utf8").read(), re.M
    ).group(1)
    init = re.search(
        r'^__version__ = "([^"]+)"', open(f"{ROOT}/src/fdia_graph/__init__.py", encoding="utf8").read(), re.M
    ).group(1)
    return py, init


def main(tag: str, notes_file: str) -> None:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise SystemExit(f"tag must look like v0.17.0, got {tag}")
    want = tag[1:]
    py, init = versions()
    if py != want or init != want:
        raise SystemExit(
            f"pyproject says {py}, __init__ says {init}, tag says {want}: merge the bump PR first"
        )
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
