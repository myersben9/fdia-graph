"""Publish data-release assets to a GitHub release (CONTRIBUTING.md, "Releasing", data releases).

    python tools/upload_assets.py vX.Y.Z notes.md file1.h5 [file2.h5 ...]

Creates the release for the tag when it does not exist yet (a lightweight tag on the current
origin/main, marked as a data release in its notes) or reuses it, and uploads every file whose
name the release does not carry yet, so an interrupted upload can be rerun. An asset already on
the release is never replaced: a published file is immutable, cut a new release instead. The
token comes from the Git Credential Manager like `tools/pr.py`; the data files themselves never
enter git.
"""

from __future__ import annotations

import os
import subprocess
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr import BASE, api, token  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from fdia_graph.registry import release_tag  # noqa: E402

UPLOADS = BASE.replace("https://api.github.com", "https://uploads.github.com")


def release_for(name: str, notes_file: str) -> dict:
    """The release of the data release `name` (its tag from `registry.release_tag`, so a package tag
    is never touched), created on origin/main when missing; an existing tag is reused only when it
    is a data tag."""
    tag = release_tag(name)
    r = requests.get(
        f"{BASE}/releases/tags/{tag}", headers={"Authorization": f"Bearer {token()}"}, timeout=60
    )
    if r.status_code == 200:
        if not tag.startswith("data-") and tag not in ("v0.7.1", "v0.7.2"):
            raise SystemExit(f"{tag} is a package release; data releases are tagged data-v<x.y.z>")
        print(f"release {tag} exists, reusing")
        return r.json()
    if r.status_code != 404:  # only "no such release" means create; auth or server errors stop here
        r.raise_for_status()
    sha = subprocess.run(
        ["git", "rev-parse", "origin/main"], capture_output=True, text=True, check=True
    ).stdout.strip()
    with open(notes_file, encoding="utf8") as fh:
        body = fh.read()
    print(f"creating release {tag} at {sha[:9]}")
    return api("POST", "/releases", json=dict(tag_name=tag, target_commitish=sha, name=tag, body=body))


def upload(rel: dict, path: str) -> None:
    name = os.path.basename(path)
    have = {a["name"] for a in rel.get("assets", [])}
    if name in have:
        print(f"  {name}: already on the release, skipped")
        return
    size = os.path.getsize(path)
    print(f"  {name}: uploading {size / 1e6:.0f} MB ...", flush=True)
    with open(path, "rb") as f:
        r = requests.post(
            f"{UPLOADS}/releases/{rel['id']}/assets",
            params={"name": name},
            headers={
                "Authorization": f"Bearer {token()}",
                "Content-Type": "application/octet-stream",
                "Content-Length": str(size),
            },
            data=f,
            timeout=3600,
        )
    if r.status_code >= 300:
        raise SystemExit(f"upload of {name} failed: {r.status_code} {r.text[:300]}")
    print(f"  {name}: done")


def main(argv: list[str]) -> None:
    if len(argv) < 3:
        raise SystemExit(__doc__)
    tag, notes, files = argv[0], argv[1], argv[2:]
    rel = release_for(tag, notes)
    for p in files:
        upload(rel, p)
        rel = api("GET", f"/releases/{rel['id']}")  # refresh the asset list after each upload


if __name__ == "__main__":
    main(sys.argv[1:])
