"""Fetch the built-in datasets from the GitHub Release and cache them under ~/.cache/fdia_graph.

The repository is public, so a download needs no account. For a private fork or a pre-release, set a
token in $FDIA_GRAPH_TOKEN (or $GITHUB_TOKEN): the download then goes through the GitHub API.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import TYPE_CHECKING, Optional

import requests

if TYPE_CHECKING:
    from .models.assets import AssetSpec
from tqdm import tqdm

from .models.assets import DownloadTarget
from .registry import CACHE_DIR


def _token() -> Optional[str]:
    # FDIA_GRAPH_TOKEN wins (a token scoped to this repository), else GITHUB_TOKEN; None: public download
    return os.environ.get("FDIA_GRAPH_TOKEN") or os.environ.get("GITHUB_TOKEN")


def _asset_url(spec: AssetSpec, session: requests.Session) -> DownloadTarget:
    """Where a release asset is downloaded from: the public download URL with no token, else the
    asset's GitHub API URL with the token in its headers."""
    tok = _token()
    if not tok:
        return DownloadTarget(
            f"https://github.com/{spec['repo']}/releases/download/{spec['release']}/{spec['file']}", {}
        )
    # with a token: the public URL of a private repository answers 404, so go through the REST API
    hdr = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    rel = session.get(
        f"https://api.github.com/repos/{spec['repo']}/releases/tags/{spec['release']}",
        headers=hdr,
        timeout=30,
    )
    rel.raise_for_status()  # 401/403: bad token; 404: no such release
    asset = next((a for a in rel.json()["assets"] if a["name"] == spec["file"]), None)
    if asset is None:
        raise FileNotFoundError(
            f"asset {spec['file']} not found in release {spec['release']} of {spec['repo']}"
        )
    # Accept: octet-stream makes the API answer with the file's bytes, not its JSON metadata
    return DownloadTarget(
        asset["url"], {"Authorization": f"Bearer {tok}", "Accept": "application/octet-stream"}
    )


def _sha256(path: str) -> str:
    # read in 1 MiB chunks, so a multi-GB file is never held in memory
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_to_file(
    session: requests.Session, url: str, headers: dict[str, str], path: str, label: str
) -> None:
    """Stream the asset into `path` with a progress bar. An HTTP error is raised before any byte is
    written."""
    with session.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))  # 0 when missing: the bar shows bytes, no percent
        with open(path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"↓ {label}") as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def ensure_local(spec: AssetSpec) -> str:
    """The local path of a registry spec's .h5 file. A built-in is downloaded from its GitHub release
    if not cached, and checked against its sha256 when the registry knows it."""
    if spec["kind"] == "local":
        p = spec["path"]
        if not os.path.exists(p):
            raise FileNotFoundError(f"local dataset '{spec['name']}' missing at {p} (was it moved/deleted?)")
        return p

    os.makedirs(CACHE_DIR, exist_ok=True)
    # the release tag is in the cached file's name, so two releases never share a file
    dest = os.path.join(CACHE_DIR, f"{spec['release']}_{spec['file']}")
    # a cached file is reused when its sha256 matches (or none is known); a corrupt one is downloaded again
    if os.path.exists(dest) and (spec.get("sha256") is None or _sha256(dest) == spec["sha256"]):
        return dest

    # download to a ".part" file unique to this process: a crash leaves no truncated file a later run
    # trusts, and two jobs fetching the same asset never write into one file
    fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, prefix=f"{spec['release']}_{spec['file']}.", suffix=".part")
    os.close(fd)
    try:
        with requests.Session() as session:
            target = _asset_url(spec, session)
            _stream_to_file(session, target.url, target.headers, tmp, spec["file"])
        if spec.get("sha256") and _sha256(tmp) != spec["sha256"]:
            raise OSError(f"checksum mismatch for {spec['file']} — download corrupted, please retry")
        _install(tmp, dest, spec.get("sha256"))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest


def _install(tmp: str, dest: str, sha256: Optional[str] = None) -> None:
    """Rename a complete, verified download to `dest`, so `dest` only ever exists whole. When another
    process installed it first and holds it open (Windows refuses the rename), theirs is kept, unless
    the asset has a `sha256` and the file does not match it."""
    try:
        os.replace(tmp, dest)
    except PermissionError:
        if not os.path.exists(dest) or (sha256 is not None and _sha256(dest) != sha256):
            raise
