"""Fetch built-in shards from the GitHub Release and cache them under ~/.cache/fdia_graph.

The repo is PUBLIC, so anonymous download is the normal path. For a private fork or pre-release, set a token
in $FDIA_GRAPH_TOKEN (or $GITHUB_TOKEN) and the downloader authenticates via the GitHub API asset endpoint;
without one it uses the public releases/download URL directly.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import hashlib
import os
import requests
from tqdm import tqdm   # download progress bar
from .registry import CACHE_DIR   # ~/.cache/fdia_graph, owned by registry.py


def _token() -> Optional[str]:
    # SDK-specific FDIA_GRAPH_TOKEN wins (scope a PAT to just this repo), else generic GITHUB_TOKEN; None -> public.
    return os.environ.get("FDIA_GRAPH_TOKEN") or os.environ.get("GITHUB_TOKEN")


def _asset_url(spec: Dict, session: requests.Session) -> Tuple[str, Dict[str, str]]:
    """Resolve a release asset to a download URL. Private repos go through the authenticated GitHub API (find
    the asset id, GET it with Accept: octet-stream); public repos use the plain browser download URL."""
    tok = _token()
    # No token -> PUBLIC: browser download URL serves bytes directly, no auth, no API round-trip.
    if not tok:
        return f"https://github.com/{spec['repo']}/releases/download/{spec['release']}/{spec['file']}", {}
    # PRIVATE: browser URL 404s without a cookie, so go through the REST API.
    # Step 1: look up the release by tag (JSON metadata incl. its assets).
    hdr = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    rel = session.get(f"https://api.github.com/repos/{spec['repo']}/releases/tags/{spec['release']}", headers=hdr, timeout=30)
    rel.raise_for_status()  # 401/403 bad token or 404 no such release
    # Step 2: find the matching asset (None if absent).
    asset = next((a for a in rel.json()["assets"] if a["name"] == spec["file"]), None)
    if asset is None:
        # Release exists but lacks this file -> spec/release mismatch, not auth.
        raise FileNotFoundError(f"asset {spec['file']} not found in release {spec['release']} of {spec['repo']}")
    # Step 3: return the asset's API url with Accept: octet-stream -> GitHub streams raw bytes (302 to a
    # signed URL) instead of JSON metadata; token authorizes the private asset.
    return asset["url"], {"Authorization": f"Bearer {tok}", "Accept": "application/octet-stream"}


def _sha256(path: str) -> str:
    # Stream in 1 MiB chunks (iter(read, b"") stops at EOF) so a multi-GB shard never loads fully into memory.
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_local(spec: Dict) -> str:
    """Given a registry spec, return a local path to the .h5, downloading it if needed.
    Built-in -> GitHub release asset (cached, sha-verified when known). Local -> the registered path."""
    # LOCAL: spec points at an existing .h5; verify it's there and hand back the path.
    if spec["kind"] == "local":
        p = spec["path"]
        if not os.path.exists(p):
            raise FileNotFoundError(f"local dataset '{spec['name']}' missing at {p} (was it moved/deleted?)")
        return p

    # BUILT-IN: fetch from the GitHub release and cache.
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Prefix the cache filename with the release tag so a version bump can't collide with a stale cached file.
    dest = os.path.join(CACHE_DIR, f"{spec['release']}_{spec['file']}")
    # Cache hit: reuse if present and (no expected hash, or sha256 matches); a corrupt file fails and re-downloads.
    if os.path.exists(dest) and (spec.get("sha256") is None or _sha256(dest) == spec["sha256"]):
        return dest

    # Download to a ".part" sidecar so a crash never leaves a truncated file that a later run trusts as complete.
    tmp = dest + ".part"
    with requests.Session() as session:
        # authenticated API asset endpoint for private repos, plain URL for public; keep the download in the session.
        url, dl_headers = _asset_url(spec, session)
        # stream=True pulls incrementally; the `with` closes the response even on error.
        with session.get(url, headers=dl_headers, stream=True, timeout=60) as r:
            r.raise_for_status()  # surface 401/403/404/5xx before writing bytes
            total = int(r.headers.get("content-length", 0))   # 0 if missing -> bar shows bytes-so-far, no %
            with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"↓ {spec['file']}") as bar:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
    # Integrity gate: verify a pinned sha256 before trusting; on mismatch delete the .part and fail loudly.
    if spec.get("sha256") and _sha256(tmp) != spec["sha256"]:
        os.remove(tmp)
        raise IOError(f"checksum mismatch for {spec['file']} — download corrupted, please retry")
    # Atomic rename only after a full, verified download, so `dest` only ever exists as a valid shard.
    os.replace(tmp, dest)
    return dest
