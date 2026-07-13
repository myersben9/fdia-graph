"""Fetch built-in shards from the GitHub Release and cache them under ~/.cache/fdia_graph.

Private-repo support: this dataset is published to a PRIVATE GitHub repo (group access), whose release
assets are not publicly downloadable. Set a GitHub token in $FDIA_GRAPH_TOKEN (or $GITHUB_TOKEN) — a
fine-grained PAT with read access to the repo — and the downloader authenticates via the GitHub API asset
endpoint. Public repos need no token. Group members each set their own token once.
"""
import hashlib, os, requests
from tqdm import tqdm
from .registry import CACHE_DIR


def _token():
    return os.environ.get("FDIA_GRAPH_TOKEN") or os.environ.get("GITHUB_TOKEN")


def _asset_url(spec, session):
    """Resolve a release asset to a download URL. For PRIVATE repos we must go through the authenticated
    GitHub API (find the asset id under the release, then GET it with Accept: octet-stream). For public
    repos the plain browser download URL works without auth."""
    tok = _token()
    if not tok:
        return f"https://github.com/{spec['repo']}/releases/download/{spec['release']}/{spec['file']}", {}
    hdr = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    rel = session.get(f"https://api.github.com/repos/{spec['repo']}/releases/tags/{spec['release']}", headers=hdr, timeout=30)
    rel.raise_for_status()
    asset = next((a for a in rel.json()["assets"] if a["name"] == spec["file"]), None)
    if asset is None:
        raise FileNotFoundError(f"asset {spec['file']} not found in release {spec['release']} of {spec['repo']}")
    return asset["url"], {"Authorization": f"Bearer {tok}", "Accept": "application/octet-stream"}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_local(spec):
    """Given a registry spec, return a local path to the .h5, downloading it if needed.
    Built-in -> GitHub release asset (cached, sha-verified when known). Local -> the registered path."""
    if spec["kind"] == "local":
        p = spec["path"]
        if not os.path.exists(p):
            raise FileNotFoundError(f"local dataset '{spec['name']}' missing at {p} (was it moved/deleted?)")
        return p

    os.makedirs(CACHE_DIR, exist_ok=True)
    dest = os.path.join(CACHE_DIR, f"{spec['release']}_{spec['file']}")
    if os.path.exists(dest) and (spec.get("sha256") is None or _sha256(dest) == spec["sha256"]):
        return dest

    tmp = dest + ".part"
    with requests.Session() as session:
        # resolve the download URL: authenticated API asset endpoint for private repos, plain URL for public.
        # keep the whole streamed download inside the session so its connection pool stays alive.
        url, dl_headers = _asset_url(spec, session)
        with session.get(url, headers=dl_headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"↓ {spec['file']}") as bar:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk); bar.update(len(chunk))
    if spec.get("sha256") and _sha256(tmp) != spec["sha256"]:
        os.remove(tmp)
        raise IOError(f"checksum mismatch for {spec['file']} — download corrupted, please retry")
    os.replace(tmp, dest)
    return dest
