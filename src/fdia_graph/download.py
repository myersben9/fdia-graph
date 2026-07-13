"""Fetch built-in shards from the GitHub Release and cache them under ~/.cache/fdia_graph.

Private-repo support: this dataset is published to a PRIVATE GitHub repo (group access), whose release
assets are not publicly downloadable. Set a GitHub token in $FDIA_GRAPH_TOKEN (or $GITHUB_TOKEN) — a
fine-grained PAT with read access to the repo — and the downloader authenticates via the GitHub API asset
endpoint. Public repos need no token. Group members each set their own token once.
"""
# hashlib -> sha256 integrity check; os -> env vars + filesystem; requests -> HTTP(S) to GitHub.
import hashlib, os, requests
# tqdm draws the download progress bar (bytes streamed) while we pull the shard.
from tqdm import tqdm
# CACHE_DIR (default ~/.cache/fdia_graph) is where verified shards live; registry.py owns it.
from .registry import CACHE_DIR


def _token():
    # Resolve the GitHub auth token. Lookup order is deliberate: the SDK-specific
    # $FDIA_GRAPH_TOKEN wins so a user can scope a fine-grained PAT to just this repo
    # without disturbing a broader $GITHUB_TOKEN already in their shell; fall back to
    # $GITHUB_TOKEN for convenience. Returns None if neither is set (public-repo path).
    return os.environ.get("FDIA_GRAPH_TOKEN") or os.environ.get("GITHUB_TOKEN")


def _asset_url(spec, session):
    """Resolve a release asset to a download URL. For PRIVATE repos we must go through the authenticated
    GitHub API (find the asset id under the release, then GET it with Accept: octet-stream). For public
    repos the plain browser download URL works without auth."""
    tok = _token()
    # No token -> assume a PUBLIC repo. The human-facing browser download URL
    # (github.com/.../releases/download/<tag>/<file>) serves the bytes directly, no auth,
    # no API round-trip. Empty {} header dict means "add nothing to the GET".
    if not tok:
        return f"https://github.com/{spec['repo']}/releases/download/{spec['release']}/{spec['file']}", {}
    # PRIVATE repo path. The browser download URL 404s without a session cookie, so we can't use
    # it programmatically — we must go through the authenticated GitHub REST API instead.
    # Step 1: look up the release by tag. `Accept: application/vnd.github+json` asks for the JSON
    # metadata describing the release (including its list of assets and their API urls).
    hdr = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    rel = session.get(f"https://api.github.com/repos/{spec['repo']}/releases/tags/{spec['release']}", headers=hdr, timeout=30)
    rel.raise_for_status()  # 401/403 (bad/insufficient token) or 404 (no such release) -> raise here.
    # Step 2: find the asset whose name matches the shard we want; next(..., None) -> None if absent.
    asset = next((a for a in rel.json()["assets"] if a["name"] == spec["file"]), None)
    if asset is None:
        # The release exists but doesn't contain this file — spec/release mismatch, not an auth issue.
        raise FileNotFoundError(f"asset {spec['file']} not found in release {spec['release']} of {spec['repo']}")
    # Step 3: return the asset's API url (api.github.com/.../releases/assets/<id>), NOT its
    # browser_download_url. Hitting that API url with `Accept: application/octet-stream` is the
    # magic switch that makes GitHub stream the raw bytes (a 302 to a signed URL) instead of
    # returning JSON metadata about the asset. Header carries the token so the private asset authorizes.
    return asset["url"], {"Authorization": f"Bearer {tok}", "Accept": "application/octet-stream"}


def _sha256(path):
    # Stream the file through sha256 in 1 MiB (1 << 20) chunks so we never load a
    # multi-GB shard fully into memory. iter(callable, sentinel) keeps calling f.read
    # until it returns b"" (EOF). Returns the lowercase hex digest to compare against spec["sha256"].
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_local(spec):
    """Given a registry spec, return a local path to the .h5, downloading it if needed.
    Built-in -> GitHub release asset (cached, sha-verified when known). Local -> the registered path."""
    # --- LOCAL branch: a user-registered dataset already on disk. No download, no cache. ---
    # The spec just points at an existing .h5; we verify it's still there and hand back the path.
    if spec["kind"] == "local":
        p = spec["path"]
        if not os.path.exists(p):
            raise FileNotFoundError(f"local dataset '{spec['name']}' missing at {p} (was it moved/deleted?)")
        return p

    # --- BUILT-IN branch: shard must be fetched from the GitHub release and cached. ---
    os.makedirs(CACHE_DIR, exist_ok=True)  # ensure ~/.cache/fdia_graph exists (no error if it already does).
    # Cache filename is prefixed with the release tag: "{release}_{file}". This namespaces shards by
    # release so bumping the dataset version (new tag) can't collide with a stale cached file of the same name.
    dest = os.path.join(CACHE_DIR, f"{spec['release']}_{spec['file']}")
    # Cache hit: reuse the cached file if it exists AND either we have no expected hash to check,
    # or its sha256 matches. (A known-bad/corrupt cached file fails the hash and falls through to re-download.)
    if os.path.exists(dest) and (spec.get("sha256") is None or _sha256(dest) == spec["sha256"]):
        return dest

    # Cache miss: download to a ".part" sidecar first so a crash mid-download never leaves a
    # truncated file at the real `dest` path (which a later run would trust as a complete cache hit).
    tmp = dest + ".part"
    with requests.Session() as session:
        # resolve the download URL: authenticated API asset endpoint for private repos, plain URL for public.
        # keep the whole streamed download inside the session so its connection pool stays alive.
        url, dl_headers = _asset_url(spec, session)
        # stream=True: pull the body incrementally instead of buffering the whole shard in RAM.
        # The `with` closes the response (and its socket) even on error.
        with session.get(url, headers=dl_headers, stream=True, timeout=60) as r:
            r.raise_for_status()  # surface 401/403/404/5xx before we start writing bytes.
            # content-length drives the progress bar total; default 0 if the header is missing
            # (bar then shows bytes-so-far without a percentage rather than crashing).
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"↓ {spec['file']}") as bar:
                # Write each 1 MiB chunk straight to the .part file and advance the bar by its size.
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk); bar.update(len(chunk))
    # Integrity gate: if the spec pins a sha256, verify the freshly downloaded bytes before
    # trusting them. On mismatch, delete the bad .part and fail loudly rather than caching garbage.
    if spec.get("sha256") and _sha256(tmp) != spec["sha256"]:
        os.remove(tmp)
        raise IOError(f"checksum mismatch for {spec['file']} — download corrupted, please retry")
    # Atomic-ish commit: rename .part -> dest only after a full, verified download. os.replace is
    # atomic on the same filesystem, so `dest` only ever exists as a complete, valid shard.
    os.replace(tmp, dest)
    return dest
