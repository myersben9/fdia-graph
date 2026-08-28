"""Dataset registry: built-in (downloadable) shards + locally generated ones.

Built-in datasets live as assets on a GitHub Release; each entry gives the release tag, filename, and a
sha256 (filled once the assets are uploaded). Locally generated datasets (via `fdia_graph.generate()`) are
recorded in a small JSON under the cache dir so they are loadable by name exactly like the built-ins.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import json
import os

# Cache dir for downloaded shards + the local-datasets JSON; override via FDIA_GRAPH_CACHE.
CACHE_DIR = os.environ.get("FDIA_GRAPH_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "fdia_graph"))
_REPO = "myersben9/fdia-graph"
# Default pinned release tag, used when GitHub isn't queried for the newest; FDIA_GRAPH_RELEASE overrides.
_RELEASE = os.environ.get("FDIA_GRAPH_RELEASE", "v0.7.2")  # pinned data release (env var overrides)
# One release channel: each tag carries all assets, so streams follow _RELEASE (env var overrides).
STREAM_RELEASE = os.environ.get("FDIA_GRAPH_STREAM_RELEASE", _RELEASE)

# Built-in shards. Each entry is the full spec download.py needs: asset `file`, `release` tag, `repo`,
# expected `sha256` (None = skip verification), and the IEEE `system` size.
BUILTIN = {
    "ieee14": {
        "file": "ml_only_ieee14.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "c19526ba6274d1f843f484a7f275c0688334a92b349b0043871713fd01275195",
        "system": 14,
    },
    "ieee30": {
        "file": "ml_only_ieee30.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "b56d51939c04363143ae6f41f5854bc31d62ce9dbc31742efa6d5145d52b797f",
        "system": 30,
    },
    "ieee57": {
        "file": "ml_only_ieee57.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "6a940f4f8b3e3abe8f11e2af707c250fa8dc7e1247ab2709bb63cd763a1dc36a",
        "system": 57,
    },
    "ieee89": {
        "file": "ml_only_ieee89.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "c390fe41ba974617b1b3a21b141cf72a30bc75221c4c858a70812bc9044cc1d4",
        "system": 89,
    },
    "ieee118": {
        "file": "ml_only_ieee118.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "0ddcb8cc7273cbc375465854368a48d00ed1435a5860b5dc4a08735c311cbb8a",
        "system": 118,
    },
    "ieee145": {
        "file": "ml_only_ieee145.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "d35c1f1f877fe09fde39ae76eb5250959442ea5b6d6b9c16951db2b11c74d206",
        "system": 145,
    },
    "ieee200": {
        "file": "ml_only_ieee200.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "3eba3cc255f85a8b71e7f97905041f1e3521b88dcb162bbf318cdd57e25e0360",
        "system": 200,
    },
    "ieee300": {
        "file": "ml_only_ieee300.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "f893927f401eca0a6aa272e649a64188ea08ba1249ff241b72352e2826a468b6",
        "system": 300,
    },
}
# Aliases so callers can pass a bus count (str or int) instead of the canonical "ieeeNN" key.
_ALIASES = {
    "14": "ieee14",
    "30": "ieee30",
    "57": "ieee57",
    "89": "ieee89",
    "118": "ieee118",
    "145": "ieee145",
    "200": "ieee200",
    "300": "ieee300",
    14: "ieee14",
    30: "ieee30",
    57: "ieee57",
    89: "ieee89",
    118: "ieee118",
    145: "ieee145",
    200: "ieee200",
    300: "ieee300",
}
# On-disk index of locally generated datasets (name -> path + meta), alongside the cached shards.
_LOCAL_JSON = os.path.join(CACHE_DIR, "local_datasets.json")


def latest_release(repo: str = _REPO) -> str:
    """Return the newest published release tag on the GitHub repo (for version-controlled datasets).

    Default (release=None in load()) pulls the NEWEST release so collaborators get current data; an explicit
    release= pins a version for reproducibility. Falls back to the built-in default tag when the API is
    unreachable (offline / rate-limited).
    """
    import requests  # lazy so merely importing the SDK doesn't require requests

    # A private repo's releases API needs auth: SDK-specific FDIA_GRAPH_TOKEN first, then generic GITHUB_TOKEN.
    tok = os.environ.get("FDIA_GRAPH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    hdr = {"Authorization": f"Bearer {tok}"} if tok else {}  # empty header -> anonymous (public repos only)
    try:
        # /releases/latest = most recent published (non-draft, non-prerelease); timeout guards a hung network.
        r = requests.get(f"https://api.github.com/repos/{repo}/releases/latest", headers=hdr, timeout=10)
        r.raise_for_status()
        return r.json()["tag_name"]
    except Exception:
        return _RELEASE  # offline / unauth / no releases yet -> pinned default


def _load_local() -> Dict[str, Dict]:
    # Read the local-datasets index; missing/corrupt -> {} (degrade to "no local datasets", never crash load()).
    if os.path.exists(_LOCAL_JSON):
        try:
            return json.load(open(_LOCAL_JSON))
        except Exception:
            return {}
    return {}


def _save_local(d: Dict[str, Dict]) -> None:
    # Persist the local-datasets index, ensuring the cache dir exists first.
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(d, open(_LOCAL_JSON, "w"), indent=2)


def register_local(name: str, path: str, meta: Optional[Dict] = None) -> str:
    """Register a locally generated .h5 under `name` so load(name) finds it."""
    local = _load_local()
    # Absolute path so load() works regardless of CWD.
    local[name] = {"path": os.path.abspath(path), "meta": meta or {}}
    _save_local(local)
    return name


def list_datasets() -> Dict[str, str]:
    """Return {name: 'builtin'|'local'} for everything loadable by name."""
    out = {k: "builtin" for k in BUILTIN}
    out.update({k: "local" for k in _load_local()})  # local entries may shadow a builtin name
    return out


def resolve(name: Union[str, int], release: Optional[str] = None) -> Dict:
    """Map a name/alias to a spec dict: {'kind': 'builtin'|'local', ...}.

    For built-ins, `release`: None -> newest published release (queried live); an explicit tag -> that exact
    release (reproducible pin). Local datasets live at a fixed path, so `release` is ignored for them.
    """
    name = _ALIASES.get(name, name)  # "118"/118 -> "ieee118"; canonical unchanged
    if name in BUILTIN:
        spec = {"kind": "builtin", "name": name, **BUILTIN[name]}
        # Pin _RELEASE, not a live "newest tag" query (a partial tag once 404'd fresh loads).
        spec["release"] = release or _RELEASE
        # The pinned sha256 describes the _RELEASE assets only; drop it for any other release rather than
        # fail the integrity check against bytes it was never computed from.
        if spec["release"] != _RELEASE:
            spec["sha256"] = None
        return spec
    local = _load_local()
    if name in local:
        return {"kind": "local", "name": name, **local[name]}
    raise KeyError(f"unknown dataset '{name}'. Known: {sorted(list_datasets())}")
