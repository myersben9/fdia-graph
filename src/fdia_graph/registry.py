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


def system_id(system: Union[str, int]) -> int:
    """Bus count of a ladder system from either spelling: "ieee118", "IEEE118", "118" or 118 -> 118.

    Every public entry point accepts both forms, so this is the one place that parses them; the
    engine, generators, streams and profiles all go through it.
    """
    try:
        return int(str(system).strip().lower().replace("ieee", ""))
    except ValueError as e:
        raise ValueError(f"system must be like 'ieee118' or 118, got {system!r}") from e


# Built-in shards. Each entry is the full spec download.py needs: asset `file`, `release` tag, `repo`,
# expected `sha256` (None = skip verification), and the IEEE `system` size.
BUILTIN = {
    "ieee14": {
        "file": "ml_only_ieee14.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "bd161b3a0a507eccbfcc8172b4839439c29721d2a648e37a681469386af6c867",
        "system": 14,
    },
    "ieee30": {
        "file": "ml_only_ieee30.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "df491a7e32906a570caa6350c2336061d6f3b670ab890b67235c9147aae17c0d",
        "system": 30,
    },
    "ieee57": {
        "file": "ml_only_ieee57.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "d631aee4bf88ff62c8cc42ba25080e613bb776b36c2fea7019ef9ba6aa1e1d5d",
        "system": 57,
    },
    "ieee89": {
        "file": "ml_only_ieee89.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "c34031ded619be874859f4fb662c2dba644bfdf3039e27a24ec18ea90ae55464",
        "system": 89,
    },
    "ieee118": {
        "file": "ml_only_ieee118.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "a8ef90092815a6888f89b0cca3bfe75c44f20fd8b34aaf49b5fc787e53f0bb73",
        "system": 118,
    },
    "ieee145": {
        "file": "ml_only_ieee145.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "ccdb6009113bb5aff30ae2e554a5bc559f39751898cbb995422aae134008155e",
        "system": 145,
    },
    "ieee200": {
        "file": "ml_only_ieee200.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "f75c1e7a32a536f157d812d6d89d6012930c63ceda9041524dbcbae8790d4d06",
        "system": 200,
    },
    "ieee300": {
        "file": "ml_only_ieee300.h5",
        "release": _RELEASE,
        "repo": _REPO,
        "sha256": "d8d8055588dfa3a39b221404604ac8ae8c0ec245074ba501788371401a026c29",
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
