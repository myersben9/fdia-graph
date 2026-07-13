"""Dataset registry: built-in (downloadable) shards + locally generated ones.

Built-in datasets live as assets on a GitHub Release; each entry gives the release tag, filename, and a
sha256 (filled once the assets are uploaded). Locally generated datasets (via `fdia_graph.generate`) are
recorded in a small JSON under the cache dir so they are loadable by name exactly like the built-ins.
"""
import json, os

CACHE_DIR = os.environ.get("FDIA_GRAPH_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "fdia_graph"))
_REPO = "myersben9/fdia-graph"
_RELEASE = os.environ.get("FDIA_GRAPH_RELEASE", "v0.1.0")

# Built-in shards. `sha256` is verified after download when set (filled in when the release is published).
BUILTIN = {
    "ieee14":  {"file": "ml_only_ieee14.h5",  "release": _RELEASE, "repo": _REPO, "sha256": None, "system": 14},
    "ieee118": {"file": "ml_only_ieee118.h5", "release": _RELEASE, "repo": _REPO, "sha256": None, "system": 118},
    "ieee300": {"file": "ml_only_ieee300.h5", "release": _RELEASE, "repo": _REPO, "sha256": None, "system": 300},
}
_ALIASES = {"14": "ieee14", "118": "ieee118", "300": "ieee300", 14: "ieee14", 118: "ieee118", 300: "ieee300"}
_LOCAL_JSON = os.path.join(CACHE_DIR, "local_datasets.json")


def latest_release(repo=_REPO):
    """Return the newest published release tag on the GitHub repo (for version-controlled datasets).

    Group workflow: every time the datasets change we cut a new release (v0.1.0, v0.2.0, ...). By default
    the SDK pulls the NEWEST release so collaborators always get current data; passing an explicit
    release= to load() pins them to a specific version for reproducibility. Falls back to the built-in
    default tag if the API is unreachable (offline / rate-limited) so loading still works.
    """
    import requests
    # a PRIVATE repo's releases API requires auth, so send the token when one is configured
    tok = os.environ.get("FDIA_GRAPH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
    try:
        r = requests.get(f"https://api.github.com/repos/{repo}/releases/latest", headers=hdr, timeout=10)
        r.raise_for_status()
        return r.json()["tag_name"]
    except Exception:
        return _RELEASE      # offline / unauth / no releases yet -> the pinned default tag still works


def _load_local():
    if os.path.exists(_LOCAL_JSON):
        try:
            return json.load(open(_LOCAL_JSON))
        except Exception:
            return {}
    return {}


def _save_local(d):
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(d, open(_LOCAL_JSON, "w"), indent=2)


def register_local(name, path, meta=None):
    """Register a locally generated .h5 under `name` so load(name) finds it."""
    local = _load_local()
    local[name] = {"path": os.path.abspath(path), "meta": meta or {}}
    _save_local(local)
    return name


def list_datasets():
    """Return {name: 'builtin'|'local'} for everything loadable by name."""
    out = {k: "builtin" for k in BUILTIN}
    out.update({k: "local" for k in _load_local()})
    return out


def resolve(name, release=None):
    """Map a name/alias to a spec dict: {'kind': 'builtin'|'local', ...}.

    For built-in datasets, `release` picks the version: None -> newest published release (queried live);
    an explicit tag like "v0.2.0" -> that exact release (reproducible pin). Local generated datasets are
    version-agnostic (they live at a fixed path), so `release` is ignored for them.
    """
    name = _ALIASES.get(name, name)
    if name in BUILTIN:
        spec = {"kind": "builtin", "name": name, **BUILTIN[name]}
        spec["release"] = release or latest_release(spec["repo"])   # None -> newest; else the pinned tag
        return spec
    local = _load_local()
    if name in local:
        return {"kind": "local", "name": name, **local[name]}
    raise KeyError(f"unknown dataset '{name}'. Known: {sorted(list_datasets())}")
