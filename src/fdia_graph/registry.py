"""Dataset registry: built-in (downloadable) shards + locally generated ones.

Built-in datasets live as assets on a GitHub Release; each entry gives the release tag, filename, and a
sha256 (filled once the assets are uploaded). Locally generated datasets (via `fdia_graph.generate()`) are
recorded in a small JSON under the cache dir so they are loadable by name exactly like the built-ins.
"""
import json, os

# Where downloaded .h5 shards (and the local-datasets JSON) are cached on disk. Override the location by
# setting FDIA_GRAPH_CACHE; otherwise default to ~/.cache/fdia_graph (expanduser resolves the home dir).
CACHE_DIR = os.environ.get("FDIA_GRAPH_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "fdia_graph"))
# The GitHub repo that hosts the dataset releases. Private, so API calls need a token (see latest_release).
_REPO = "myersben9/fdia-graph"
# Default pinned release tag. This is the version the SDK falls back to when it can't (or isn't asked to)
# query GitHub for the newest one. Set FDIA_GRAPH_RELEASE to override this default globally without a code
# change (e.g. to pin a whole environment to an older dataset version).
_RELEASE = os.environ.get("FDIA_GRAPH_RELEASE", "v0.6.0")   # newest dataset version (v0.6.0: two-sided attack-plausibility band [~2% noise floor, 20% cap], At exempt; Ad uniform-in-band, Ar bounded, Al distributional load-conserving budget)

# Built-in shards. `sha256` is verified after download when set (filled in when the release is published).
# Each entry is the full spec download.py needs: which asset `file` to fetch, from which `release` tag, on
# which `repo`, its expected `sha256` (None = skip verification for now), and the IEEE `system` size it maps to.
BUILTIN = {
    "ieee14":  {"file": "ml_only_ieee14.h5",  "release": _RELEASE, "repo": _REPO,
                "sha256": "b778513ee0745dd423ca422948b2c426e329d4f3ec4b2814bb4f9a68791590a8", "system": 14},
    "ieee118": {"file": "ml_only_ieee118.h5", "release": _RELEASE, "repo": _REPO,
                "sha256": "9be47b161bc78ce58b5a19598478068d9062e848926bde0278689348b25fff0e", "system": 118},
    "ieee300": {"file": "ml_only_ieee300.h5", "release": _RELEASE, "repo": _REPO,
                "sha256": "66faac385c536a8f4a30c5286c164c32abb736b772877007d122623dfd8eb4d4", "system": 300},
}
# Convenience name aliases so callers can pass a bus count instead of the canonical "ieeeNN" key. Both the
# string ("118") and the int (118) forms map to the same canonical name — resolve() normalizes via this map.
_ALIASES = {"14": "ieee14", "118": "ieee118", "300": "ieee300", 14: "ieee14", 118: "ieee118", 300: "ieee300"}
# On-disk index of locally generated datasets (name -> path + meta), stored alongside the cached shards.
_LOCAL_JSON = os.path.join(CACHE_DIR, "local_datasets.json")


def latest_release(repo=_REPO):
    """Return the newest published release tag on the GitHub repo (for version-controlled datasets).

    Group workflow: every time the datasets change we cut a new release (v0.1.0, v0.2.0, ...). By default
    the SDK pulls the NEWEST release so collaborators always get current data; passing an explicit
    release= to load() pins them to a specific version for reproducibility. Falls back to the built-in
    default tag if the API is unreachable (offline / rate-limited) so loading still works.
    """
    import requests   # imported lazily so merely importing the SDK doesn't require the requests dependency
    # a PRIVATE repo's releases API requires auth, so send the token when one is configured. Look first at the
    # SDK-specific FDIA_GRAPH_TOKEN, then fall back to a generic GITHUB_TOKEN if that's all the env has.
    tok = os.environ.get("FDIA_GRAPH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    # Build the auth header only when we actually have a token; an empty header means an anonymous request
    # (which works for public repos but will 404/403 on this private one).
    hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
    try:
        # GitHub's "/releases/latest" endpoint returns the most recent *published* (non-draft, non-prerelease)
        # release. timeout=10 keeps a hung network from blocking load() indefinitely.
        r = requests.get(f"https://api.github.com/repos/{repo}/releases/latest", headers=hdr, timeout=10)
        r.raise_for_status()        # turn any non-2xx (401 unauth, 404 no releases, 403 rate-limit) into an exception
        return r.json()["tag_name"]  # the JSON body's tag_name is the git tag, e.g. "v0.3.0" — exactly what resolve() wants
    except Exception:
        return _RELEASE      # offline / unauth / no releases yet -> the pinned default tag still works


def _load_local():
    # Read the local-datasets index off disk, returning {} if it doesn't exist yet or is unreadable/corrupt
    # (a broken JSON file should degrade to "no local datasets", never crash a load()).
    if os.path.exists(_LOCAL_JSON):
        try:
            return json.load(open(_LOCAL_JSON))
        except Exception:
            return {}
    return {}


def _save_local(d):
    # Persist the local-datasets index, making sure the cache dir exists first. indent=2 keeps it human-readable.
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(d, open(_LOCAL_JSON, "w"), indent=2)


def register_local(name, path, meta=None):
    """Register a locally generated .h5 under `name` so load(name) finds it."""
    local = _load_local()                                             # load existing entries
    # Store an absolute path (so load() works regardless of CWD) plus optional metadata, keyed by `name`.
    local[name] = {"path": os.path.abspath(path), "meta": meta or {}}
    _save_local(local)                                               # write the updated index back to disk
    return name


def list_datasets():
    """Return {name: 'builtin'|'local'} for everything loadable by name."""
    out = {k: "builtin" for k in BUILTIN}          # start with the hard-coded downloadable shards
    out.update({k: "local" for k in _load_local()})  # then overlay any locally generated ones (may shadow a builtin name)
    return out


def resolve(name, release=None):
    """Map a name/alias to a spec dict: {'kind': 'builtin'|'local', ...}.

    For built-in datasets, `release` picks the version: None -> newest published release (queried live);
    an explicit tag like "v0.2.0" -> that exact release (reproducible pin). Local generated datasets are
    version-agnostic (they live at a fixed path), so `release` is ignored for them.
    """
    # Normalize aliases ("118"/118 -> "ieee118"); a name that's already canonical is returned unchanged.
    name = _ALIASES.get(name, name)
    if name in BUILTIN:
        # Copy the built-in spec and tag it as a downloadable ("builtin") dataset.
        spec = {"kind": "builtin", "name": name, **BUILTIN[name]}
        # Decide the release: an explicit `release` arg pins a reproducible version; otherwise (None) we ask
        # GitHub for the newest one, which itself falls back to _RELEASE if the API is unreachable.
        spec["release"] = release or latest_release(spec["repo"])   # None -> newest; else the pinned tag
        # The pinned sha256 describes the _RELEASE assets ONLY. If the resolved release is any other tag
        # (a newer one from the API, or an older explicit pin), drop the pin rather than hard-fail the
        # integrity check against bytes it was never computed from.
        if spec["release"] != _RELEASE:
            spec["sha256"] = None
        return spec
    # Not a built-in: check the locally generated datasets. `release` is irrelevant here (fixed on-disk path).
    local = _load_local()
    if name in local:
        return {"kind": "local", "name": name, **local[name]}
    # Neither built-in nor local -> unknown; surface the full set of known names to help the caller.
    raise KeyError(f"unknown dataset '{name}'. Known: {sorted(list_datasets())}")
