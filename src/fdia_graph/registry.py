"""Dataset registry: built-in (downloadable) datasets + locally generated ones.

Built-in datasets live as assets on a GitHub Release; the ladder names them, the release tag picks
the layout (record shards and npz pools before v0.8.0, one timeline per system and HDF5 pools from
v0.8.0) and the registry knows the sha256 of the releases it pins. Locally generated datasets (via
`fdia_graph.generate()`) are recorded in a small JSON under the cache dir so they are loadable by
name exactly like the built-ins.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Union

from .models.assets import AssetSpec  # noqa: F401  re-exported: defined here before the models package

# Cache dir for downloaded shards + the local-datasets JSON; override via FDIA_GRAPH_CACHE.
CACHE_DIR = os.environ.get("FDIA_GRAPH_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "fdia_graph"))
_REPO = "myersben9/fdia-graph"
# Default pinned release tag, used when GitHub isn't queried for the newest; FDIA_GRAPH_RELEASE overrides.
_RELEASE = os.environ.get("FDIA_GRAPH_RELEASE", "v0.8.0")  # pinned data release (env var overrides)
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


# The ladder: canonical name -> bus count. Every entry is downloadable at every data release.
BUILTIN = {f"ieee{C}": {"system": C} for C in (14, 30, 57, 89, 118, 145, 200, 300)}
# Aliases so callers can pass a bus count (str or int) instead of the canonical "ieeeNN" key.
_ALIASES = {
    **{str(C): f"ieee{C}" for C in (14, 30, 57, 89, 118, 145, 200, 300)},
    **{C: f"ieee{C}" for C in (14, 30, 57, 89, 118, 145, 200, 300)},
}
# What each data release ships per system. Releases before v0.8.0 are record shards (`ml_only_ieee{C}.h5`)
# with npz operating-point pools and separate stream files; from v0.8.0 one timeline per system
# (`timeline_ieee{C}.h5`) and the pool as HDF5. `fg.load(..., release="v0.7.2")` still reads the shards.
_TIMELINE_FROM = (0, 8, 0)
# sha256 of the assets, per release; a release not listed downloads unverified.
_SHA256: dict[str, dict[str, str]] = {
    "v0.7.2": {
        "ieee14": "bd161b3a0a507eccbfcc8172b4839439c29721d2a648e37a681469386af6c867",
        "ieee30": "df491a7e32906a570caa6350c2336061d6f3b670ab890b67235c9147aae17c0d",
        "ieee57": "d631aee4bf88ff62c8cc42ba25080e613bb776b36c2fea7019ef9ba6aa1e1d5d",
        "ieee89": "c34031ded619be874859f4fb662c2dba644bfdf3039e27a24ec18ea90ae55464",
        "ieee118": "a8ef90092815a6888f89b0cca3bfe75c44f20fd8b34aaf49b5fc787e53f0bb73",
        "ieee145": "ccdb6009113bb5aff30ae2e554a5bc559f39751898cbb995422aae134008155e",
        "ieee200": "f75c1e7a32a536f157d812d6d89d6012930c63ceda9041524dbcbae8790d4d06",
        "ieee300": "d8d8055588dfa3a39b221404604ac8ae8c0ec245074ba501788371401a026c29",
    },
    "v0.8.0": {
        "ieee14": "c1d3fafcb37e3c2b57a07523fee6bd0a658ed9cd4b77bc6b38697762e13a5a28",
        "ieee30": "9ad2bd82adef5dbd8c08587982c0f50bafe584c8318f13e4b8082795f7418a66",
        "ieee57": "085a5fbf6c0b743f04ba8d290dcd09fb6151e8bf0173250258ef3565cfca8948",
        "ieee89": "47fbf40c061852fe0c39a43105036bd27c2f2b34c8cb52dc31a43204acd91789",
        "ieee118": "bb5fbda40e6643d47938b23b63704ab8bb81d07383678fa394b7efe159cf2dd5",
        "ieee145": "aaa5fc23f93d138cbebb25724d5479b532fab7fdffa2b55738c6018aa936f9e7",
        "ieee200": "be4ad6a6d598aac9b4fe31c05048c50f50086822140c2001ce0ecdad77edb697",
        "ieee300": "b4f08db8b643a58d7050e2079b5890c3acf4f09712ab6efa6cb7f8f165232e7f",
    },
}


_DATA_TAG_PREFIX = "data-"  # GitHub tags of data releases from v0.8.0: "data-v0.8.0"
_BARE_DATA_TAGS = ("v0.7.1", "v0.7.2")  # the two data releases tagged before the namespace existed


def release_tuple(release: str) -> tuple[int, ...]:
    """ "v0.8.0", "0.8.0" or "data-v0.8.0" -> (0, 8, 0), for ordering data releases."""
    m = re.fullmatch(r"(?:data-)?v?(\d+)\.(\d+)\.(\d+)", release.strip())
    if not m:
        raise ValueError(f"a data release is a name like 'v0.8.0', got {release!r}")
    return tuple(int(x) for x in m.groups())


def release_name(release: str) -> str:
    """The short name of a data release ("v0.8.0"), from the name or its GitHub tag."""
    t = release_tuple(release)
    return "v" + ".".join(str(x) for x in t)


def release_tag(release: str) -> str:
    """The GitHub tag that carries a data release's assets. Package versions own the bare "v0.x.y"
    tags (PyPI 0.8.0 is `v0.8.0`), so a data release from v0.8.0 lives at "data-v0.8.0"; the two
    releases tagged before that keep their bare tags."""
    name = release_name(release)
    return name if name in _BARE_DATA_TAGS else _DATA_TAG_PREFIX + name


def is_timeline_release(release: str) -> bool:
    """Whether a data release ships timelines (v0.8.0 and later) rather than record shards."""
    return release_tuple(release) >= _TIMELINE_FROM


def dataset_file(C: int, release: str) -> str:
    """The dataset asset of `system` at `release`."""
    return f"timeline_ieee{C}.h5" if is_timeline_release(release) else f"ml_only_ieee{C}.h5"


def pool_spec(system: Union[str, int], release: Optional[str] = None) -> AssetSpec:
    """The operating-point pool of a system at a data release: HDF5 (dataset `X`) from v0.8.0, npz before."""
    C = system_id(system)
    rel = release_name(release or _RELEASE)
    ext = "h5" if is_timeline_release(rel) else "npz"
    return AssetSpec(
        "builtin", f"pool{C}", file=f"pool_ieee{C}.{ext}", release=release_tag(rel), repo=_REPO, system=C
    )


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
        # every published release; the newest DATA release by version, since package releases share
        # the repository (timeout guards a hung network)
        r = requests.get(
            f"https://api.github.com/repos/{repo}/releases?per_page=100", headers=hdr, timeout=10
        )
        r.raise_for_status()
        return _newest_data_release(r.json()) or _RELEASE
    except Exception:
        return _RELEASE  # offline / unauth / no releases yet -> pinned default


def _newest_data_release(releases: list[dict]) -> Optional[str]:
    """The short name of the newest published data release among GitHub release records, or None:
    package releases share the repository, so only data tags count."""
    tags = [x["tag_name"] for x in releases if not x.get("draft") and not x.get("prerelease")]
    data = [t for t in tags if t.startswith(_DATA_TAG_PREFIX) or t in _BARE_DATA_TAGS]
    return release_name(max(data, key=release_tuple)) if data else None


def _load_local() -> dict[str, dict]:
    # Read the local-datasets index; missing/corrupt -> {} (degrade to "no local datasets", never crash load()).
    if os.path.exists(_LOCAL_JSON):
        try:
            return json.load(open(_LOCAL_JSON))
        except Exception:
            return {}
    return {}


def _save_local(d: dict[str, dict]) -> None:
    # Persist the local-datasets index, ensuring the cache dir exists first.
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(d, open(_LOCAL_JSON, "w"), indent=2)


def register_local(name: str, path: str, meta: Optional[dict] = None) -> str:
    """Register a locally generated .h5 under `name` so load(name) finds it."""
    local = _load_local()
    # Absolute path so load() works regardless of CWD.
    local[name] = {"path": os.path.abspath(path), "meta": meta or {}}
    _save_local(local)
    return name


def list_datasets() -> dict[str, str]:
    """Return {name: 'builtin'|'local'} for everything loadable by name."""
    out = {k: "builtin" for k in BUILTIN}
    out.update({k: "local" for k in _load_local()})  # local entries may shadow a builtin name
    return out


def resolve(name: Union[str, int], release: Optional[str] = None) -> AssetSpec:
    """Map a name/alias to its AssetSpec (kind "builtin" or "local").

    A local registration wins over a built-in of the same name (`fg.generate(system, "ieee14")`
    then serves as "ieee14"), as `list_datasets` reports. For built-ins, `release`: None -> the
    pinned _RELEASE; an explicit name ("v0.8.0", or its tag "data-v0.8.0") -> that exact release
    (reproducible pin), with the file name that release used (see `dataset_file`), its sha256 when
    the registry knows it, and the GitHub tag the assets live under (`release_tag`) in `release`.
    Local datasets live at a fixed path, so `release` is ignored for them.
    """
    name = _ALIASES.get(name, name)  # "118"/118 -> "ieee118"; canonical unchanged
    local = _load_local()
    if name in local:  # a local registration shadows a built-in name, as list_datasets says
        return AssetSpec("local", name, path=local[name]["path"], meta=local[name].get("meta"))
    if name in BUILTIN:
        C = BUILTIN[name]["system"]
        rel = release_name(release or _RELEASE)
        return AssetSpec(
            "builtin",
            name,
            file=dataset_file(C, rel),
            release=release_tag(rel),
            repo=_REPO,
            sha256=_SHA256.get(rel, {}).get(name),
            system=C,
        )
    raise KeyError(f"unknown dataset '{name}'. Known: {sorted(list_datasets())}")
