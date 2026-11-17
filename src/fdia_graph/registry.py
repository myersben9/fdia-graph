"""Dataset registry: built-in (downloadable) datasets + locally generated ones.

Built-in datasets live as assets on a GitHub Release: one timeline per system and its operating-point
pool as HDF5, from data release v0.8.0 (the record shards of earlier releases are read by fdia-graph
0.21 and earlier). The ladder names them and the registry knows the sha256 of the releases it pins. Locally generated datasets (via
`fdia_graph.generate()`) are recorded in a small JSON under the cache dir so they are loadable by
name exactly like the built-ins.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Union

from .constants.files import LOCAL_INDEX
from .constants.releases import ALIASES, BUILTIN, DATA_TAG_PREFIX, PINNED_RELEASE, REPO, SHA256
from .models.assets import AssetSpec
from .models.inputs import DatasetName, ReleaseName, SystemRef

# Cache dir for downloaded shards + the local-datasets JSON; override via FDIA_GRAPH_CACHE.
CACHE_DIR = os.environ.get("FDIA_GRAPH_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "fdia_graph"))
# The pinned data release that `load(release=None)` reads; FDIA_GRAPH_RELEASE overrides.
_RELEASE = os.environ.get("FDIA_GRAPH_RELEASE", PINNED_RELEASE)


def system_id(system: Union[str, int]) -> int:
    """Bus count of a ladder system from either spelling: "ieee118", "IEEE118", "118" or 118 -> 118.

    Every public entry point accepts both forms, so this is the one place that parses them; the
    engine, generators and profiles all go through it.
    """
    return SystemRef(system).number


def release_tuple(release: str) -> tuple[int, ...]:
    """ "v0.8.0", "0.8.0" or "data-v0.8.0" -> (0, 8, 0), for ordering data releases."""
    return ReleaseName(release).numbers


def release_name(release: str) -> str:
    """The short name of a data release ("v0.8.0"), from the name or its GitHub tag."""
    t = release_tuple(release)
    return "v" + ".".join(str(x) for x in t)


def release_tag(release: str) -> str:
    """The GitHub tag that carries a data release's assets. Package versions own the bare "v0.x.y"
    tags (PyPI 0.8.0 is `v0.8.0`), so a data release lives at "data-v0.8.0"."""
    return DATA_TAG_PREFIX + release_name(release)


def dataset_file(C: int) -> str:
    """The dataset asset of a system: its timeline."""
    return f"timeline_ieee{C}.h5"


def pool_spec(system: Union[str, int], release: Optional[str] = None) -> AssetSpec:
    """The operating-point pool of a system at a data release: HDF5, dataset `X`."""
    C = system_id(system)
    rel = release_name(release or _RELEASE)
    return AssetSpec(
        "builtin", f"pool{C}", file=f"pool_ieee{C}.h5", release=release_tag(rel), repo=REPO, system=C
    )


# On-disk index of locally generated datasets (name -> path + meta), alongside the cached shards.
_LOCAL_JSON = os.path.join(CACHE_DIR, LOCAL_INDEX)


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
    then serves as "ieee14"), as `list_datasets` reports. A built-in name is one name in any case,
    so "IEEE14" is shadowed exactly like "ieee14"; other local names are matched exactly. For built-ins, `release`: None -> the
    pinned _RELEASE; an explicit name ("v0.8.0", or its tag "data-v0.8.0") -> that exact release
    (reproducible pin), with its sha256 when
    the registry knows it, and the GitHub tag the assets live under (`release_tag`) in `release`.
    Local datasets live at a fixed path, so `release` is ignored for them.
    """
    local = _load_local()
    name = DatasetName(name, frozenset(local), frozenset(BUILTIN), ALIASES).key  # "118"/118 -> "ieee118"
    if name in local:  # a local registration shadows a built-in name, as list_datasets says
        return AssetSpec("local", name, path=local[name]["path"], meta=local[name].get("meta"))
    if name in BUILTIN:
        C = BUILTIN[name]["system"]
        rel = release_name(release or _RELEASE)
        return AssetSpec(
            "builtin",
            name,
            file=dataset_file(C),
            release=release_tag(rel),
            repo=REPO,
            sha256=SHA256.get(rel, {}).get(name),
            system=C,
        )
    raise KeyError(f"unknown dataset '{name}'. Known: {sorted(list_datasets())}")
