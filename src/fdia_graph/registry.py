"""The dataset registry: the built-in (downloadable) datasets and the locally generated ones.

A built-in dataset is an asset of a GitHub Release: one timeline per system and its operating-point
pool, as HDF5. The registry knows the sha256 of the releases it pins. A dataset made by
`fdia_graph.generate()` is recorded in a small JSON file under the cache directory, so it loads by
name like a built-in.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Union

from .constants.files import LOCAL_INDEX
from .constants.releases import ALIASES, BUILTIN, DATA_TAG_PREFIX, PINNED_RELEASE, REPO, SHA256
from .models.assets import AssetSpec
from .models.inputs import DatasetName, ReleaseName, SystemRef

# where downloads and the local-datasets JSON are kept; FDIA_GRAPH_CACHE overrides
CACHE_DIR = os.environ.get("FDIA_GRAPH_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "fdia_graph"))
# the data release `load(release=None)` reads; FDIA_GRAPH_RELEASE overrides
_RELEASE = os.environ.get("FDIA_GRAPH_RELEASE", PINNED_RELEASE)


def system_id(system: Union[str, int]) -> int:
    """The bus count of a ladder system from any spelling: "ieee118", "IEEE118", "118" or 118 -> 118."""
    return SystemRef(system).number


def release_tuple(release: str) -> tuple[int, ...]:
    """ "v0.8.0", "0.8.0" or "data-v0.8.0" -> (0, 8, 0), for ordering data releases."""
    return ReleaseName(release).numbers


def release_name(release: str) -> str:
    """The short name of a data release ("v0.8.0"), from the name or its GitHub tag."""
    t = release_tuple(release)
    return "v" + ".".join(str(x) for x in t)


def release_tag(release: str) -> str:
    """The GitHub tag that carries a data release's assets: "data-v0.8.0". The bare "v0.x.y" tags
    belong to the package versions."""
    return DATA_TAG_PREFIX + release_name(release)


def dataset_file(C: int) -> str:
    """The dataset asset of a system: its timeline."""
    return f"timeline_ieee{C}.h5"


def pool_spec(system: Union[str, int], release: Optional[str] = None) -> AssetSpec:
    """The operating-point pool of a system at a data release: HDF5, dataset `X`."""
    C = system_id(system)
    tag = release_tag(release_name(release or _RELEASE))
    return AssetSpec("builtin", f"pool{C}", file=f"pool_ieee{C}.h5", release=tag, repo=REPO, system=C)


# the index of locally generated datasets: name -> path and meta
_LOCAL_JSON = os.path.join(CACHE_DIR, LOCAL_INDEX)


def _load_local() -> dict[str, dict]:
    # a missing or corrupt index reads as no local datasets, so load() never fails on it
    if os.path.exists(_LOCAL_JSON):
        try:
            return json.load(open(_LOCAL_JSON))
        except Exception:
            return {}
    return {}


def _save_local(d: dict[str, dict]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(d, open(_LOCAL_JSON, "w"), indent=2)


def register_local(name: str, path: str, meta: Optional[dict] = None) -> str:
    """Register a locally generated .h5 under `name` so load(name) finds it."""
    local = _load_local()
    # absolute, so load() works from any working directory
    local[name] = {"path": os.path.abspath(path), "meta": meta or {}}
    _save_local(local)
    return name


def list_datasets() -> dict[str, str]:
    """{name: "builtin" or "local"} of everything loadable by name."""
    out = {k: "builtin" for k in BUILTIN}
    out.update({k: "local" for k in _load_local()})  # local entries may shadow a builtin name
    return out


def resolve(name: Union[str, int], release: Optional[str] = None) -> AssetSpec:
    """The AssetSpec of a name or alias (kind "builtin" or "local"). A built-in name matches in any letter
    case and is shadowed by a local registration of it; any other local name matches exactly. For a
    built-in, `release` None is the pinned release and a name ("v0.8.0" or "data-v0.8.0") that exact one,
    and the spec's `release` is its GitHub tag (`release_tag`); a local dataset ignores `release`."""
    local = _load_local()
    name = DatasetName(name, frozenset(local), frozenset(BUILTIN), ALIASES).key  # "118"/118 -> "ieee118"
    if name in local:
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
