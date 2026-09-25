"""How files are found: where a dataset comes from, where its bytes are fetched from, and one line
of the N-1 candidate list."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

from .base import Bundle


@dataclass(frozen=True, eq=False)
class AssetSpec(Bundle):
    """Where a dataset comes from: a built-in release asset (kind "builtin": file, release, repo,
    optional pinned sha256) or a locally generated file (kind "local": path, meta). Indexable like
    the dict it replaces (spec["release"], spec.get("sha256"))."""

    kind: str
    name: str
    file: Optional[str] = None
    release: Optional[str] = None
    repo: Optional[str] = None
    sha256: Optional[str] = None
    system: Optional[int] = None
    path: Optional[str] = None
    meta: Optional[dict[str, Any]] = None


class DownloadTarget(NamedTuple):
    """Where the bytes of a release asset are fetched from: the URL and the request headers."""

    url: str
    headers: dict[str, str]


@dataclass(frozen=True, eq=False)
class LineCandidate(Bundle):
    """One line of `line_outage_candidates`: its pandapower index and branch position, terminals,
    name, intact-case active flow, and, when rejected, the reason."""

    line: int  # pandapower line index (N-1 timeline generation is disabled; the engine keeps `outage`)
    pos: int  # branch position in edge_index
    from_bus: int  # from-end bus
    to_bus: int  # to-end bus
    name: str  # line name from the case, or line<idx>
    base_flow_mw: float  # active flow in the intact case, MW
    reason: Optional[str] = None  # why the line was rejected (islands the grid), rejected list only
