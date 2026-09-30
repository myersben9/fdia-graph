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
