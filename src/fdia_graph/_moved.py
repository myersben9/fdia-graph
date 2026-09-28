"""Names that moved to another module and stay importable from the old one for one minor release.

A module that lost names defines `_MOVED = {old name: (new location, the object)}` and

    def __getattr__(name: str) -> object:
        return moved(__name__, name, _MOVED)

so the old import still works, with a DeprecationWarning that says where the name lives now.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

RETIRES = "0.22"  # the release that drops the old paths: one minor release after the move


def moved(module: str, name: str, table: Mapping[str, tuple[str, object]]) -> object:
    """The object `module.name` now names elsewhere, with a DeprecationWarning; AttributeError when
    the module never had the name."""
    if name not in table:
        raise AttributeError(f"module {module!r} has no attribute {name!r}")
    where, value = table[name]
    warnings.warn(
        f"{module}.{name} moved to {where} and retires in {RETIRES}; use it from there",
        DeprecationWarning,
        stacklevel=3,
    )
    return value
