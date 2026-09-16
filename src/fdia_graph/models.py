"""Bundle: a typed record that still behaves as the dict it used to be.

The package's public functions have always returned dicts of arrays (records, batches, streams,
score tables). A Bundle is a frozen dataclass whose fields are those keys, so a reader sees them
in one place and pyright checks them, and it is also a Mapping, so every existing use keeps
working: ``out["node_x"]``, ``**out``, ``for k in out``, ``"swing" in out``, ``json.dump(out.to_dict())``,
and PyTorch's default collate (which treats a Mapping as a dict). Fields set to None are absent
from the mapping view, which is how optional layers behave today.

    @dataclass(frozen=True, eq=False)
    class TrueState(Bundle):
        x: np.ndarray      # the 2N-1 state per record
        thsl: np.ndarray   # the slack angle reference per record

See docs/DATA_MODELS_PLAN.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Dict, Iterator, Tuple


@dataclass(frozen=True, eq=False)
class Bundle(Mapping):
    """Base of every typed record: a frozen dataclass that is also a read-only Mapping of its
    non-None fields. Subclass with ``@dataclass(frozen=True, eq=False)``."""

    _names_cache: ClassVar[Tuple[str, ...]] = ()

    @classmethod
    def _names(cls) -> Tuple[str, ...]:
        if "_names_cache" not in cls.__dict__:  # computed once per subclass, after the decorator ran
            cls._names_cache = tuple(f.name for f in fields(cls))
        return cls._names_cache

    def _present(self) -> Tuple[str, ...]:
        return tuple(n for n in self._names() if getattr(self, n) is not None)

    def __getitem__(self, key: str) -> Any:
        if key not in self._names() or getattr(self, key) is None:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._present())

    def __len__(self) -> int:
        return len(self._present())

    def to_dict(self) -> Dict[str, Any]:
        """The plain dict this bundle replaces: every non-None field, in field order."""
        return {n: getattr(self, n) for n in self._present()}

    def __repr__(self) -> str:
        parts = []
        for n in self._present():
            v = getattr(self, n)
            parts.append(f"{n}=<{type(v).__name__}{list(v.shape)}>" if hasattr(v, "shape") else f"{n}={v!r}")
        return f"{type(self).__name__}({', '.join(parts)})"
