"""Bundle: a typed record that still IS the dict it used to be.

The package's public functions have always returned dicts of arrays (records, batches, streams,
score tables). A Bundle is a frozen dataclass whose fields are those keys, so a reader sees them
in one place and pyright checks them, and it is also a real ``dict`` subclass, so every existing
use keeps working unchanged: ``out["node_x"]``, ``**out``, ``for k in out``, ``"swing" in out``,
``json.dump(out)``, ``isinstance(out, dict)``, pickling, and PyTorch's default collate. Fields set
to None are absent from the dict view, which is how optional layers behave today.

    @dataclass(frozen=True, eq=False)
    class TrueState(Bundle):
        x: np.ndarray      # the 2N-1 state per record
        thsl: np.ndarray   # the slack angle reference per record

A field whose dict key is not a valid attribute name (``"global"``) is declared with a trailing
underscore and mapped through ``_keys``:

    @dataclass(frozen=True, eq=False)
    class JacobianOutputs(Bundle):
        _keys = {"global_": "global"}
        bus: np.ndarray
        global_: np.ndarray

A required field the old dict listed last (a score table's ``geo`` row) is named in ``_tail`` so
the dict view keeps the old key order; dataclasses need required fields before optional ones.

Treat a bundle as read-only: the dataclass side is frozen, and the dict side mirrors it at
construction. See docs/DATA_MODELS_PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar, Dict, Tuple


@dataclass(frozen=True, eq=False)
class Bundle(dict):
    """Base of every typed record: a frozen dataclass that is also the dict of its non-None
    fields. Subclass with ``@dataclass(frozen=True, eq=False)``."""

    _keys: ClassVar[Dict[str, str]] = {}  # field name -> dict key, only where they differ
    _tail: ClassVar[Tuple[str, ...]] = ()  # fields the dict view lists last, where the old dict put them last
    _names_cache: ClassVar[Tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        dict.__init__(self, {self.key_of(n): getattr(self, n) for n in self._present()})

    @classmethod
    def _names(cls) -> Tuple[str, ...]:
        if "_names_cache" not in cls.__dict__:  # computed once per subclass, after the decorator ran
            names = [f.name for f in fields(cls)]
            cls._names_cache = tuple(n for n in names if n not in cls._tail) + tuple(
                n for n in names if n in cls._tail
            )
        return cls._names_cache

    @classmethod
    def key_of(cls, field_name: str) -> str:
        """The dict key of a field (the field name unless ``_keys`` says otherwise)."""
        return cls._keys.get(field_name, field_name)

    def _present(self) -> Tuple[str, ...]:
        return tuple(n for n in self._names() if getattr(self, n) is not None)

    def to_dict(self) -> Dict[str, Any]:
        """A plain dict copy of the bundle: every non-None field under its dict key."""
        return dict(self)

    def __repr__(self) -> str:
        parts = []
        for n in self._present():
            v = getattr(self, n)
            parts.append(f"{n}=<{type(v).__name__}{list(v.shape)}>" if hasattr(v, "shape") else f"{n}={v!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    # The dict side is a mirror of the frozen fields; refuse the mutations that would desynchronize it.
    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError(f"{type(self).__name__} is read-only; build a new one or use to_dict()")

    def __delitem__(self, key: str) -> None:
        raise TypeError(f"{type(self).__name__} is read-only; build a new one or use to_dict()")

    def __reduce__(self) -> Any:  # pickle through the fields, not the dict storage
        return (type(self), tuple(getattr(self, n) for n in self._names()))
