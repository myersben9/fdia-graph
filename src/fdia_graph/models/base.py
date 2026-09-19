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

A bundle can inherit field groups (``models/fields.py``), whose fields all default to None so any
combination composes; it then names the fields it cannot do without in ``_required`` (a None
there raises ``TypeError`` at construction, as a missing argument does) and its dict-key order in
``_order`` (the order the old dict had), since dataclass inheritance fixes the field order.

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
construction. See docs/plans/DATA_MODELS_PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar, TypeVar

_B = TypeVar("_B", bound="Bundle")


@dataclass(frozen=True, eq=False)
class Bundle(dict):
    """Base of every typed record: a frozen dataclass that is also the dict of its non-None
    fields. Subclass with ``@dataclass(frozen=True, eq=False)``."""

    _keys: ClassVar[dict[str, str]] = {}  # field name -> dict key, only where they differ
    _tail: ClassVar[tuple[str, ...]] = ()  # fields the dict view lists last, where the old dict put them last
    _order: ClassVar[tuple[str, ...]] = ()  # the full dict-key order, when it differs from the field order
    _required: ClassVar[tuple[str, ...]] = ()  # fields that may not be None (construction raises TypeError)
    _names_cache: ClassVar[tuple[str, ...]] = ()

    def __new__(cls, *args: Any, **kwargs: Any) -> "Bundle":
        # A bundle built from field groups has a field order set by inheritance, so positional
        # arguments would bind silently to the wrong fields: those bundles are keyword-only.
        if args and cls._order:
            raise TypeError(f"{cls.__name__} is built from field groups; pass its fields by keyword")
        return dict.__new__(cls)

    def __post_init__(self) -> None:
        missing = [n for n in self._required if getattr(self, n) is None]
        if missing:
            raise TypeError(f"{type(self).__name__} is missing required field(s) {missing}")
        dict.__init__(self, {self.key_of(n): getattr(self, n) for n in self._present()})

    @classmethod
    def _names(cls) -> tuple[str, ...]:
        if "_names_cache" not in cls.__dict__:  # computed once per subclass, after the decorator ran
            cls._names_cache = cls._dict_order([f.name for f in fields(cls)])
        return cls._names_cache

    @classmethod
    def _dict_order(cls, names: list[str]) -> tuple[str, ...]:
        """The field names in dict-key order: `_order` when the bundle declares it (field groups fix
        the dataclass order), else the field order with `_tail` moved to the end."""
        if cls._order:
            field_of = {cls.key_of(n): n for n in names}
            ordered = [field_of[k] for k in cls._order]
            return tuple(ordered) + tuple(n for n in names if n not in ordered)
        return tuple(n for n in names if n not in cls._tail) + tuple(n for n in names if n in cls._tail)

    @classmethod
    def key_of(cls, field_name: str) -> str:
        """The dict key of a field (the field name unless ``_keys`` says otherwise)."""
        return cls._keys.get(field_name, field_name)

    def _present(self) -> tuple[str, ...]:
        return tuple(n for n in self._names() if getattr(self, n) is not None)

    def to_dict(self) -> dict[str, Any]:
        """A plain dict copy of the bundle: every non-None field under its dict key."""
        return dict(self)

    def __repr__(self) -> str:
        parts = []
        for n in self._present():
            v = getattr(self, n)
            parts.append(f"{n}=<{type(v).__name__}{list(v.shape)}>" if hasattr(v, "shape") else f"{n}={v!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    @classmethod
    def ordered(cls: type[_B], mapping: dict[str, Any]) -> _B:
        """A bundle whose dict view lists the keys in the mapping's order rather than the field order
        (a caller's requested field order in `to_numpy`). Keys are dict keys; None values are absent."""
        field_of = {cls.key_of(n): n for n in cls._names()}
        obj = cls(**{field_of[k]: v for k, v in mapping.items()})
        dict.clear(obj)  # the base methods, since the overrides below refuse
        dict.update(obj, {k: v for k, v in mapping.items() if v is not None})
        return obj

    # The dict side is a mirror of the frozen fields; every mutating entry point of dict refuses, since
    # dict's C implementation of update/pop/clear does not go through __setitem__/__delitem__.
    def _read_only(self) -> TypeError:
        return TypeError(f"{type(self).__name__} is read-only; build a new one or use to_dict()")

    def __setitem__(self, key: str, value: Any) -> None:
        raise self._read_only()

    def __delitem__(self, key: str) -> None:
        raise self._read_only()

    def __ior__(self, other: Any) -> Bundle:
        raise self._read_only()

    def update(self, *args: Any, **kwargs: Any) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        raise self._read_only()

    def setdefault(self, *args: Any, **kwargs: Any) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        raise self._read_only()

    def pop(self, *args: Any, **kwargs: Any) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        raise self._read_only()

    def popitem(self) -> Any:
        raise self._read_only()

    def clear(self) -> None:
        raise self._read_only()

    def __reduce__(self) -> Any:  # pickle through the fields by name, and keep the dict view's key order
        return (_rebuild, (type(self), {n: getattr(self, n) for n in self._names()}, list(self)))


def _rebuild(cls: type[_B], field_values: dict[str, Any], key_order: list) -> _B:
    """Unpickle: construct by field name (positional order and dict order can differ), then restore
    the dict view's key order."""
    obj = cls(**field_values)
    if list(obj) != key_order:
        items = {k: dict.__getitem__(obj, k) for k in key_order}
        dict.clear(obj)
        dict.update(obj, items)
    return obj
