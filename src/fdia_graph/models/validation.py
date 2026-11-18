"""The one place inputs are checked (CONTRIBUTING.md, "an input is checked in one place").

A model declares its rules in each field's annotation (`c: Annotated[float, Positive()] = 1.5`), and
`Validated` applies them in order when the model is built. A converting rule replaces the value:
`OneOf` stores the choice's canonical string, `AsArray` a numpy array, `Parses` what its parser
returns. None skips the rules of an `Optional` field and is refused for any other. Last, the model's
`invariants()` (conditions across fields) must hold. A failure raises `ConfigError` (a `ValueError`):
"<Model>.<field> <what the rule says>, got <value>". Functions never check their arguments; they
build the model (`models.inputs` for arrays).
"""

from __future__ import annotations

import math
import numbers
import typing
from collections.abc import Callable, Iterable, Sized
from dataclasses import Field, fields
from typing import ClassVar, Protocol, SupportsFloat, Union, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..errors import ConfigError
from .choices import Choice


# ---- rules ------------------------------------------------------------------------------------------
@runtime_checkable
class Ordered(Protocol):
    """A value a bound compares with: anything with the rich comparisons (a number, a numpy scalar). A value
    whose comparison raises (a string against a number) fails the rule as `Rule.apply` says."""

    def __lt__(self, other: float, /) -> object: ...
    def __le__(self, other: float, /) -> object: ...
    def __gt__(self, other: float, /) -> object: ...
    def __ge__(self, other: float, /) -> object: ...


class Rule:
    """One condition on a field's value and the phrase that states it."""

    says = ""

    def holds(self, value: object) -> bool:
        raise NotImplementedError

    def apply(self, value: object, where: str) -> object:
        """The value, unchanged, or ConfigError naming `where` when the rule does not hold. A value
        the rule cannot evaluate (a string where a number belongs) fails it with the same message."""
        try:
            holds = bool(self.holds(value))
        except (TypeError, ValueError, OverflowError):  # 10**1000 overflows a float conversion
            holds = False
        if not holds:
            raise ConfigError(f"{where} {self.says}, got {value!r}")
        return value


class OneOf(Rule):
    """One value of a `Choice`; the value is stored as the choice's canonical string."""

    def __init__(self, choice: type[Choice]) -> None:
        self.choice, self.says = choice, f"must be one of {choice.values()}"

    def apply(self, value: object, where: str) -> str:
        try:
            return self.choice(value).value
        except ValueError:
            raise ConfigError(f"{where} {self.says}, got {value!r}") from None


class Positive(Rule):
    says = "must be > 0"

    def holds(self, value: object) -> bool:
        return isinstance(value, Ordered) and bool(value > 0)


class Finite(Rule):
    says = "must be finite"

    def holds(self, value: object) -> bool:
        return isinstance(value, SupportsFloat) and math.isfinite(value)


class Boolean(Rule):
    """True or False itself, so a string such as "False" (truthy) is refused rather than read as on."""

    says = "must be True or False"

    def holds(self, value: object) -> bool:
        return isinstance(value, (bool, np.bool_))


class Integer(Rule):
    says = "must be an integer"

    def holds(self, value: object) -> bool:
        return isinstance(value, numbers.Integral) and not isinstance(value, bool)


class AtLeast(Rule):
    def __init__(self, low: float) -> None:
        self.low, self.says = low, f"must be >= {low}"

    def holds(self, value: object) -> bool:
        return isinstance(value, Ordered) and bool(value >= self.low)


class InRange(Rule):
    """lo < value < hi, each end open or closed: InRange(0, 1) is (0, 1), InRange(0, 1, hi_closed=True) (0, 1]."""

    def __init__(self, lo: float, hi: float, lo_closed: bool = False, hi_closed: bool = False) -> None:
        self.lo, self.hi, self.lo_closed, self.hi_closed = lo, hi, lo_closed, hi_closed
        self.says = f"must be in {'[' if lo_closed else '('}{lo:g}, {hi:g}{']' if hi_closed else ')'}"

    def holds(self, value: object) -> bool:
        if not isinstance(value, Ordered):
            return False
        above = value >= self.lo if self.lo_closed else value > self.lo
        below = value <= self.hi if self.hi_closed else value < self.hi
        return bool(above) and bool(below)


class Required(Rule):
    """None is refused even where the type allows it (a caller's Optional that must be given)."""

    def __init__(self, says: str = "is required") -> None:
        self.says = says

    def holds(self, value: object) -> bool:
        return value is not None


class AsArray(Rule):
    """Converts the value to a numpy array (of `dtype` when given)."""

    says = "must be array-like"

    def __init__(self, dtype: npt.DTypeLike = None) -> None:
        self.dtype = dtype

    def apply(self, value: object, where: str) -> np.ndarray:
        try:
            return np.asarray(value, self.dtype)
        except (TypeError, ValueError, OverflowError):
            raise ConfigError(f"{where} {self.says}, got {value!r}") from None


class AsTuple(Rule):
    """Converts a list, tuple or other iterable to a tuple; a string, a mapping or a non-iterable is
    not a sequence of items and is refused."""

    says = "must be a sequence"

    def apply(self, value: object, where: str) -> tuple[object, ...]:
        if isinstance(value, (str, bytes, dict)) or not isinstance(value, Iterable):
            raise ConfigError(f"{where} {self.says}, got {value!r}")
        try:
            return tuple(value)
        except TypeError:
            raise ConfigError(f"{where} {self.says}, got {value!r}") from None


class Dims(Rule):
    def __init__(self, ndim: int) -> None:
        self.ndim, self.says = ndim, f"must be {ndim}-dimensional"

    def holds(self, value: object) -> bool:
        return np.asarray(value).ndim == self.ndim


class NonEmpty(Rule):
    says = "must not be empty"

    def holds(self, value: object) -> bool:
        if isinstance(value, np.ndarray):
            return value.size > 0
        return isinstance(value, Sized) and len(value) > 0


class Parses(Rule):
    """Replaces the value with `parse(value)`; a parser that raises ValueError, TypeError or
    KeyError means the value is not of the form the field takes."""

    def __init__(self, parse: Callable[..., object], says: str) -> None:
        self.parse, self.says = parse, says

    def apply(self, value: object, where: str) -> object:
        try:
            return self.parse(value)
        except (ValueError, TypeError, KeyError):
            raise ConfigError(f"{where} {self.says}, got {value!r}") from None


class WhenSet(Rule):
    """For a setting added after files exist: a record leaves the field out when its value is
    `legacy`, what those files meant without the field, so their record and hash stand. `legacy` is
    frozen: it must never change, whatever the field's default becomes."""

    def __init__(self, legacy: object) -> None:
        self.legacy = legacy

    def apply(self, value: object, where: str) -> object:
        return value


# ---- the engine -------------------------------------------------------------------------------------
class Validated:
    """Base of every model whose fields are checked on construction (a frozen dataclass)."""

    # what a failed invariant raises: ConfigError for an input, or a named data condition
    # (`fdia_graph.errors`) for a model that states what the data lacks
    error: ClassVar[type[ValueError]] = ConfigError
    # every model is a dataclass (the subclass is decorated); declared so `fields(model)` type-checks
    __dataclass_fields__: ClassVar[dict[str, Field[object]]]

    def __post_init__(self) -> None:
        validate(self)

    def invariants(self) -> Iterable[tuple[bool, str]]:
        """(condition, what it says) pairs across fields; every condition must hold."""
        return ()


def validate(model: Validated) -> None:
    """Convert and check every field of `model`, then its invariants."""
    name = type(model).__name__
    hints = typing.get_type_hints(type(model), include_extras=True)
    for f in fields(model):
        where = f"{name}.{f.name}"
        object.__setattr__(model, f.name, _checked(getattr(model, f.name), *_unpack(hints[f.name]), where))
    _check_invariants(model, name)


def _checked(value: object, rules: tuple[Rule, ...], optional: bool, where: str) -> object:
    """One field's value after its rules; None only where the annotation allows it."""
    if value is None:
        required = next((r for r in rules if isinstance(r, Required)), None)
        if optional and required is None:
            return None
        raise ConfigError(f"{where} {required.says if required else 'is required'}")
    for rule in rules:
        value = rule.apply(value, where)
    return value


def _check_invariants(model: Validated, name: str) -> None:
    """The model's conditions across fields, taken lazily so a later condition never sees input an
    earlier one refused; one that cannot be evaluated means the input is malformed. A named data
    condition keeps its message: it says what the data lacks, not which model."""
    failed = None
    try:
        for holds, says in model.invariants():
            if not holds:
                failed = says
                break
    except ConfigError:
        raise
    except (TypeError, ValueError, IndexError, KeyError, AttributeError, OverflowError) as e:
        raise ConfigError(f"{name}: the input is malformed ({e})") from None
    if failed is not None:
        raise ConfigError(f"{name}: {failed}") if model.error is ConfigError else model.error(failed)


def _unpack(hint: object) -> tuple[tuple[Rule, ...], bool]:
    """(the field's rules, whether None is allowed) from its annotation."""
    rules: tuple[Rule, ...] = ()
    if typing.get_origin(hint) is typing.Annotated:
        hint, *meta = typing.get_args(hint)
        rules = tuple(m for m in meta if isinstance(m, Rule))
    return rules, typing.get_origin(hint) is Union and type(None) in typing.get_args(hint)
