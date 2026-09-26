"""The one place inputs are checked (docs/plans/VALIDATION_PLAN.md).

A model declares what each field may be in the field's annotation, and `Validated` checks every
field when the model is built:

    @dataclass(frozen=True)
    class HuberConfig(Validated):
        c: Annotated[float, Positive()] = 1.5
        reweight: Annotated[Optional[str], OneOf(Reweight)] = None

Every rule in a field's `Annotated` metadata is applied in order. A converting rule also replaces the
value: `OneOf` stores the choice's canonical string ("NYISO" becomes "nyiso"), `AsArray` a numpy array,
`Parses` whatever its parser returns. A None value skips the rules of an `Optional` field and is
refused for any other: the annotation says the field is required. Last, the model's
`invariants()` (conditions across fields) must hold. A failure raises `ConfigError`, a `ValueError`, with one message shape:
"<Model>.<field> <what the rule says>, got <value>". Functions never check their arguments
themselves; they build the model.

Arrays are checked the same way: a formula or a parser builds its input model (`models.inputs`)
from its arguments, so its body only ever sees values the model has accepted.
"""

from __future__ import annotations

import math
import numbers
import typing
from collections.abc import Iterable
from dataclasses import fields
from typing import Any, Union

import numpy as np

from .choices import Choice


class ConfigError(ValueError):
    """An input that its model does not allow. A ValueError, so callers catching that keep working."""


class MissingCapability(ConfigError):
    """A dataset view that lacks what a consumer needs (`dataset.base.DatasetBase.require`)."""


# ---- rules ------------------------------------------------------------------------------------------
class Rule:
    """One condition on a field's value and the phrase that states it."""

    says = ""

    def holds(self, value: Any) -> bool:
        raise NotImplementedError

    def apply(self, value: Any, where: str) -> Any:
        """The value, unchanged, or ConfigError naming `where` when the rule does not hold. A value
        the rule cannot even evaluate (a string where a number belongs) has failed it too, so a
        malformed argument gets the same message as an out-of-range one."""
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

    def apply(self, value: Any, where: str) -> Any:
        try:
            return self.choice(value).value
        except ValueError:
            raise ConfigError(f"{where} {self.says}, got {value!r}") from None


class Positive(Rule):
    says = "must be > 0"

    def holds(self, value: Any) -> bool:
        return value > 0


class Finite(Rule):
    says = "must be finite"

    def holds(self, value: Any) -> bool:
        return math.isfinite(value)


class Integer(Rule):
    says = "must be an integer"

    def holds(self, value: Any) -> bool:
        return isinstance(value, numbers.Integral) and not isinstance(value, bool)


class AtLeast(Rule):
    def __init__(self, low: float) -> None:
        self.low, self.says = low, f"must be >= {low}"

    def holds(self, value: Any) -> bool:
        return value >= self.low


class InRange(Rule):
    """lo < value < hi, each end open or closed: InRange(0, 1) is (0, 1), InRange(0, 1, hi_closed=True) (0, 1]."""

    def __init__(self, lo: float, hi: float, lo_closed: bool = False, hi_closed: bool = False) -> None:
        self.lo, self.hi, self.lo_closed, self.hi_closed = lo, hi, lo_closed, hi_closed
        self.says = f"must be in {'[' if lo_closed else '('}{lo:g}, {hi:g}{']' if hi_closed else ')'}"

    def holds(self, value: Any) -> bool:
        above = value >= self.lo if self.lo_closed else value > self.lo
        below = value <= self.hi if self.hi_closed else value < self.hi
        return above and below


class Required(Rule):
    """None is refused even where the type allows it (a caller's Optional that must be given)."""

    def __init__(self, says: str = "is required") -> None:
        self.says = says

    def holds(self, value: Any) -> bool:
        return value is not None


class AsArray(Rule):
    """Converts the value to a numpy array (of `dtype` when given)."""

    says = "must be array-like"

    def __init__(self, dtype: Any = None) -> None:
        self.dtype = dtype

    def apply(self, value: Any, where: str) -> Any:
        try:
            return np.asarray(value, self.dtype)
        except (TypeError, ValueError):
            raise ConfigError(f"{where} {self.says}, got {value!r}") from None


class Dims(Rule):
    def __init__(self, ndim: int) -> None:
        self.ndim, self.says = ndim, f"must be {ndim}-dimensional"

    def holds(self, value: Any) -> bool:
        return np.ndim(value) == self.ndim


class IntegerDtype(Rule):
    says = "must hold integers"

    def holds(self, value: Any) -> bool:
        return np.issubdtype(np.asarray(value).dtype, np.integer)


class NonEmpty(Rule):
    says = "must not be empty"

    def holds(self, value: Any) -> bool:
        return np.size(value) > 0 if isinstance(value, np.ndarray) else len(value) > 0


class AllFinite(Rule):
    says = "must be finite everywhere"

    def holds(self, value: Any) -> bool:
        return bool(np.isfinite(np.asarray(value, np.float64)).all())


class AllPositive(Rule):
    says = "must be positive everywhere"

    def holds(self, value: Any) -> bool:
        return bool((np.asarray(value, np.float64) > 0).all())


class Parses(Rule):
    """Replaces the value with `parse(value)`; a parser that raises ValueError, TypeError or
    KeyError means the value is not of the form the field takes."""

    def __init__(self, parse: Any, says: str) -> None:
        self.parse, self.says = parse, says

    def apply(self, value: Any, where: str) -> Any:
        try:
            return self.parse(value)
        except (ValueError, TypeError, KeyError):
            raise ConfigError(f"{where} {self.says}, got {value!r}") from None


# ---- the engine -------------------------------------------------------------------------------------
class Validated:
    """Base of every model whose fields are checked on construction (a frozen dataclass)."""

    def __post_init__(self) -> None:
        validate(self)

    def invariants(self) -> Iterable[tuple[bool, str]]:
        """(condition, what it says) pairs across fields; every condition must hold."""
        return ()


def validate(model: Any) -> None:
    """Convert and check every field of `model`, then its invariants."""
    name = type(model).__name__
    hints = typing.get_type_hints(type(model), include_extras=True)
    for f in fields(model):
        where = f"{name}.{f.name}"
        object.__setattr__(model, f.name, _checked(getattr(model, f.name), *_unpack(hints[f.name]), where))
    _check_invariants(model, name)


def _checked(value: Any, rules: tuple[Rule, ...], optional: bool, where: str) -> Any:
    """One field's value after its rules; None only where the annotation allows it."""
    if value is None:
        required = next((r for r in rules if isinstance(r, Required)), None)
        if optional and required is None:
            return None
        raise ConfigError(f"{where} {required.says if required else 'is required'}")
    for rule in rules:
        value = rule.apply(value, where)
    return value


def _check_invariants(model: Any, name: str) -> None:
    """The model's conditions across fields, taken lazily so a later condition never sees input an
    earlier one refused; one that cannot even be evaluated means the input is malformed."""
    try:
        for holds, says in model.invariants():
            if not holds:
                raise ConfigError(f"{name}: {says}")
    except ConfigError:
        raise
    except (TypeError, ValueError, IndexError, OverflowError) as e:
        raise ConfigError(f"{name}: the input is malformed ({e})") from None


def _unpack(hint: Any) -> tuple[tuple[Rule, ...], bool]:
    """(the field's rules, whether None is allowed) from its annotation."""
    rules: tuple[Rule, ...] = ()
    if typing.get_origin(hint) is typing.Annotated:
        hint, *meta = typing.get_args(hint)
        rules = tuple(m for m in meta if isinstance(m, Rule))
    return rules, typing.get_origin(hint) is Union and type(None) in typing.get_args(hint)
