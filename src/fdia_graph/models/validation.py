"""The one place inputs are checked (docs/plans/VALIDATION_PLAN.md).

A model declares what each field may be in the field's annotation, and `Validated` checks every
field when the model is built:

    @dataclass(frozen=True)
    class HuberConfig(Validated):
        c: Annotated[float, Positive()] = 1.5
        reweight: Annotated[Optional[str], OneOf(Reweight)] = None

Every rule in a field's `Annotated` metadata is applied in order (`OneOf` also converts the value to
the choice's canonical string, so "NYISO" is stored as "nyiso"); a None value of an `Optional` field
skips them. Last, the model's `invariants()` (conditions across fields) must hold. A failure raises `ConfigError`, a `ValueError`, with one message shape:
"<Model>.<field> <what the rule says>, got <value>". Functions never check their arguments
themselves; they build the model.

`expect`, `present`, `expect_ndim` and `expect_same_shape` are the same checks for the formulas, which take
bare arrays by design: one line declares a formula's contract and the message is built here.
"""

from __future__ import annotations

import math
import numbers
import typing
from collections.abc import Iterable
from dataclasses import fields
from typing import Any, Optional, TypeVar, Union

import numpy as np

from .choices import Choice

T = TypeVar("T")


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
        """The value, unchanged, or ConfigError naming `where` when the rule does not hold."""
        if not self.holds(value):
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
        rules, optional = _unpack(hints[f.name])
        value = getattr(model, f.name)
        if value is None and optional:
            continue
        for rule in rules:
            value = rule.apply(value, f"{name}.{f.name}")
        object.__setattr__(model, f.name, value)
    for holds, says in model.invariants():
        if not holds:
            raise ConfigError(f"{name}: {says}")


def _unpack(hint: Any) -> tuple[tuple[Rule, ...], bool]:
    """(the field's rules, whether None is allowed) from its annotation."""
    rules: tuple[Rule, ...] = ()
    if typing.get_origin(hint) is typing.Annotated:
        hint, *meta = typing.get_args(hint)
        rules = tuple(m for m in meta if isinstance(m, Rule))
    return rules, typing.get_origin(hint) is Union and type(None) in typing.get_args(hint)


# ---- array contracts of the formulas -----------------------------------------------------------------
def expect(condition: object, says: str) -> None:
    """A stated precondition (tested for truth, as `if` would); `says` is the requirement in words."""
    if not condition:
        raise ConfigError(says)


def present(value: Optional[T], says: str) -> T:
    """`value`, which must not be None; the return type carries that to the type checker."""
    if value is None:
        raise ConfigError(says)
    return value


def expect_ndim(x: Any, ndim: int, what: str) -> np.ndarray:
    """`x` as an array with exactly `ndim` dimensions."""
    a = np.asarray(x)
    expect(a.ndim == ndim, f"{what} must be {ndim}-dimensional, got shape {a.shape}")
    return a


def expect_same_shape(a: Any, b: Any, what: str) -> None:
    expect(np.shape(a) == np.shape(b), f"{what} must have one shape, got {np.shape(a)} and {np.shape(b)}")
