"""Arguments that take one value from a fixed set.

Each such argument is declared once, as a `Choice` enum beside the code it belongs to (`Units` with
the dataset, `Calibrate` with the estimator), and converted where it enters: ``Units(units)``
returns the member or raises. The allowed values then live in exactly one place, the error message
is built from them, and a caller that already holds a member passes it through unchanged.

A member is its string value (``Units.PU == "pu"``, ``f"{Units.PU}" == "pu"``), so every public
signature keeps accepting plain strings and comparisons against string literals keep working.
"""

from __future__ import annotations

import re
from enum import Enum


class Choice(str, Enum):
    """A string option from a fixed set. Converting an unknown value raises one message that names
    the parameter and lists every allowed value; the parameter name is the class name in snake case
    (``AmDirection`` reads as ``am_direction``)."""

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Choice:
        raise ValueError(f"{cls.param()} must be one of {cls.values()}, got {value!r}")

    @classmethod
    def param(cls) -> str:
        """The argument name this choice validates, from the class name."""
        return re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()

    @classmethod
    def values(cls) -> list[str]:
        """Every allowed value, in declaration order."""
        return [m.value for m in cls]
