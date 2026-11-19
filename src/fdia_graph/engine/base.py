"""A part of the generator: one job (emitting scans, choosing an area, solving a false state, ...)
as one class that holds the generator it belongs to.

`FdiaGenerator` (core.py) holds the grid, the meters and the random stream, and one object of each
part, named after its class (`g.minimizer` is the `Minimizer`). A part reads that state as
`self.g.<name>` and calls another part as `self.g.<part>.<method>`, so a call names the class its
method is in."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import FdiaGenerator


class Part:
    """One part of the generator `g`."""

    def __init__(self, g: FdiaGenerator) -> None:
        self.g = g
