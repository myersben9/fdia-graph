"""Public names that load on first use, for the nyfed packages.

Each package lists its public names with the module that defines them, so `from fdia_graph.operator import
respond` works while `import fdia_graph.operator` imports nothing heavy. That keeps the dataset readers free of
pandapower and Pyomo, which only the pool and the operator need. The SDK's top-level package does the same.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping


def attach(package: str, names: Mapping[str, str]) -> tuple[Callable[[str], object], Callable[[], list[str]]]:
    """`__getattr__` and `__dir__` for `package`, whose public `names` map each name to the module (relative to the
    package, ".case") that defines it. A name is imported when first asked for, then kept on the package."""

    def __getattr__(name: str) -> object:
        if name not in names:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(names[name], package), name)
        setattr(sys.modules[package], name, value)  # later lookups skip this function
        return value

    def __dir__() -> list[str]:
        return sorted(set(vars(sys.modules[package])) | set(names))

    return __getattr__, __dir__
