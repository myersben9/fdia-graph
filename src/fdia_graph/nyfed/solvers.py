"""The optimization solver every nyfed OPF runs on, found and checked in one place.

Ipopt comes from conda-forge into the environment's `Library/bin` folder on Windows, and it needs the DLLs in that
folder on PATH to start. When it cannot start, Pyomo reports a failed solve rather than a missing program, so a
caller that reads a failed solve as "infeasible" would quietly build nothing. `ipopt` therefore puts the folder on
PATH, checks that the solver runs, and raises `SolverUnavailable` with the fix when it does not.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from fdia_graph.errors import SolverUnavailable

__all__ = ["ipopt"]


def ipopt(**options: Any) -> Any:
    """A Pyomo `SolverFactory` for Ipopt with `options` set, checked to run.

    Raises:
        SolverUnavailable: Pyomo is not installed, or Ipopt is not found or does not start.
    """
    try:
        import pyomo.environ as pe
    except ImportError as err:  # the [ny] extra brings Pyomo
        raise SolverUnavailable("Pyomo is not installed: pip install 'fdia-graph[ny]'") from err
    lib = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(lib) and lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    exe = os.path.join(lib, "ipopt.exe")
    solver = pe.SolverFactory("ipopt", executable=exe) if os.path.exists(exe) else pe.SolverFactory("ipopt")
    runs = solver.available(exception_flag=False)
    if runs:  # found is not enough: a missing DLL only shows when the program is started
        try:
            runs = solver.version() is not None
        except Exception:  # Pyomo raises RuntimeError or ApplicationError here, by version
            runs = False
    if not runs:
        raise SolverUnavailable(
            "Ipopt is not found or does not start. Install it into the environment "
            "(conda install -c conda-forge ipopt) and run with that environment active, so its Library/bin "
            "folder is on PATH (a missing DLL shows as exit code 0xC0000135)."
        )
    solver.options.update(options)
    return solver
