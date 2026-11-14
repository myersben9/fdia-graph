"""The optimization solver every nygrid OPF runs on, found and checked in one place.

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
    path = os.environ.get("PATH", "")
    on_path = {os.path.normcase(os.path.normpath(p)) for p in path.split(os.pathsep) if p}
    if (
        os.path.isdir(lib) and os.path.normcase(os.path.normpath(lib)) not in on_path
    ):  # whole entries, not text
        os.environ["PATH"] = lib + os.pathsep + path
    exe = os.path.join(lib, "ipopt.exe")
    solver = pe.SolverFactory("ipopt", executable=exe) if os.path.exists(exe) else pe.SolverFactory("ipopt")
    runs = solver.available(exception_flag=False)
    if runs:  # found is not enough: a missing DLL only shows when the program is started
        try:
            version = solver.version()
        except Exception:  # Pyomo raises RuntimeError or ApplicationError here, by version
            version = None
        # Pyomo returns (0, 0, 0, 0) when the program printed no version, as one that cannot start does
        runs = version is not None and any(version)
    if not runs:
        raise SolverUnavailable(
            "Ipopt is not found or does not start. Install it into the environment "
            "(conda install -c conda-forge ipopt) and run with that environment active, so its Library/bin "
            "folder is on PATH (a missing DLL shows as exit code 0xC0000135)."
        )
    solver.options.update(options)
    return solver
