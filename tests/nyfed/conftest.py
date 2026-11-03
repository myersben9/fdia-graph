"""The New York tests need the `[ny]` extra (Pyomo, pandas, pandapower, SciPy, imported when the package's
modules load); without it the folder is left out of collection rather than failing on import. With it,
each test still skips when the NYgrid checkout or the built pool is missing under NYFED_ROOT."""

import importlib.util

if any(importlib.util.find_spec(m) is None for m in ("pyomo", "pandas", "pandapower", "scipy")):
    collect_ignore_glob = ["test_nyfed_*.py"]
