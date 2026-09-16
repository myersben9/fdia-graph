"""The mathematics of the package as named, pure functions: numpy in, numpy out, no files, no
dataset objects, no torch. Each function carries its equation and a source key from
docs/reference/REFERENCES.md; docs/reference/FORMULAS.md is the catalogue.

Provisional until 1.0: names may still be adjusted once, with a deprecation alias.
"""

from .network import branch_admittances, branch_flows, bus_injections, complex_voltages, series_admittance

__all__ = ["branch_admittances", "branch_flows", "bus_injections", "complex_voltages", "series_admittance"]
