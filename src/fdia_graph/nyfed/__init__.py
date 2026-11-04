"""The New York grid: stealthy attacks on NYISO interfaces and the operator that answers them.

`fdia_graph.nyfed` builds a federated FDIA dataset and studies attacks through the New York operator's own rules.
The work runs in stages, one subpackage each, every stage reading what the one before it wrote:

1. `grid`: the network. NYgrid's modified NPCC-140 case as a `Case` record, its admittances (`Network`), the
   NYISO interfaces with their real 2019 limits and flows, and one client per utility territory (`Ownership`).
2. `scenario`: the operating points. The 2019 loads by bus, an hourly dispatch, and a ramp-limited AC dispatch every
   five minutes, solved into the `Pool` of true states (`PoolBuilder`).
3. `sensing`: the meters. A SCADA and PMU `MeterPlan` owned by the clients, the measurement function h(x)
   (`Measurement`) and each meter's error (`NoiseModel`).
4. `attacks`: the attacker. A local false state re-solved over a subnetwork (`Attacker`), and the attack families
   that design an `Episode` for a window of true states (`AttackFamily`).
5. `timeline`: the dataset. A year of noisy scans with episodes placed at random and labelled, written to
   `timeline.h5` (`TimelineBuilder`, built for the release by `production_builder`), read back as clients by
   `Dataset`.
6. `operator`: the consequence. The operator's five-minute dispatch, regulation and alarms on what it reads, run
   against the true grid (`respond`, `summarize`).

Entry points: `production_builder` builds the release timeline, `Dataset` reads one, `operator_day` and `respond`
run the operator on an attacked day. Every public name below loads on first use. When a stage fails it raises a
named error from `fdia_graph.errors` or records why it produced less: see docs/nyfed/TROUBLESHOOTING.md, and
docs/nyfed/ARCHITECTURE.md for how the modules fit together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fdia_graph.nyfed._lazy import attach

if TYPE_CHECKING:  # the names below, with their real signatures for type checkers
    from .build import fetch_inputs, production_builder
    from .config import Settings, rng
    from .dataset import ClientView, Dataset
    from .io import FAMILIES, SPLITS, TimelineWriter, open_timeline, read_pool, write_pool
    from .operator.response import operator_day, respond, summarize
    from .solvers import ipopt

# each public name and the module that defines it, imported on first use
_PUBLIC = {
    "production_builder": ".build",
    "fetch_inputs": ".build",
    "Settings": ".config",
    "rng": ".config",
    "Dataset": ".dataset",
    "ClientView": ".dataset",
    "read_pool": ".io",
    "write_pool": ".io",
    "open_timeline": ".io",
    "TimelineWriter": ".io",
    "SPLITS": ".io",
    "FAMILIES": ".io",
    "operator_day": ".operator.response",
    "respond": ".operator.response",
    "summarize": ".operator.response",
    "ipopt": ".solvers",
}

__all__ = [
    "ClientView",
    "Dataset",
    "FAMILIES",
    "SPLITS",
    "Settings",
    "TimelineWriter",
    "fetch_inputs",
    "ipopt",
    "open_timeline",
    "operator_day",
    "production_builder",
    "read_pool",
    "respond",
    "rng",
    "summarize",
    "write_pool",
]
__getattr__, __dir__ = attach(__name__, _PUBLIC)
