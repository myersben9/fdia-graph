"""A certificate for the interchange-consistent attacker (N68). For each onset and interface, the largest
push of the reported interface flow ANY attacker can make (any support, any area) in the DC model, with
every bus outside New York held at its true injection and New York's net injection unchanged (so the net
interchange stays true), each attackable New York load moved by at most tau of its load and each unit by
at most `band` of its upper limit, zero-injection buses held. Stealth, AC, reactive and step bounds only
shrink it, so a bound below the hour's real headroom proves no consistent attack exists. The same LP with
the external buses free shows how much the false interchange buys.

    python tools/nyfed/push_bound.py OUT.csv out/damage_study/damage.json [more onset lists ...]

Run from the folder whose `data/` holds the inputs; the onsets are read from damage_study.py's json files.
"""

import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from fdia_graph.nyfed.build import production_builder
from fdia_graph.nyfed.grid.interfaces import hourly_interfaces
from fdia_graph.nyfed.operator.sced import shift_factors

NAMES = ["SPR/DUN-SOUTH", "UPNY CONED", "CENTRAL EAST - VC", "MOSES SOUTH"]
TAUS, BANDS = (0.05, 0.1, 0.2, 0.3), (0.03, 0.05, 0.1)
RISE = 16

t0 = time.perf_counter()
b = production_builder("out/pool_2019_v3.h5", ("Am",), workers=1, survey=0)
att, case, pool = b.att, b.att.net.case, b.pool
ifs = hourly_interfaces(case, pool.time, "data/NYgrid/Data", NAMES)
ptdf = shift_factors(case)
sf = np.array([it.signs @ ptdf[it.branches] for it in ifs])  # [I, N]
ny = att.owner >= 0
pmax_bus = np.zeros(len(case.bus_id))
np.add.at(pmax_bus, case.unit_bus, np.where(np.isfinite(case.pmax), case.pmax, 0.0))
print(f"built [{time.perf_counter() - t0:.0f}s]", flush=True)

onsets = []
for f in sys.argv[2:]:
    onsets += json.load(open(f))["onsets"]
onsets = sorted(set(int(o) for o in onsets))
rows = []
for o in onsets:
    r = o + RISE - 1  # the last rise scan, where the goal reaches the limit
    vm, va = pool.vm[r], pool.va[r]
    if not np.isfinite(vm).all():
        continue
    load = att.bus_load(vm, va)
    free = np.zeros(len(case.bus_id), bool)
    free[att.free_injection_buses(vm, va, np.arange(len(case.bus_id)))] = True
    for i, it in enumerate(ifs):
        H = float(it.limit[r] - it.flow[r])
        for tau in TAUS:
            for band in BANDS:
                cap = np.where(att.gen_bus, band * pmax_bus, tau * np.clip(load, 0, None))
                cap = np.where(free & ny, cap, 0.0)
                cap[att.slack] = (
                    band * pmax_bus[att.slack]
                )  # the reference bus units, metered, within the band
                for consistent in (True, False):
                    lo, hi = -cap.copy(), cap.copy()
                    if not consistent:  # a bus outside New York absorbs whatever the attack needs
                        lo[~ny], hi[~ny] = -np.inf, np.inf
                    a_eq = (ny if consistent else np.ones_like(ny)).astype(float)[None, :]
                    res = linprog(-sf[i], A_eq=a_eq, b_eq=[0.0], bounds=list(zip(lo, hi)), method="highs")
                    push = -res.fun if res.status == 0 else np.nan
                    if res.status == 3:
                        push = np.inf
                    rows.append(
                        {
                            "onset": o,
                            "time": str(pool.time.astype("datetime64[s]")[o]),
                            "interface": it.name,
                            "headroom_mw": H,
                            "tau": tau,
                            "band": band,
                            "consistent": consistent,
                            "push_mw": push,
                        }
                    )
    print(o, f"[{time.perf_counter() - t0:.0f}s]", flush=True)
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
