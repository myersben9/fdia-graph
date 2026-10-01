"""Select trusted meters on one system and write a run of `trust.selection` to the results store.

Set FG_SYSTEM (default ieee14), FG_K (the budget, default 20) and FG_EPISODES (DQN training
episodes, default 200; 0 skips the DQN). Needs the [se] extra, and [torch] for the DQN.
"""

import os
import time

import fdia_graph as fg
from fdia_graph.results import Run, Store
from fdia_graph.trust import TrustedMeters, TrustedMetersDQN

SYSTEM = os.environ.get("FG_SYSTEM", "ieee14")
K = int(os.environ.get("FG_K", "20"))
EPISODES = int(os.environ.get("FG_EPISODES", "200"))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
STORE = Store(os.path.join(HERE, "..", "..", "results"))  # the repository's results store

train = fg.load(SYSTEM, split="train")
test = fg.load(SYSTEM, split="test", order="time")
report = {}
arms = {"greedy": TrustedMeters(k=K)}
if EPISODES > 0:
    arms["dqn"] = TrustedMetersDQN(
        k=K, episodes=EPISODES, seed=0
    )  # the published tables were made with seed 0
for name, tm in arms.items():
    t0 = time.time()
    tm.fit(train)
    fit_s = time.time() - t0
    rep = tm.score(test)
    report[name] = {**rep.to_dict(), "fit_seconds": round(fit_s, 1), "m": tm.m}
    print(
        f"  {name:6s} fit {fit_s:6.0f}s  secured {len(tm.order):3d}  cost {tm.attack_cost([]):.0f} -> "
        f"{tm.attack_cost():.0f}  "
        + "  ".join(
            f"{f} {rep.detected_before[f]:.2f}->{rep.detected_after[f]:.2f}" for f in rep.detected_before
        )
    )
release = fg.resolve(SYSTEM).release or ""
with Run("trust.selection", system=SYSTEM, settings={"k": K, "episodes": EPISODES}, data_release=release, seed=0, store=STORE) as out:
    for sel, v in report.items():
        for i, meter in enumerate(v["order"]):
            out.add("selected_meter", meter, method=sel, step=i)
        for i, cost in enumerate(v["cost"]):
            out.add("attack_cost", cost, method=sel, step=i)
        for which in ("detected_before", "detected_after"):
            for fam, x in v[which].items():
                out.add(which, x, method=sel, family=fam)
        out.add("false_alarm_rate", v["false_alarm"], method=sel, family="benign")
        out.add("fit_seconds", v["fit_seconds"], method=sel)
        out.add("meters", v["m"], method=sel)
print(f"[ok] wrote trust.selection for {SYSTEM}")
