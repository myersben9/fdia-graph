"""What a trusted-meter selection does for state estimation and localization, not only for the
residual test: the estimators and localizers of the other two guides scored on the plain timeline
and on secured copies of it (`TrustSelector.secured_copy`, the selected meters reading their benign
value on every frame). Writes runs of `trust.secured_localization` and `trust.secured_estimation`
to the results store (`results/`).

Set FG_SYSTEM (default ieee14) and FG_K (the budget, default 20); the copies and the score caches
are keyed by the secured meters and the source file, so a new selection or budget never reuses a
stale entry. The selections come from
the store's `trust.selection` run when `run_trust.py` has written one (its greedy and DQN orders), else
the greedy selector is fitted here. Needs the [se] and [torch] extras; the copies and the per-arm caches go
under results/cache/. The gated estimators run twice on a secured copy: as they are, and with the
secured meters exempt from the gate (`GatedPrior(secured=...)`), which is the arm that wins.
"""

import hashlib
import json
import os
import time

import numpy as np

import fdia_graph as fg
from fdia_graph.localization import BusCNN, ResidualLocalizer
from fdia_graph.registry import register_local
from fdia_graph.results import Run, Store
from fdia_graph.se import GatedPrior, SubspacePrior
from fdia_graph.trust import TrustedMeters

SYSTEM = os.environ.get("FG_SYSTEM", "ieee14")
K = int(os.environ.get("FG_K", "20"))
HP = {"ieee14": (1.5, 0.20), "ieee118": (2.5, 0.50), "ieee300": (6.0, 0.50)}  # docs/se/run_se.py
c, rank = HP.get(SYSTEM, (1.5, 0.5))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
STORE = Store(os.path.join(HERE, "..", "..", "results"))  # the repository's results store


def selections(train):
    """{arm: (secured meter indices, the fitted selector or None)}: the orders run_trust.py stored,
    else a greedy fit."""
    tm = TrustedMeters(k=K).fit(train)  # its estimator gives every copy the meter layout
    stored = {}
    release = fg.resolve(SYSTEM).release or ""
    same_release = [
        p.run_id for p in STORE.runs() if p.experiment == "trust.selection" and p.data_release == release
    ]
    # one run's orders whole, for this budget and this release's meter layout, never steps from two runs
    run = STORE.newest_run(
        "trust.selection", run_id=same_release, system=SYSTEM, metric="selected_meter", k=K
    )
    for name in ("greedy", "dqn"):
        steps = STORE.query(
            "trust.selection", run_id=run, system=SYSTEM, method=name, metric="selected_meter"
        )
        if steps:
            stored[name] = [int(r.value) for r in sorted(steps, key=lambda r: int(r.tag("step")))][:K]
    return (stored, tm) if stored else ({"greedy": [int(i) for i in tm.select()]}, tm)


def fingerprint(meters, source):
    """What a cached copy or score depends on: the secured meters in order and the source file's
    size and modification time, so a new selection or a regenerated timeline never reuses a stale
    cache entry."""
    st = os.stat(source)
    key = f"{list(map(int, meters))}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def cached(tag, fp, name, fit_score):
    """Run an arm once; later runs with the same fingerprint read its scores from the cache."""
    f = os.path.join(CACHE, f"secured_{SYSTEM}_{tag}_{fp}_{name}.json")
    if os.path.exists(f):
        return json.load(open(f))
    t0 = time.time()
    r = fit_score()
    r = json.loads(json.dumps(r))
    json.dump(r, open(f, "w"))
    print(f"  {tag:7s} {name:24s} {time.time() - t0:6.0f}s", flush=True)
    return r


def localization(train, test):
    def resid():
        m = ResidualLocalizer().fit(train)
        return m.score(test)

    def cnn():
        m = BusCNN(features="full14+jac").fit(train)
        return m.score(test)

    return {"residual": resid, "cnn+jac": cnn}


def estimation(train, test, secured):
    kw = dict(rank_frac=rank, reweight="huber", c=c)

    def run(est):
        est.fit(train)
        return est.score(test)

    arms = {
        "prior+huber": lambda: run(SubspacePrior(**kw)),
        "prior+huber+cnn": lambda: run(GatedPrior(gate=BusCNN().fit(train), **kw)),
        "prior+huber+residual": lambda: run(GatedPrior(gate=ResidualLocalizer().fit(train), **kw)),
        "prior+huber+oracle": lambda: run(GatedPrior(gate="oracle", **kw)),
    }
    if len(secured):
        arms["prior+huber+cnn+trust"] = lambda: run(
            GatedPrior(gate=BusCNN().fit(train), secured=secured, **kw)
        )
        arms["prior+huber+residual+trust"] = lambda: run(
            GatedPrior(gate=ResidualLocalizer().fit(train), secured=secured, **kw)
        )
        arms["prior+huber+oracle+trust"] = lambda: run(GatedPrior(gate="oracle", secured=secured, **kw))
    return arms


plain_train, plain_test = fg.load(SYSTEM, split="train"), fg.load(SYSTEM, split="test")
source = fg.load(SYSTEM, order="time")
orders, tm = selections(plain_train)
arms = {"plain": (SYSTEM, [], fingerprint([], source.path))}
for sel, meters in orders.items():
    fp = fingerprint(meters, source.path)
    name = f"{SYSTEM}_{sel}{K}_{fp}"
    path = os.path.join(CACHE, f"timeline_{name}.h5")
    if not os.path.exists(path):
        tm.order = list(meters)  # the stored selection, through the same estimator layout
        tm.secured_copy(source, path, name=name)
    else:
        register_local(name, path)
    arms[sel] = (name, list(meters), fp)

report = {"k": K, "orders": {k: v for k, v in orders.items()}}
for tag, (name, meters, fp) in arms.items():
    train, test = fg.load(name, split="train"), fg.load(name, split="test")
    report[tag] = {"localization": {}, "estimation": {}, "fingerprint": fp}
    for arm, fn in localization(train, test).items():
        report[tag]["localization"][arm] = cached(tag, fp, "loc_" + arm, fn)
    for arm, fn in estimation(train, test, np.asarray(meters, int)).items():
        report[tag]["estimation"][arm] = cached(tag, fp, "se_" + arm, fn)

release = fg.resolve(SYSTEM).release or ""
for part, experiment in (
    ("localization", "trust.secured_localization"),
    ("estimation", "trust.secured_estimation"),
):
    with Run(
        experiment,
        system=SYSTEM,
        settings={"k": K, "orders": report["orders"]},
        data_release=release,
        seed=0,
        store=STORE,
    ) as out:
        for tag in arms:
            out.add_tree(report[tag][part], levels=("method", "family"), secured=tag, k=K)

fams = ["Aq", "At", "Al", "Am", "Ad", "As", "Ar"]
print(f"\n{SYSTEM}, {K} secured meters; node-F1 (detection rate) per family, then macro-F1 / benign FA")
for arm in ("residual", "cnn+jac"):
    print(f"\nlocalization: {arm}")
    for tag in arms:
        r = report[tag]["localization"][arm]
        cells = " ".join(f"{f} {r[f]['node_f1']:.2f} ({r[f]['detection_rate']:.2f})" for f in fams if f in r)
        print(f"  {tag:7s} {cells}  all {r['all']['macro_f1']:.3f} / {r['all']['macro_fr']:.4f}")
print("\nestimation: geometric-mean angle MAE (deg), then per family")
for arm in report["plain"]["estimation"] | {a: None for t in arms for a in report[t]["estimation"]}:
    print(f"\n{arm}")
    for tag in arms:
        r = report[tag]["estimation"].get(arm)
        if r:
            per = " ".join(f"{f} {r[f]['angle_mae_deg']:.3f}" for f in fams if f in r)
            print(f"  {tag:7s} geo {r['geo']['angle_mae_deg']:.4f}   {per}")
print(f"[ok] wrote trust.secured_localization and trust.secured_estimation for {SYSTEM}")
