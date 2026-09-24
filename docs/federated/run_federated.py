"""Run the federated localization papers' protocol on one system and write results/fed_<system>.json.

The protocol behind the federated papers' headline table, on the v0.8.1 timelines (pinned):

- zero-shot split: train and val hold benign + Aq + Ad only, test adds As and Ar;
- FedBusCNN and FedBusMLP (fdia_graph.federated), FedAvg with uniform weights, 60 rounds of
  3 local epochs, AdamW lr 5e-4 and weight decay 0.01, batch 256, gradient clip 1.0, the
  final-round weights;
- K = 1, 2, 3 clients from the spectral partition (random_state 42), local power balance;
- features "full14+jac": the papers' 14-dim per-bus vector plus the 8-channel Jacobian block,
  computed once centrally (the one feature a client does not build from its own meters);
- the papers' validation-best global threshold, summed over per-client confusion counts;
- seeds 123, 124, 125.

Scores are the papers' per-bus macro F1, DR and FR over the attackable buses (`score_perbus`),
with FR over every test record (the paper's Table IV) and over benign records only, overall and
per family (that family plus benign). Set FG_SYSTEM (default ieee14). Each run is saved to
results/runs/ as soon as it finishes and skipped on a re-run; delete a file to recompute it.
Tables and figures come from make_report.py. Needs the [federated] and [se] extras; uses the GPU
when visible. About ten minutes a run on one GPU for every system here.
"""

import glob
import json
import os
import time

import numpy as np

import fdia_graph as fg
from fdia_graph.federated import FedBusCNN, FedBusMLP

SYSTEM = os.environ.get("FG_SYSTEM", "ieee14")
RELEASE = "v0.8.1"  # pinned: the committed runs are this release's, whatever FDIA_GRAPH_RELEASE says
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
RUNS = os.path.join(OUT, "runs")
os.makedirs(RUNS, exist_ok=True)
MODELS = {"cnn": FedBusCNN, "mlp": FedBusMLP}
CLIENTS = (1, 2, 3)
SEEDS = (123, 124, 125)
FAMILIES = ("all", "Aq", "Ad", "As", "Ar")


def block(b):
    """The macro scores of one PerBusMetrics block."""
    return {k: float(getattr(b, k)) for k in ("macro_f1", "macro_dr", "macro_fr", "macro_auprc")}


_splits = {}


def splits():
    """The zero-shot train / val / test splits, loaded on the first run that needs fitting."""
    if not _splits:
        zs = dict(families=[0, 1, 2], release=RELEASE)
        _splits["train"] = fg.load(SYSTEM, split="train", **zs)
        _splits["val"] = fg.load(SYSTEM, split="val", **zs)
        _splits["test"] = fg.load(SYSTEM, split="test", families=[0, 1, 2, 3, 4], release=RELEASE)
        print(f"{SYSTEM}: " + "  ".join(f"{k} {len(v)}" for k, v in _splits.items()))
    return _splits["train"], _splits["val"], _splits["test"]


def run_one(name, K, seed):
    """Fit one federated model and score it, or read the saved run."""
    path = os.path.join(RUNS, f"{SYSTEM}_{name}_K{K}_s{seed}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    train, val, test = splits()
    t0 = time.time()
    m = MODELS[name](K=K, seed=seed, features="full14+jac").fit(train, val=val)
    res = {"tau": float(m.tau), "fit_s": time.time() - t0}
    for fr in ("all", "benign"):
        ps = m.score_perbus(test, buses="attackable", fr_over=fr)
        res[fr] = {f: block(getattr(ps, f)) for f in FAMILIES if getattr(ps, f) is not None}
    with open(path + ".tmp", "w") as fh:  # write then rename, so an interrupted run leaves no file to skip
        json.dump(res, fh, indent=1)
    os.replace(path + ".tmp", path)
    a = res["all"]["all"]
    print(
        f"  {name} K{K} s{seed}  tau {m.tau:.2f}  F1 {a['macro_f1']:.4f}  FR {a['macro_fr']:.5f}  {res['fit_s']:.0f}s"
    )
    return res


def aggregate(runs):
    """Mean and standard deviation over seeds of every score."""
    out = {}
    for fr in ("all", "benign"):
        out[fr] = {}
        for f in runs[0][fr]:
            out[fr][f] = {}
            for k in runs[0][fr][f]:
                v = np.array([r[fr][f][k] for r in runs])
                out[fr][f][k] = {"mean": float(v.mean()), "std": float(v.std())}
    out["tau"] = [r["tau"] for r in runs]
    return out


res = {"system": SYSTEM}
for name in MODELS:
    for K in CLIENTS:
        res[f"{name}_K{K}"] = aggregate([run_one(name, K, s) for s in SEEDS])
with open(os.path.join(OUT, f"fed_{SYSTEM}.json"), "w") as fh:
    json.dump(res, fh, indent=1)
print(f"[ok] wrote fed_{SYSTEM}.json ({len(glob.glob(os.path.join(RUNS, SYSTEM + '_*.json')))} runs)")
