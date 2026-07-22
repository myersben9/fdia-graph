#!/usr/bin/env python
"""Task 2 -- aggregate the AC accuracy-class noise sweep (results/sweep_pinn_ac/, CASE x AC_CLASS x
W_PHYS x SEED) to answer: does the physics penalty (w_phys>0) help the graph SE at REALISTIC
accuracy-class noise, or does the physics-null finding hold across the whole noise range?

For each (system, ac_class_pct) we average the SE angle MAE over seeds at every w_phys, then compare
w_phys=0 against the BEST w_phys>0. Reported metrics:
  benign  = per_family.benign.th_mae_se_metered           (clean SE angle MAE at metered buses, deg)
  attacked= overall.th_mae_se_metered_attacked            (SE angle MAE under the attack mix, deg)
  V       = per_family.benign.V_mae_se_metered            (clean SE voltage MAE, pu)
'helps' means best-w>0 lowers the metric vs w=0 by more than a small tolerance. Writes
results/ac_noise_summary.json and prints a per-(system, class) verdict table."""
import os, json, glob, numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SW = os.path.join(HERE, "results", "sweep_pinn_ac")
OUT = os.path.join(HERE, "results", "ac_noise_summary.json")

METRICS = {
    "benign_th_se_deg":   ("per_family", "benign", "th_mae_se_metered"),
    "attacked_th_se_deg": ("overall", None, "th_mae_se_metered_attacked"),
    "benign_V_se_pu":     ("per_family", "benign", "V_mae_se_metered"),
}
REL_TOL = 2.0   # percent; a change smaller than this is "flat / null"


def dig(d, path):
    sect, fam, key = path
    node = d[sect]
    if fam is not None:
        node = node[fam]
    return node.get(key)


# group runs: (system, ac_class, w_phys) -> {metric: [values over seeds]}
groups = defaultdict(lambda: defaultdict(list))
n_files = 0
for fp in sorted(glob.glob(os.path.join(SW, "*.json"))):
    try:
        d = json.load(open(fp))
    except Exception as e:
        print("skip", os.path.basename(fp), e); continue
    n_files += 1
    sysk = d["system"]; ac = float(d["ac_class_pct"]); wp = float(d["w_phys"])
    for mname, path in METRICS.items():
        v = dig(d, path)
        if v is not None:
            groups[(sysk, ac)][(wp, mname)].append(float(v))

# per (system, ac_class): mean over seeds at each w_phys, then w0 vs best w>0
summary = {}
verdict_rows = []
for (sysk, ac) in sorted(groups.keys()):
    g = groups[(sysk, ac)]
    wvals = sorted({wp for (wp, _m) in g.keys()})
    per_metric = {}
    for mname in METRICS:
        means = {}
        for wp in wvals:
            vals = g.get((wp, mname), [])
            if vals:
                means[wp] = {"mean": round(float(np.mean(vals)), 5),
                             "std": round(float(np.std(vals)), 5), "n": len(vals)}
        if 0.0 not in means:
            continue
        base = means[0.0]["mean"]
        pos = {w: v for w, v in means.items() if w > 0}
        if not pos:
            continue
        best_w = min(pos, key=lambda w: pos[w]["mean"])
        best = pos[best_w]["mean"]
        rel = 100.0 * (best - base) / base if base != 0 else 0.0
        helps = rel < -REL_TOL          # physics lowers error by more than tolerance
        per_metric[mname] = {
            "w0_mean": base, "best_wpos": best_w, "best_wpos_mean": best,
            "rel_change_pct": round(rel, 2),
            "verdict": "physics helps" if helps else ("physics hurts" if rel > REL_TOL else "flat/null"),
            "per_w_mean": {str(w): means[w] for w in wvals if w in means},
        }
    summary[f"{sysk}_ac{ac}"] = {"system": sysk, "ac_class_pct": ac, "metrics": per_metric}
    if "benign_th_se_deg" in per_metric and "attacked_th_se_deg" in per_metric:
        b = per_metric["benign_th_se_deg"]; a = per_metric["attacked_th_se_deg"]
        verdict_rows.append((sysk, ac, b["w0_mean"], b["best_wpos"], b["best_wpos_mean"], b["rel_change_pct"],
                             b["verdict"], a["rel_change_pct"], a["verdict"]))

# overall verdict: does physics help ANYWHERE at low noise?
any_help = []
for key, blk in summary.items():
    for mname, m in blk["metrics"].items():
        if m["verdict"] == "physics helps":
            any_help.append((key, mname, m["rel_change_pct"], m["best_wpos"]))

out = {
    "n_files": n_files,
    "metric_definitions": {k: ".".join(str(x) for x in v if x is not None) for k, v in METRICS.items()},
    "rel_tol_pct": REL_TOL,
    "per_system_class": summary,
    "physics_helps_anywhere": any_help,
    "overall_verdict": ("physics NULL across all systems and noise classes"
                        if not any_help else
                        f"physics helps in {len(any_help)} (system,class,metric) cells"),
}
json.dump(out, open(OUT, "w"), indent=2)

print(f"\naggregated {n_files} sweep files -> results/ac_noise_summary.json\n")
print(f"{'system':8s} {'class%':>6s} | benign th_SE (deg)  w0 -> best(w>0)   rel%   verdict     | attacked rel%  verdict")
print("-" * 108)
for (sysk, ac, w0, bw, bm, rel, verd, arel, averd) in verdict_rows:
    print(f"{sysk:8s} {ac:6.1f} | {w0:8.4f} -> {bm:8.4f} @w{bw:<4g} {rel:+6.1f}  {verd:12s} | {arel:+6.1f}  {averd}")
print("\nOVERALL:", out["overall_verdict"])
if any_help:
    for (key, mname, rel, bw) in any_help:
        print(f"   physics helps: {key}  {mname}  {rel:+.1f}% @ w={bw}")
