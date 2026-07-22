#!/usr/bin/env python
"""5-seed re-aggregation + significance testing for the ml_centralized ladder and temporal baselines.

Reviewer (GPT) held the paper at 4/10: "2-3 seeds are too few to claim the differences are real."
This script re-aggregates mean+/-std over ALL seeds present on disk (auto-upgrades 3->5 as seeds 126/127
land) for every key rung x system, and runs a per-system paired significance test on the per-seed swf1
values for the three claims that carry the paper's thesis.

PAIRING (the trap to get right):
  mlp_full14 / gcn_full14 / xgb / tcn / gru all index the SAME dataset replicas by seed
  (seed s -> shard ml_only_ieee{C}_s{s}.h5). So per-seed swf1 for these models IS genuinely PAIRED:
  the same random dataset draw underlies model A and model B at seed s -> paired t-test + Wilcoxon.
  ARMA+attn instead uses model-INIT seeds 0,1,2 on the single canonical seed-123 shard (documented
  reuse; std ~0.003). That is a DIFFERENT seed space and a different n, so XGB-vs-ARMA is NOT paired ->
  Welch unpaired t-test + Mann-Whitney U + bootstrap of the mean gap, with the caveat stated.

swf1 = per-sample macro node-F1 on attacked TEST records, threshold tuned on VAL attacked records.
"""
import os, json, glob
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
LAD = os.path.join(RES, "ladder"); TMP = os.path.join(RES, "temporal")
FAM = ["Aq", "Ad", "As", "Ar", "At", "Al"]
SYS = ["14", "118", "300"]
DATA_SEEDS = [123, 124, 125, 126, 127]   # dataset-replica seeds (only those present on disk are used)


def load_ladder(model, C):
    """mlp_full14 / gcn_full14 per-seed swf1 dict {seed: rec} from results/ladder/."""
    out = {}
    for s in DATA_SEEDS:
        fp = os.path.join(LAD, f"{model}_full14_ieee{C}_seed{s}.json")
        if os.path.exists(fp):
            out[s] = json.load(open(fp))
    return out


def load_temporal(model, C):
    """gru / tcn / xgb per-seed swf1 dict {seed: rec} from results/temporal/."""
    out = {}
    for s in DATA_SEEDS:
        fp = os.path.join(TMP, f"ieee{C}_{model}_s{s}.json")
        if os.path.exists(fp):
            out[s] = json.load(open(fp))
    return out


# ARMA+attn reference (model-init seeds 0,1,2 on the canonical seed-123 shard) from corrected_ladder_full.json
ARMA = json.load(open(os.path.join(RES, "corrected_ladder_full.json")))["armaattn_reference"]["armaattn"]


def collect(getter, model):
    """Return {C: {seed: swf1}} and {C: {seed: perfam}} for a data-seeded model."""
    sw, pf = {}, {}
    for C in SYS:
        recs = getter(model, C)
        sw[C] = {s: r["swf1"] for s, r in recs.items()}
        pf[C] = {s: r.get("per_family_swf1", {}) for s, r in recs.items()}
    return sw, pf


def summarize(seed_to_sw, seed_to_pf):
    vals = np.array([seed_to_sw[s] for s in sorted(seed_to_sw)], float)
    seeds = sorted(seed_to_sw)
    fam_mean = {f: float(np.mean([seed_to_pf[s].get(f, np.nan) for s in seeds])) for f in FAM}
    return {
        "n": len(vals), "seeds": seeds,
        "swf1_per_seed": {int(s): round(seed_to_sw[s], 4) for s in seeds},
        "swf1_mean": round(float(np.mean(vals)), 4),
        "swf1_std_sample": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
        "swf1_std_pop": round(float(np.std(vals)), 4),
        "per_family_swf1_mean": {k: round(v, 4) for k, v in fam_mean.items()},
    }


def cohen_d_paired(a, b):
    d = a - b
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if sd > 0 else float("inf") * np.sign(np.mean(d))


def cohen_d_unpaired(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else float("inf")


def paired_test(swA, swB, nameA, nameB):
    """Genuinely paired (same dataset-seed underlies both). Paired t + Wilcoxon signed-rank + paired bootstrap."""
    common = sorted(set(swA) & set(swB))
    a = np.array([swA[s] for s in common], float); b = np.array([swB[s] for s in common], float)
    n = len(common)
    res = {"comparison": f"{nameA} vs {nameB}", "pairing": "paired (shared dataset-seeds)",
           "seeds": [int(s) for s in common], "n_pairs": n,
           "mean_A": round(float(a.mean()), 4), "mean_B": round(float(b.mean()), 4),
           "mean_diff_A_minus_B": round(float((a-b).mean()), 4)}
    if n < 2:
        res["note"] = "insufficient pairs"; return res
    t, p_t = stats.ttest_rel(a, b)
    res["paired_t"] = {"t": round(float(t), 3), "p": round(float(p_t), 4)}
    # Wilcoxon needs non-zero diffs; guard.
    try:
        if np.allclose(a, b):
            res["wilcoxon"] = {"note": "all diffs ~0", "p": 1.0}
        else:
            w, p_w = stats.wilcoxon(a, b)
            res["wilcoxon"] = {"W": round(float(w), 3), "p": round(float(p_w), 4)}
    except Exception as e:
        res["wilcoxon"] = {"error": str(e)}
    res["cohen_dz"] = round(cohen_d_paired(a, b), 3)
    # paired bootstrap on the per-seed differences
    d = a - b; rng = np.random.default_rng(0)
    bs = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(10000)])
    res["bootstrap_meandiff_CI95"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    res["bootstrap_p_two_sided"] = round(float(2*min((bs <= 0).mean(), (bs >= 0).mean())), 4)
    return res


def unpaired_test(swA, swB_dict, nameA, nameB):
    """XGB(data-seeds) vs ARMA(model-init seeds 0-2). Different seed spaces -> Welch + Mann-Whitney + bootstrap."""
    a = np.array([swA[s] for s in sorted(swA)], float)
    b = np.array(list(swB_dict.values()), float)
    res = {"comparison": f"{nameA} vs {nameB}",
           "pairing": "UNPAIRED (different seed spaces: data-seeds vs ARMA model-init seeds 0-2)",
           "n_A": len(a), "n_B": len(b),
           "mean_A": round(float(a.mean()), 4), "mean_B": round(float(b.mean()), 4),
           "mean_diff_A_minus_B": round(float(a.mean()-b.mean()), 4)}
    if len(a) < 2 or len(b) < 2:
        res["note"] = "insufficient samples"; return res
    t, p_t = stats.ttest_ind(a, b, equal_var=False)
    res["welch_t"] = {"t": round(float(t), 3), "p": round(float(p_t), 4)}
    u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
    res["mann_whitney"] = {"U": round(float(u), 3), "p": round(float(p_u), 4)}
    res["cohen_d"] = round(cohen_d_unpaired(a, b), 3)
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean() for _ in range(10000)])
    res["bootstrap_meandiff_CI95"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    res["bootstrap_p_two_sided"] = round(float(2*min((bs <= 0).mean(), (bs >= 0).mean())), 4)
    return res


def main():
    mlp_sw, mlp_pf = collect(load_ladder, "mlp")
    gcn_sw, gcn_pf = collect(load_ladder, "gcn")
    xgb_sw, xgb_pf = collect(load_temporal, "xgb")
    tcn_sw, tcn_pf = collect(load_temporal, "tcn")
    gru_sw, gru_pf = collect(load_temporal, "gru")

    rungs = {"mlp_full14": (mlp_sw, mlp_pf), "gcn_full14": (gcn_sw, gcn_pf),
             "xgboost": (xgb_sw, xgb_pf), "tcn": (tcn_sw, tcn_pf), "gru": (gru_sw, gru_pf)}

    table = {}
    for name, (sw, pf) in rungs.items():
        table[name] = {}
        for C in SYS:
            if sw[C]:
                table[name][C] = summarize(sw[C], pf[C])
    # ARMA rung straight from reference
    table["arma_attn"] = {}
    for C in SYS:
        ps = ARMA[C]["per_seed"]
        table["arma_attn"][C] = {"n": len(ps), "seeds": [int(k) for k in ps],
                                 "seed_space": "model-init 0,1,2 on seed-123 shard (documented reuse)",
                                 "swf1_per_seed": {int(k): v for k, v in ps.items()},
                                 "swf1_mean": ARMA[C]["mean"], "swf1_std_pop": ARMA[C]["std"],
                                 "per_family_swf1_mean": ARMA[C].get("perfam_seed0", {})}

    # ---- significance per system ----
    sig = {}
    for C in SYS:
        sig[C] = {
            "xgb_vs_arma": unpaired_test(xgb_sw[C], ARMA[C]["per_seed"], "XGBoost", "ARMA+attn"),
            "xgb_vs_gcn": paired_test(xgb_sw[C], gcn_sw[C], "XGBoost", "GCN-full14"),
            "mlp_vs_gcn": paired_test(mlp_sw[C], gcn_sw[C], "MLP-full14", "GCN-full14"),
        }

    out = {"note": "5-seed re-aggregation + significance (reviewer response). swf1 = per-sample macro node-F1 "
                   "on attacked test records, val-tuned threshold. data-seed models (mlp/gcn/xgb/tcn/gru) share "
                   "dataset replicas by seed -> paired; ARMA+attn uses model-init seeds 0-2 on seed-123 shard.",
           "rungs": table, "significance": sig}
    json.dump(out, open(os.path.join(RES, "ladder_5seed.json"), "w"), indent=2)

    # ---- console ----
    print("\n=== 5-SEED mean +/- std (sample, ddof=1) swf1 ===")
    print(f"{'rung':12s} {'sys':4s} {'n':>2s}  {'mean':>6s} {'std':>6s}   {'At':>5s}   per-seed")
    for name in ("mlp_full14", "gcn_full14", "xgboost", "tcn", "gru", "arma_attn"):
        for C in SYS:
            d = table[name].get(C)
            if not d: continue
            std = d.get("swf1_std_sample", d.get("swf1_std_pop", 0))
            at = d["per_family_swf1_mean"].get("At", float("nan"))
            ps = ",".join(f"{v:.3f}" for v in d["swf1_per_seed"].values())
            print(f"{name:12s} {C:4s} {d['n']:>2d}  {d['swf1_mean']:.3f} {std:.3f}   {at:.3f}   [{ps}]")

    print("\n=== SIGNIFICANCE (per system) ===")
    for C in SYS:
        print(f"\n--- IEEE-{C} ---")
        for key in ("xgb_vs_arma", "xgb_vs_gcn", "mlp_vs_gcn"):
            r = sig[C][key]
            print(f"  {r['comparison']:26s} [{r['pairing']}]")
            print(f"     mean {r['mean_A']:.4f} vs {r['mean_B']:.4f}  diff {r['mean_diff_A_minus_B']:+.4f}")
            if "paired_t" in r:
                print(f"     paired t p={r['paired_t']['p']:.4f}  wilcoxon p={r['wilcoxon'].get('p','?')}  "
                      f"d_z={r.get('cohen_dz')}  bootstrap p={r.get('bootstrap_p_two_sided')} CI{r.get('bootstrap_meandiff_CI95')}")
            elif "welch_t" in r:
                print(f"     welch t p={r['welch_t']['p']:.4f}  mann-whitney p={r['mann_whitney']['p']:.4f}  "
                      f"d={r.get('cohen_d')}  bootstrap p={r.get('bootstrap_p_two_sided')} CI{r.get('bootstrap_meandiff_CI95')}")
            else:
                print(f"     {r.get('note','')}")
    print("\nwrote results/ladder_5seed.json")


if __name__ == "__main__":
    main()
