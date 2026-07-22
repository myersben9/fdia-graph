"""5-seed aggregation + significance test for the ml_centralized ladder + temporal baselines.
Answers the reviewer: are the differences real? XGBoost(no graph) vs ARMA+attn(graph) vs GCN."""
import json, glob, math, statistics as st

def load(d, pat):
    fs = sorted(glob.glob(f"results/{d}/{pat}.json"))
    out = {}
    for f in fs:
        j = json.load(open(f)); out[j.get("seed", f)] = j
    return out

def swf1_list(d, pat):
    return [j["swf1"] for j in load(d, pat).values()]

def welch(a, b):
    if len(a) < 2 or len(b) < 2: return None
    ma, mb = st.mean(a), st.mean(b); va, vb = st.pvariance(a, ma) or 1e-12, st.pvariance(b, mb) or 1e-12
    va, vb = st.variance(a), st.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(va/na + vb/nb)
    if se == 0: return {"t": float("inf"), "df": na+nb-2, "mean_diff": ma-mb, "p_approx": 0.0}
    t = (ma - mb) / se
    df = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
    # two-sided p via survival of t-dist approximated by normal for reporting (df small, so also give |t|)
    from math import erf
    p_normal = 2 * (1 - 0.5*(1+erf(abs(t)/math.sqrt(2))))
    return {"t": round(t, 3), "df": round(df, 1), "mean_diff": round(ma-mb, 4), "p_normal_approx": round(p_normal, 4)}

SYS = ["14", "118", "300"]
models = {
    "mlp_full14": ("ladder", "mlp_full14_ieee{S}_seed*"),
    "gcn_full14": ("ladder", "gcn_full14_ieee{S}_seed*"),
    "xgb": ("temporal", "ieee{S}_xgb_s*"),
    "tcn": ("temporal", "ieee{S}_tcn_s*"),
    "gru": ("temporal", "ieee{S}_gru_s*"),
}
# ARMA+attn reference per-seed (seeds 0-2) from corrected_ladder.json
arma = {}
try:
    ref = json.load(open("results/corrected_ladder.json")).get("arma_reference_seeds012", {}).get("armaattn", {})
    for s in SYS:
        ps = ref.get(s, {}).get("per_seed", {})
        arma[s] = [v for v in ps.values()]
except Exception as e:
    print("arma ref load issue:", e)

agg = {}
print("=== 5-SEED LADDER (swf1 mean +/- std, n) ===")
for m, (d, pat) in models.items():
    agg[m] = {}
    row = []
    for s in SYS:
        v = swf1_list(d, pat.format(S=s))
        agg[m][s] = {"vals": v, "mean": round(st.mean(v), 3), "std": round(st.pstdev(v), 3), "n": len(v)}
        row.append(f"{s}:{agg[m][s]['mean']:.3f}±{agg[m][s]['std']:.3f}(n{len(v)})")
    print(f"  {m:12s} " + "  ".join(row))
print(f"  {'armaattn(ref)':12s} " + "  ".join(f"{s}:{round(st.mean(arma[s]),3) if arma.get(s) else '--'}(n{len(arma.get(s,[]))})" for s in SYS))

print("\n=== SIGNIFICANCE (Welch t, two-sided normal-approx p; n small so read |t| too) ===")
sig = {}
for s in SYS:
    x = agg["xgb"][s]["vals"]; g = agg["gcn_full14"][s]["vals"]; mlp = agg["mlp_full14"][s]["vals"]; a = arma.get(s, [])
    sig[s] = {
        "xgb_vs_arma": welch(x, a),      # expect NOT significant -> graph doesn't help
        "xgb_vs_gcn": welch(x, g),       # expect significant -> XGBoost >> GCN
        "mlpfull_vs_gcn": welch(mlp, g), # expect significant
    }
    print(f"IEEE-{s}:")
    print(f"   XGB vs ARMA+attn: {sig[s]['xgb_vs_arma']}   <- want p>0.05 (graph not significantly better)")
    print(f"   XGB vs GCN:       {sig[s]['xgb_vs_gcn']}   <- want p<0.05 (XGB significantly better)")
    print(f"   MLP-full vs GCN:  {sig[s]['mlpfull_vs_gcn']}")

# per-family At at 5 seeds (featureful models)
print("\n=== per-family At at 5 seeds (frozen frontier check) ===")
for m, (d, pat) in models.items():
    if m in ("mlp_full14", "xgb", "tcn", "gru"):
        line = []
        for s in SYS:
            ats = [j.get("per_family_swf1", {}).get("At") for j in load(d, pat.format(S=s)).values()]
            ats = [a for a in ats if a is not None]
            line.append(f"{s}:{round(st.mean(ats),3) if ats else '--'}")
        print(f"  {m:12s} At = " + "  ".join(line))

json.dump({"ladder": agg, "arma_ref": {s: arma.get(s) for s in SYS}, "significance": sig},
          open("results/ladder_5seed_significance.json", "w"), indent=1)
print("\nwrote results/ladder_5seed_significance.json")
