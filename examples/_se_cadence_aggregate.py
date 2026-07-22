#!/usr/bin/env python
"""Aggregate the CONTROLLED cadence experiment into a tidy results JSON + CSV sidecar.

Consumes the per-cadence AC-SE runs produced by _se_cadence_control_driver.py
(results/se_cadence_ctrl_ieee{CASE}_cad{CAD}.json), one file per cadence, each holding
3-seed per-seed benign + attacked angle/voltage MAE for Static-WLS / AC-EKF / Learned-AC.

It builds, for every (cadence, system, scenario, estimator, channel):
  - the 3-seed mean +/- std MAE (angle in deg, voltage in p.u.), and
  - a COMBINED error metric = equal-weight mean of the benign and attacked MAE, computed
    PER SEED then averaged (so seed spread propagates into the combined std), and
  - the win ratio vs Static-WLS on the combined metric (combined_WLS / combined_est; >1 = beats WLS).

METERED vs UNMETERED note (honest): the AC measurement set is [Pinj, Qinj, Pf, Qf (all buses/branches),
|V| at every non-slack bus]. So on the VOLTAGE channel every non-slack bus carries a direct |V| PMU
meter -> reported voltage MAE is the METERED-bus error. On the ANGLE channel NO bus has a direct angle
meter (angle is inferred from P/Q flows) -> reported angle MAE is the UNMETERED-bus error. The bus_set
column records this mapping; a further metered-vs-unmetered split within a channel is degenerate in this
estimator (all-metered on |V|, all-unmetered on angle) so it is not fabricated.

Output: results/se_controlled_cadence.json + results/se_controlled_cadence.csv (also copied to
handmade/results/ for the paper build). No plotting here; data only (figures restyled later from the CSV)."""
import os, json, glob, csv, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
EST = [("wls", "Static-WLS"), ("ekf", "AC-EKF"), ("learned", "Learned-AC")]
CHAN = [("th", "angle", "deg", "unmetered (no direct angle meter; inferred from P/Q flows)"),
        ("v", "voltage", "p.u.", "metered (direct |V| PMU at every non-slack bus)")]

files = sorted(glob.glob(os.path.join(RES, "se_cadence_ctrl_ieee*_cad*.json")))
if not files:
    raise SystemExit("no se_cadence_ctrl_ieee*_cad*.json found -- run _se_cadence_control_driver.py first")

runs = []
for f in files:
    d = json.load(open(f))
    runs.append(d)

# group by system
by_sys = {}
for d in runs:
    by_sys.setdefault(d["system"], []).append(d)

out = {"experiment": "controlled measurement-cadence sweep, AC recursive state estimation",
       "controlled_variable": "measurement cadence (s)", "held_fixed": [
           "IEEE case + topology", "real 5-min NYISO slow anchor + calibrated OU fast-load band (0.4%, tau=30s)",
           "256-step sequence length", "estimator architectures + training recipe", "seeds 123/124/125",
           "stealthy angle-ramp attack (0.6-1.8 deg, 30-120 s physical duration)"],
       "combined_metric": "equal-weight mean of benign and attacked MAE, per channel (angle, voltage)",
       "metered_unmetered_note": "voltage channel = metered buses (|V| PMU everywhere); angle channel = unmetered buses",
       "systems": {}}
csv_rows = []

for sys_name in sorted(by_sys):
    ds = sorted(by_sys[sys_name], key=lambda x: x["cadence_s"])
    sys_out = {"per_cadence": {}}
    for d in ds:
        cad = d["cadence_s"]; ps = d["per_seed"]; nseed = len(ps)
        cad_out = {"n_seeds": nseed, "phi_ou": d.get("phi_ou"), "n_anchors": d.get("n_anchors"),
                   "seq_len": d.get("seq_len"), "n_free_buses": d["n_free_buses"], "estimators": {}}
        # combined metric baseline (WLS) per channel, per seed, for win ratios
        wls_comb = {}
        for ck, cname, unit, busset in CHAN:
            wls_comb[cname] = np.array([0.5 * (ps[i]["benign"][f"{ck}_wls"] + ps[i]["attacked"][f"{ck}_wls"])
                                        for i in range(nseed)])
        for ek, ename in EST:
            est_out = {}
            for ck, cname, unit, busset in CHAN:
                ben = np.array([ps[i]["benign"][f"{ck}_{ek}"] for i in range(nseed)])
                atk = np.array([ps[i]["attacked"][f"{ck}_{ek}"] for i in range(nseed)])
                comb = 0.5 * (ben + atk)
                win = float(np.mean(wls_comb[cname] / comb))            # >1 = beats WLS combined
                rec = {"benign": {"mae_mean": round(float(ben.mean()), 6), "mae_std": round(float(ben.std()), 6)},
                       "attacked": {"mae_mean": round(float(atk.mean()), 6), "mae_std": round(float(atk.std()), 6)},
                       "combined": {"mae_mean": round(float(comb.mean()), 6), "mae_std": round(float(comb.std()), 6)},
                       "win_ratio_vs_wls_combined": round(win, 3), "unit": unit, "bus_set": busset}
                est_out[cname] = rec
                for scen, arr in (("benign", ben), ("attacked", atk), ("combined", comb)):
                    csv_rows.append(dict(cadence_s=cad, system=sys_name, scenario=scen, estimator=ename,
                                         channel=cname, unit=unit,
                                         mae_mean=round(float(arr.mean()), 6), mae_std=round(float(arr.std()), 6),
                                         bus_set=busset,
                                         win_ratio_vs_wls_combined=(round(win, 3) if scen == "combined" else "")))
            cad_out["estimators"][ename] = est_out
        sys_out["per_cadence"][str(cad)] = cad_out
    out["systems"][sys_name] = sys_out

jp = os.path.join(RES, "se_controlled_cadence.json")
json.dump(out, open(jp, "w"), indent=2)
cp = os.path.join(RES, "se_controlled_cadence.csv")
cols = ["cadence_s", "system", "scenario", "estimator", "channel", "unit", "mae_mean", "mae_std",
        "bus_set", "win_ratio_vs_wls_combined"]
with open(cp, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for r in sorted(csv_rows, key=lambda r: (r["system"], r["cadence_s"], r["channel"], r["estimator"], r["scenario"])):
        w.writerow(r)

# copy to handmade/results for the paper build
hm = r"C:/Users/bm539044/desktop/fedpig/handmade/results"
if os.path.isdir(hm):
    for src in (jp, cp):
        json.dump(out, open(os.path.join(hm, "se_controlled_cadence.json"), "w"), indent=2) if src == jp else None
    import shutil; shutil.copy(cp, os.path.join(hm, "se_controlled_cadence.csv"))

print(f"wrote {jp}\nwrote {cp}")
# quick console verdict table
for sys_name in sorted(out["systems"]):
    print(f"\n=== {sys_name} ===")
    pc = out["systems"][sys_name]["per_cadence"]
    for cad in sorted(pc, key=float):
        c = pc[cad]; print(f"  cadence {cad}s (phi_ou={c['phi_ou']}, {c['n_seeds']} seeds)")
        for ename in ("Static-WLS", "AC-EKF", "Learned-AC"):
            e = c["estimators"][ename]
            a = e["angle"]; v = e["voltage"]
            print(f"    {ename:11s} angle: ben {a['benign']['mae_mean']:.4f} atk {a['attacked']['mae_mean']:.4f} "
                  f"comb {a['combined']['mae_mean']:.4f} win {a['win_ratio_vs_wls_combined']:.2f}x | "
                  f"volt comb {v['combined']['mae_mean']:.5f} win {v['win_ratio_vs_wls_combined']:.2f}x")
