# -*- coding: utf-8 -*-
"""OPERATIONAL-IMPACT sweep: does a stealthy, BDD-passing FDIA make the operator/EMS misjudge line
thermal safety on IEEE bus systems, and by how much?

For each sampled operating point we compute TRUE line loadings (%) and the loadings the operator would
reconstruct under a stealthy attack (Aq = bounded per-bus load rescale + AGC-balanced AC re-solve; Al =
LRA load redistribution). Because the attacks are state-consistent (validated BDD-stealthy elsewhere), the
operator's WLS state estimate equals the attacked counterfactual, so attacked line loadings = the loadings
of the attacked AC re-solve.

Thermal limits: pandapower's IEEE cases ship with placeholder max_i_ka ~200x the real peak current, so
loading_percent is meaningless as-is. We assign realistic per-line ratings derived from the operating pool:
max_i_ka[line] = (peak |i_ka| over RATING_N nominal pool states) / RATING_RHO, i.e. the busiest line at
nominal peak load sits at RATING_RHO (=0.85 -> 85%). This is a documented, reproducible thermal-limit model,
NOT shipped with the case; it is the same for the nominal and stressed conditions.

Stress: we also run a "stressed operating condition" scaling load+gen by s>1 (mirrors generate_states) so
some lines sit near/over their limits and masked/fabricated overloads become physically possible.

Outputs: results/operator_impact.json (per family, per stress: metrics) and results/operator_impact.npz
(raw per-line loading-error arrays for the figure). Seeded (123)."""
import os, sys, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fdia_graph._core import FdiaGenerator

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
SEED = 123
RATING_N = 300            # nominal pool states used to derive per-line thermal ratings
RATING_RHO = 0.85         # busiest line at nominal peak sits at 85% -> headroom for stress/attacks
CASES = [int(c) for c in os.environ.get("CASES", "118,14").split(",")]
N_PER = int(os.environ.get("N_PER", "400"))          # samples per family per stress condition
STRESS = [1.0, 1.3, 1.5]                              # 1.0 = nominal, >1 = stressed operating condition
INTENSITY = 0.15                                      # Aq per-bus rescale in 1.05..1.15 (~1.10); Al bound 15%
THRESHOLDS = [100.0, 90.0]                            # overload thresholds (operator's safety line)


def get_pool(C):
    p = os.path.join(os.environ.get("FDIA_LOCAL_SHARDS", os.path.join(HERE, "release_v0.4.0")), f"pool_ieee{C}.npz")
    if os.path.exists(p):
        return np.load(p)["X"].astype(np.float64)
    # fallback: build a small pool from the NYISO profile (per task instructions)
    from fdia_graph.profiles import fetch_profile, generate_states
    S = fetch_profile("nyiso", "2024-01-08", "2024-01-12", resample_min=1)
    return generate_states(C, S, seed=SEED).astype(np.float64)


def true_load(g, Xt):
    return (Xt[g.load_bus, 0] + g.load_genP), Xt[g.load_bus, 1].copy()


def stressed_state(g, Xt, s):
    """Stressed operating STATE [N,4]: scale true op-point load AND gen by s, re-solve (mirrors
    generate_states which scales load+gen together). Returns shunt-corrected injection state, or None."""
    if s == 1.0:
        return Xt
    net = g._solvenet
    net.load["p_mw"] = (Xt[g.load_bus, 0] + g.load_genP) * s
    net.load["q_mvar"] = Xt[g.load_bus, 1] * s
    gbus = net.gen["bus"].values
    Lfull = np.zeros(g.C)
    for val, b in zip(Xt[g.load_bus, 0] + g.load_genP, g.load_bus): Lfull[int(b)] += val
    ncnt = {}
    for b in gbus: ncnt[int(b)] = ncnt.get(int(b), 0) + 1
    gp = np.array([(Lfull[int(b)] - Xt[int(b), 0]) / ncnt[int(b)] for b in gbus]) * s
    net.gen["p_mw"] = gp
    net.gen["vm_pu"] = [Xt[int(b), 2] for b in gbus]
    sb = net.ext_grid["bus"].values
    net.ext_grid["vm_pu"] = [Xt[int(b), 2] for b in sb]
    net.ext_grid["va_degree"] = [Xt[int(b), 3] for b in sb]
    try: g.pp.runpp(net)
    except Exception: return None
    Pi = net.res_bus.p_mw.values.copy(); Qi = net.res_bus.q_mvar.values.copy()
    for i in net.shunt.index:
        b = net.shunt.at[i, "bus"]; Pi[b] -= net.res_shunt.p_mw[i]; Qi[b] -= net.res_shunt.q_mvar[i]
    return np.column_stack([Pi, Qi, net.res_bus.vm_pu.values, net.res_bus.va_degree.values])


def derive_ratings(g, X, rng):
    """Per-line max_i_ka = peak nominal current / RATING_RHO, from RATING_N nominal pool states."""
    ipk = np.zeros(g.nl)
    for t in rng.choice(len(X), min(RATING_N, len(X)), replace=False):
        Xt = X[int(t)]; Lp, Lq = true_load(g, Xt)
        net = g.solve(Lp.copy(), Lq.copy(), Xt=Xt, Lp_true=Lp.copy())
        if net is None: continue
        ipk = np.maximum(ipk, net.res_line.i_ka.values)
    ipk = np.maximum(ipk, 1e-6)
    return ipk / RATING_RHO


def loadings(net, g):
    return net.res_line.loading_percent.values[:g.nl].copy()


def run_case(C):
    X = get_pool(C)
    g = FdiaGenerator(C, seed=SEED)
    g._pick_lra_target(INTENSITY, min(6, len(g.load_bus)), n_targets=15)
    rng = np.random.default_rng(SEED)
    ratings = derive_ratings(g, X, np.random.default_rng(SEED))
    g._solvenet.line["max_i_ka"] = ratings        # apply realistic thermal limits to the reused solve net
    nlb = len(g.load_bus)
    print(f"[ieee{C}] pool={X.shape} lines={g.nl} rating rho={RATING_RHO} "
          f"derived max_i_ka med={np.median(ratings):.3f}ka", flush=True)

    out = {}                 # family -> stress -> metrics
    raw = {}                 # npz payload: per-line error arrays
    families = ["benign", "Aq", "Al"]     # benign = null-attack control
    for fam in families:
        out[fam] = {}
        for s in STRESS:
            errs_all = []            # per-line loading error (attacked - true), all lines, all samples
            max_line_err = []        # per-sample max |line loading error| (%pt)
            true_maxload = []        # per-sample true max line loading (%) for context
            masked = {th: 0 for th in THRESHOLDS}
            fabricated = {th: 0 for th in THRESHOLDS}
            flips = {th: 0 for th in THRESHOLDS}
            n = 0; tries = 0
            while n < N_PER and tries < N_PER * 30:
                tries += 1
                Xt0 = X[int(rng.integers(len(X)))]
                Xt = stressed_state(g, Xt0, s)
                if Xt is None: continue
                Lp, Lq = true_load(g, Xt)
                net_t = g.solve(Lp.copy(), Lq.copy(), Xt=Xt, Lp_true=Lp.copy())
                if net_t is None: continue
                Lt = loadings(net_t, g)
                # build attacked loads
                if fam == "benign":
                    Lp_a, Lq_a = Lp.copy(), Lq.copy()          # null attack (control)
                    atk_ok = True
                elif fam == "Aq":
                    k = int(rng.integers(1, min(6, nlb) + 1))
                    a = rng.choice(nlb, k, replace=False)
                    Lp_a = Lp.copy(); Lp_a[a] *= 1 + rng.uniform(0.05, INTENSITY, size=k)
                    Lq_a = Lq.copy(); atk_ok = True
                else:  # Al / LRA
                    d, a = g.lra_delta(Lp, INTENSITY, min(6, nlb))
                    if len(a) == 0: continue
                    Lp_a = Lp + d; Lq_a = Lq.copy(); atk_ok = True
                net_a = g.solve(Lp_a, Lq_a, Xt=Xt, Lp_true=Lp.copy())
                if net_a is None: continue
                La = loadings(net_a, g)
                err = La - Lt
                errs_all.append(err.astype(np.float32))
                max_line_err.append(float(np.max(np.abs(err))))
                true_maxload.append(float(np.nanmax(Lt)))
                for th in THRESHOLDS:
                    t_unsafe = Lt > th; a_unsafe = La > th
                    masked[th] += int(np.sum(t_unsafe & ~a_unsafe))       # truly unsafe, reads safe
                    fabricated[th] += int(np.sum(~t_unsafe & a_unsafe))   # truly safe, reads unsafe
                    flips[th] += int(np.any(t_unsafe != a_unsafe))        # >=1 line flips status
                n += 1
            E = np.concatenate(errs_all) if errs_all else np.zeros(0, np.float32)
            aE = np.abs(E)
            raw[f"ieee{C}_{fam}_s{s}_err"] = E
            raw[f"ieee{C}_{fam}_s{s}_maxline"] = np.array(max_line_err, np.float32)
            out[fam][f"s{s}"] = dict(
                n_samples=n, stress=s,
                mean_abs_line_err_pct=round(float(aE.mean()), 4) if len(aE) else 0.0,
                median_abs_line_err_pct=round(float(np.median(aE)), 4) if len(aE) else 0.0,
                p99_abs_line_err_pct=round(float(np.percentile(aE, 99)), 4) if len(aE) else 0.0,
                max_line_err_pct=round(float(aE.max()), 4) if len(aE) else 0.0,
                mean_sample_max_line_err_pct=round(float(np.mean(max_line_err)), 4) if max_line_err else 0.0,
                pct_samples_ge1_flip={f"th{int(th)}": round(100.0 * flips[th] / max(n, 1), 2) for th in THRESHOLDS},
                masked_overload_events={f"th{int(th)}": masked[th] for th in THRESHOLDS},
                fabricated_overload_events={f"th{int(th)}": fabricated[th] for th in THRESHOLDS},
                median_true_max_loading_pct=round(float(np.median(true_maxload)), 1) if true_maxload else None,
                pct_samples_with_true_overload_th100=round(100.0 * np.mean(np.array(true_maxload) > 100) if true_maxload else 0.0, 1),
            )
            m = out[fam][f"s{s}"]
            print(f"  [{fam:6s} s={s}] n={n} mean|err|={m['mean_abs_line_err_pct']:.3f}%pt "
                  f"maxerr={m['max_line_err_pct']:.2f}%pt flip@100={m['pct_samples_ge1_flip']['th100']}% "
                  f"masked@100={m['masked_overload_events']['th100']} fabr@100={m['fabricated_overload_events']['th100']}",
                  flush=True)
    return out, raw


def main():
    result = {"metric": "line-overload masking: TRUE vs stealthy-attacked pandapower line loadings "
                        "(res_line.loading_percent). Attacked loadings = AC re-solve of the state-consistent "
                        "attack (operator's WLS estimate under a stealthy attack = attacked counterfactual).",
              "thermal_limit_model": f"per-line max_i_ka = peak nominal |i_ka| over {RATING_N} pool states "
                                     f"/ {RATING_RHO} (busiest nominal line ~= {int(RATING_RHO*100)}%); "
                                     "IEEE case ships placeholder limits ~200x too large.",
              "seed": SEED, "n_per_family_per_stress": N_PER, "intensity": INTENSITY,
              "stress_levels": STRESS, "thresholds_pct": THRESHOLDS, "cases": {}}
    all_raw = {}
    for C in CASES:
        o, r = run_case(C)
        result["cases"][f"ieee{C}"] = o
        all_raw.update(r)
    json.dump(result, open(os.path.join(RES, "operator_impact.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(RES, "operator_impact.npz"), **all_raw)
    print(f"\n[done] {os.path.join(RES, 'operator_impact.json')} + operator_impact.npz")


if __name__ == "__main__":
    main()
