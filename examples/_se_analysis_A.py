#!/usr/bin/env python
"""Part A: quantify WHY learned SE does not uniformly beat WLS on fdia-graph v0.4.x.

CPU-only (no GPU contention). For each system C in {14,118,300}:
  - per-family attack magnitude ||z - z_clean|| in sigma-units (RMS z-score of the corruption over
    ACTIVE measurements); z_clean = h(x_true) via the same AC operator the PINN uses.
  - learned-advantage over WLS on |V| and theta, per family, joined from results/se_*.json.
  - measurement redundancy (active meas / (2N-1)) and angle observability (frac of buses with angle meter).
Writes results/se_analysis.json. Seed 123. Reuses SD from _se_wls_v040.py, physics from _se_pinn_v040.py."""
import os, json, numpy as np, torch, h5py
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import pandapower as pp, pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus

DEV = "cpu"                                                    # force CPU: GPU is shared / busy
HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.environ.get("FDIA_LOCAL_SHARDS", os.path.join(HERE, "release_v0.4.1")); RES = os.path.join(HERE, "results")
FAMILIES = {0: "benign", 1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 5: "At", 6: "Al"}
SD = dict(pf=0.02, qf=0.02, v=0.005, pi=0.03, qi=0.03, va=0.005)   # from _se_wls_v040.py
VA_DEG = np.degrees(SD["va"])                                      # angle-meter sigma in degrees
NETS = {14: pn.case14, 118: pn.case118, 300: pn.case300}
PER_FAM = int(os.environ.get("PER_FAM", "400"))
rng = np.random.default_rng(123)


def build_ac(C):
    base = NETS[C](); pp.runpp(base); _ppc = base._ppc
    _Ybus, _Yf, _Yt = makeYbus(_ppc["baseMVA"], _ppc["bus"], _ppc["branch"]); bMVA = _ppc["baseMVA"]
    lut = base._pd2ppc_lookups["bus"][:C].astype(np.int64)
    fb = _ppc["branch"][:, 0].real.astype(np.int64); nppc = _ppc["bus"].shape[0]
    Ybus = torch.as_tensor(np.asarray(_Ybus.todense()), dtype=torch.complex64)
    Yf = torch.as_tensor(np.asarray(_Yf.todense()), dtype=torch.complex64)
    LUT = torch.as_tensor(lut); FB = torch.as_tensor(fb)

    def ac(Vmag, theta_deg):
        b = Vmag.shape[0]; Vc_pp = torch.polar(Vmag, torch.deg2rad(theta_deg))
        Vc = torch.zeros(b, nppc, dtype=torch.complex64); Vc[:, LUT] = Vc_pp
        Sbus = Vc * torch.conj(Vc @ Ybus.T) * bMVA
        Sf = Vc[:, FB] * torch.conj(Vc @ Yf.T) * bMVA
        return Sbus.real[:, LUT], Sbus.imag[:, LUT], Sf.real, Sf.imag
    return ac


def load(C):
    H5 = os.path.join(REL, f"ml_only_ieee{C}.h5"); POOL = os.path.join(REL, f"pool_ieee{C}.npz")
    with h5py.File(H5, "r") as f:
        d = f["data"]; A = {k: d[k][:] for k in ("node_x", "node_m", "edge_x", "edge_m", "family", "timestep", "split")}
    POOLX = np.load(POOL)["X"].astype(np.float32)
    Xtrue = POOLX[A["timestep"].astype(np.int64)]
    return A, Xtrue


def main():
    out = {"note": "attack magnitude = RMS z-score of (z - h(x_true)) over ACTIVE measurements; "
                    "advantage = WLS_mae - learned_mae (positive = learned wins)", "systems": {}}
    for C in (14, 118, 300):
        ac = build_ac(C); A, Xtrue = load(C)
        se = json.load(open(os.path.join(RES, f"se_{C}.json")))          # learned (PINN w_phys=0.2)
        wls = json.load(open(os.path.join(RES, f"se_{C}_wls.json")))
        N = A["node_x"].shape[1]
        # redundancy + angle observability over test records
        te = np.where(A["split"] == 2)[0]
        nact = A["node_m"][te].sum((1, 2)) + A["edge_m"][te].sum((1, 2))
        redundancy = float(nact.mean() / (2 * N - 1))
        ang_obs = float(A["node_m"][te, :, 3].mean())
        Vsd = float(np.std(Xtrue[te, :, 2])); THsd = float(np.std(Xtrue[te, :, 3]))
        sysrec = {"N": int(N), "redundancy_meas_per_state": round(redundancy, 3),
                  "angle_observability_frac": round(ang_obs, 3),
                  "state_spread_V_std_pu": round(Vsd, 4), "state_spread_theta_std_deg": round(THsd, 3),
                  "per_family": {}}
        for k, name in FAMILIES.items():
            idx = te[A["family"][te] == k]
            if len(idx) == 0: continue
            if len(idx) > PER_FAM: idx = rng.choice(idx, PER_FAM, replace=False)
            nx = A["node_x"][idx]; nm = A["node_m"][idx].astype(bool)
            ex = A["edge_x"][idx]; em = A["edge_m"][idx].astype(bool)
            Vt = torch.as_tensor(Xtrue[idx, :, 2]); THt = torch.as_tensor(Xtrue[idx, :, 3])
            with torch.no_grad():
                Pb, Qb, Pf, Qf = ac(Vt, THt)
            Pb = Pb.numpy(); Qb = Qb.numpy(); Pf = Pf.numpy(); Qf = Qf.numpy()
            # sigma per measurement type (matches WLS construction)
            sV = SD["v"]; sTH = VA_DEG
            sPi = np.maximum(np.abs(Pb) * SD["pi"], 1e-3); sQi = np.maximum(np.abs(Qb) * SD["qi"], 1e-3)
            sPf = np.maximum(np.abs(Pf) * SD["pf"], 1e-3); sQf = np.maximum(np.abs(Qf) * SD["qf"], 1e-3)
            # z-scored corruption at active meters
            zsq = []; nact_r = np.zeros(len(idx))
            def add(res, sig, mask):
                z = (res / sig)
                z = np.where(mask, z * z, 0.0)
                zsq.append(z.sum(1)); nact_r[:] += mask.sum(1)
            add(nx[:, :, 0] - Xtrue[idx, :, 2], sV, nm[:, :, 0])           # V
            add(nx[:, :, 1] - Pb, sPi, nm[:, :, 1])                        # Pinj
            add(nx[:, :, 2] - Qb, sQi, nm[:, :, 2])                        # Qinj
            add(nx[:, :, 3] - Xtrue[idx, :, 3], sTH, nm[:, :, 3])          # theta
            add(ex[:, :, 0] - Pf, sPf, em[:, :, 0])                        # Pf
            add(ex[:, :, 1] - Qf, sQf, em[:, :, 1])                        # Qf
            zsum = np.sum(zsq, 0)
            mag_sigma = float(np.sqrt(zsum / np.maximum(nact_r, 1)).mean())  # RMS z-score per record, meaned
            # join learned + wls per-family metrics
            L = se["per_family"].get(name, {}); W = wls["per_family"].get(name, {})
            lv = L.get("V_mae_se_metered"); lt = L.get("th_mae_se_metered")
            wv = W.get("V_mae_wls"); wt = W.get("th_mae_wls")
            mv = L.get("V_mae_meter"); mt = L.get("th_mae_meter")
            sysrec["per_family"][name] = dict(
                attack_mag_sigma=round(mag_sigma, 2),
                V_learned=lv, V_wls=wv, V_meter=mv,
                th_learned=lt, th_wls=wt, th_meter=mt,
                adv_V=(round(wv - lv, 4) if wv is not None and lv is not None else None),
                adv_th=(round(wt - lt, 3) if wt is not None and lt is not None else None))
        out["systems"][f"ieee{C}"] = sysrec
        print(f"ieee{C}: N={N} redundancy={redundancy:.2f} ang_obs={ang_obs:.2f} THstd={THsd:.2f}deg")
        for nm2, r in sysrec["per_family"].items():
            print(f"  {nm2:7s} mag={r['attack_mag_sigma']:6.2f}s | V L={r['V_learned']} W={r['V_wls']} advV={r['adv_V']}"
                  f" | th L={r['th_learned']} W={r['th_wls']} advth={r['adv_th']}")
    json.dump(out, open(os.path.join(RES, "se_analysis.json"), "w"), indent=2)
    print("\nwrote results/se_analysis.json")


if __name__ == "__main__":
    main()
