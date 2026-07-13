"""Dataset generation engine (attack simulation + realistic measurement emission).

Refactored from the validated reference generator into a knob-driven class. Requires pandapower
(pip install 'fdia-graph[generate]'). Benign records are emitted EXACTLY from a stored operating state
(0-error AC flows, no re-solve); only attacks re-solve a power flow. Attack families:
  Ao   state-consistent load redistribution (stealthy)
  ramp temporal creeping redistribution, multi-timestep sequence (stealthy)
  LRA  targeted masked-overload, Yuan/Li/Ren IEEE T-SG 2011 (stealthy)
  Ad/As/Ar  measurement-level corruption (BDD-detectable contrast set)
"""
import numpy as np

FAM_ID = {"benign": 0, "Ao": 1, "Ad": 2, "As": 3, "Ar": 4, "ramp": 5, "LRA": 6}
_CASE = {14: "case14", 118: "case118", 300: "case300"}


class FdiaGenerator:
    def __init__(self, system, seed=123, vbus_frac=0.6, pmu_frac=0.2, flow_frac=0.90):
        import pandapower as pp, pandapower.networks as pn
        from pandapower.pypower.makeYbus import makeYbus
        from pandapower.pypower.makePTDF import makePTDF
        self.pp = pp
        self.C = int(system); self.rng = np.random.default_rng(seed)
        self.SD = dict(pf=0.02, qf=0.02, v=0.005, pi=0.03, qi=0.03, va=0.005)
        self.NET = getattr(pn, _CASE[self.C])
        base = self.NET(); pp.runpp(base); self.base = base
        C = self.C
        self.load_bus = base.load["bus"].values
        inj = np.unique(np.r_[base.gen.bus.values, base.load.bus.values, base.ext_grid.bus.values, base.shunt.bus.values])
        self.zero_inj = [b for b in range(C) if b not in set(inj)]
        self.M = dict(vbus=set(self.rng.choice(C, int(vbus_frac*C), replace=False).tolist()),
                      pmu=set(self.rng.choice(C, max(1, int(pmu_frac*C)), replace=False).tolist()),
                      inj=sorted(set(inj.tolist())))
        self.flow_meter = self.rng.random(len(base.line)+len(base.trafo)) < flow_frac
        self.ei = np.vstack([np.r_[base.line.from_bus.values, base.trafo.hv_bus.values],
                             np.r_[base.line.to_bus.values, base.trafo.lv_bus.values]]).astype(np.int32)
        self.E = self.ei.shape[1]; self.nl = len(base.line)
        self.x_react = np.r_[base.line.x_ohm_per_km.values*base.line.length_km.values, base.trafo.vk_percent.values].astype(np.float32)
        ppc = base._ppc
        self._Ybus, self._Yf, self._Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
        self._bMVA = ppc["baseMVA"]; self._lut = base._pd2ppc_lookups["bus"]
        self._fb = ppc["branch"][:, 0].real.astype(int); self._nppc = ppc["bus"].shape[0]
        genP = {}
        for r in base.gen.itertuples(): genP[int(r.bus)] = genP.get(int(r.bus), 0.0) + r.p_mw
        self.load_genP = np.array([genP.get(int(b), 0.0) for b in self.load_bus])
        self._ptdf = makePTDF(self._bMVA, ppc["bus"], ppc["branch"])
        self._ptdf_lb = self._ptdf[:, [self._lut[b] for b in range(C)]][:, self.load_bus]
        self._solvenet = self.NET()
        self.benign_buf = []

    # ---- emission ----
    def _n(self, s): return self.rng.normal(0, s)

    def emit_from_state(self, X):
        C, SD, M = self.C, self.SD, self.M
        Pi, Qi, V, TH = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        Vc = np.zeros(self._nppc, complex)
        for b in range(C): Vc[self._lut[b]] = V[b]*np.exp(1j*np.deg2rad(TH[b]))
        Sf = Vc[self._fb]*np.conj(self._Yf@Vc)*self._bMVA
        nx = np.zeros((C, 4), np.float32); nm = np.zeros((C, 4), np.uint8)
        for b in range(C):
            if b in M["vbus"] or b in M["pmu"]: nx[b, 0] = V[b]+self._n(SD["v"]); nm[b, 0] = 1
            if b in M["inj"] or b in self.zero_inj:
                nx[b, 1] = Pi[b]+self._n(abs(Pi[b])*SD["pi"]+1e-3); nx[b, 2] = Qi[b]+self._n(abs(Qi[b])*SD["qi"]+1e-3); nm[b, 1:3] = 1
            if b in M["pmu"]: nx[b, 3] = TH[b]+self._n(np.degrees(SD["va"])); nm[b, 3] = 1
        ex = np.zeros((self.E, 2), np.float32); em = np.zeros((self.E, 2), np.uint8)
        for e in range(self.E):
            if self.flow_meter[e]:
                ex[e, 0] = Sf.real[e]+self._n(abs(Sf.real[e])*SD["pf"]+1e-3); ex[e, 1] = Sf.imag[e]+self._n(abs(Sf.imag[e])*SD["qf"]+1e-3); em[e] = 1
        return nx, nm, ex, em

    def emit(self, net):
        C, SD, M = self.C, self.SD, self.M
        Pi = net.res_bus.p_mw.values.copy(); Qi = net.res_bus.q_mvar.values.copy()
        for i in net.shunt.index:
            b = net.shunt.at[i, "bus"]; Pi[b] -= net.res_shunt.p_mw[i]; Qi[b] -= net.res_shunt.q_mvar[i]
        V = net.res_bus.vm_pu.values; TH = net.res_bus.va_degree.values
        nx = np.zeros((C, 4), np.float32); nm = np.zeros((C, 4), np.uint8)
        for b in range(C):
            if b in M["vbus"] or b in M["pmu"]: nx[b, 0] = V[b]+self._n(SD["v"]); nm[b, 0] = 1
            if b in M["inj"] or b in self.zero_inj:
                nx[b, 1] = Pi[b]+self._n(abs(Pi[b])*SD["pi"]+1e-3); nx[b, 2] = Qi[b]+self._n(abs(Qi[b])*SD["qi"]+1e-3); nm[b, 1:3] = 1
            if b in M["pmu"]: nx[b, 3] = TH[b]+self._n(np.degrees(SD["va"])); nm[b, 3] = 1
        Pf = np.r_[net.res_line.p_from_mw.values, net.res_trafo.p_hv_mw.values]
        Qf = np.r_[net.res_line.q_from_mvar.values, net.res_trafo.q_hv_mvar.values]
        ex = np.zeros((self.E, 2), np.float32); em = np.zeros((self.E, 2), np.uint8)
        for e in range(self.E):
            if self.flow_meter[e]:
                ex[e, 0] = Pf[e]+self._n(abs(Pf[e])*SD["pf"]+1e-3); ex[e, 1] = Qf[e]+self._n(abs(Qf[e])*SD["qf"]+1e-3); em[e] = 1
        return nx, nm, ex, em

    def solve(self, Lp, Lq):
        self._solvenet.load["p_mw"] = Lp; self._solvenet.load["q_mvar"] = Lq
        try: self.pp.runpp(self._solvenet); return self._solvenet
        except Exception: return None

    def corrupt(self, nx, ex, atk, kind, replay):
        inc = [e for e in range(self.E) if self.ei[0, e] in atk or self.ei[1, e] in atk]
        for b in atk:
            if kind == "Ad": nx[b, 1:3] += self.rng.normal(0, 0.3*np.abs(nx[b, 1:3])+0.05, 2); nx[b, 0] += self.rng.normal(0, 0.02)
            elif kind == "As": nx[b, 1:3] *= self.rng.uniform(1.25, 1.5)
            elif kind == "Ar" and replay is not None: nx[b, :] = replay[b, :]
        for e in inc:
            if kind == "Ad": ex[e] += self.rng.normal(0, 0.3*np.abs(ex[e])+0.05, 2)
            elif kind == "As": ex[e] *= self.rng.uniform(1.25, 1.5)
        return nx, ex

    # ---- LRA (Yuan et al. 2011) target line + delta ----
    def _lra_for_line(self, L, Lp, rel, K, rand=False):
        # rand=True samples attacked buses from the top-2K high-PTDF candidates so the bus SET varies per
        # record (not memorizable); rand=False (target ranking) stays deterministic.
        pl = self._ptdf_lb[L]; cap = rel*np.abs(Lp); score = np.abs(pl)*cap
        def pick(side):
            side = side[np.argsort(-score[side])]
            if len(side) == 0: return side
            top = side[:2*K]; k = min(K, len(top))
            return self.rng.choice(top, k, replace=False) if rand else top[:k]
        pos = pick(np.where(pl > 0)[0]); neg = pick(np.where(pl < 0)[0])
        if len(pos) == 0 or len(neg) == 0: return None
        up, dn = cap[pos].copy(), cap[neg].copy(); budget = min(up.sum(), dn.sum())
        if budget <= 0: return None
        up *= budget/up.sum(); dn *= budget/dn.sum()
        d = np.zeros_like(Lp); d[pos] = up; d[neg] = -dn
        return d, np.r_[pos, neg], float(-np.sum(pl*d))

    def _pick_lra_target(self, rel, K, n_targets=15):
        # Rank lines by achievable conserving-redistribution flow change and keep the top-`n_targets` as a
        # target POOL. Varying the target per attack (below) diversifies the attacked-bus set so LRA is not
        # trivially localizable (a single fixed target lets a model just memorize those buses).
        bl = self.base.load.p_mw.values
        pot = [(L, self._lra_for_line(L, bl, rel, K)) for L in range(self.nl)]
        pot = [(L, r) for L, r in pot if r is not None]
        pot.sort(key=lambda x: -abs(x[1][2]))
        self._Lcands = [L for L, _ in pot[:min(n_targets, len(pot))]]
        self._sgn = {L: (float(np.sign(self.base.res_line.p_from_mw.values[L])) or 1.0) for L in self._Lcands}
        self._Ltgt = self._Lcands[0]

    def lra_delta(self, Lp, rel, K):
        L = int(self.rng.choice(self._Lcands))                # random target line per attack
        r = self._lra_for_line(L, Lp, rel, K, rand=True)      # + randomized bus subset -> not memorizable
        return (r[0]*self._sgn[L], r[1]) if r is not None else (np.zeros_like(Lp), np.array([], int))
