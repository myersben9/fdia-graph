"""Dataset generation engine (attack simulation + realistic measurement emission).

Refactored from the validated reference generator into a knob-driven class. Requires pandapower
(pip install 'fdia-graph[generate]'). Benign records are emitted EXACTLY from a stored operating state
(0-error AC flows, no re-solve); only attacks re-solve a power flow. Attack families:
  Aq   stealthy load scaling: bounded per-bus load rescale + AC re-solve with AGC-balanced generation,
       yielding a fully power-flow-consistent counterfactual (stealthy). OUR contribution — the AC/physical
       realization of the state-consistent attack concept (cf. Boyaci et al. 2022 "Ao"; Liu et al. 2011 FDIA)
  ramp temporal creeping load surge that ramps up then back down over a multi-timestep sequence (stealthy);
       ramp attack of Haghshenas, Hasnat & Naeini, "A Temporal Graph Neural Network for Cyber Attack Detection
       and Localization in Smart Grids", IEEE ISGT 2023
  LRA  targeted masked-overload, Yuan/Li/Ren IEEE T-SG 2011 (stealthy)
  Ad/As/Ar  measurement-level corruption (BDD-detectable contrast set)
"""
import numpy as np

# Integer label for each attack family. These IDs are written into the per-bus label tensor `y`
# (0 = clean bus, >0 = attacked bus of that family) and consumed downstream by dataset.py.
# "Ao"/"SLS" are backward-compatible aliases for Aq (id 1) so older scripts/datasets still resolve.
FAM_ID = {"benign": 0, "Aq": 1, "SLS": 1, "Ao": 1, "Ad": 2, "As": 3, "Ar": 4, "ramp": 5, "LRA": 6}
# Maps a bus-count knob (14/118/300) to the pandapower.networks factory name for that IEEE case.
_CASE = {14: "case14", 118: "case118", 300: "case300"}


class FdiaGenerator:
    def __init__(self, system, seed=123, vbus_frac=0.6, pmu_frac=0.2, flow_frac=0.90):
        # pandapower is an optional/heavy dependency, so import lazily inside the constructor
        # (only when someone actually generates data) rather than at module import time.
        import pandapower as pp, pandapower.networks as pn
        # makeYbus builds the complex nodal admittance matrix Y (the grid's electrical topology);
        # makePTDF builds the Power Transfer Distribution Factors (linear line-flow sensitivities to
        # bus injections) used to target the LRA attack at a specific line.
        from pandapower.pypower.makeYbus import makeYbus
        from pandapower.pypower.makePTDF import makePTDF
        self.pp = pp
        # C = bus count (e.g. 118). rng = seeded generator so the whole dataset is reproducible.
        self.C = int(system); self.rng = np.random.default_rng(seed)
        # Per-quantity measurement noise std-devs (relative for flows/injections, absolute for V/angle):
        # pf/qf = branch active/reactive flow, v = voltage magnitude, pi/qi = bus active/reactive
        # injection, va = voltage angle. These model real SCADA/PMU meter accuracy.
        self.SD = dict(pf=0.02, qf=0.02, v=0.005, pi=0.03, qi=0.03, va=0.005)
        # Grab the network factory (e.g. pn.case118) for this case.
        self.NET = getattr(pn, _CASE[self.C])
        # Build one instance and solve its AC power flow to get the base operating point.
        base = self.NET(); pp.runpp(base); self.base = base
        C = self.C
        # Bus indices that carry a load (their p_mw/q_mvar are what attacks redistribute).
        self.load_bus = base.load["bus"].values
        # Every bus that has SOME injection element attached (generator, load, slack/ext_grid, shunt).
        inj = np.unique(np.r_[base.gen.bus.values, base.load.bus.values, base.ext_grid.bus.values, base.shunt.bus.values])
        # Zero-injection buses: pure junctions with no source/sink. Their net injection is physically
        # exactly 0, a strong known constraint — we still emit a (near-zero) injection measurement there.
        self.zero_inj = [b for b in range(C) if b not in set(inj)]
        # Sparse-metering plan (which buses are actually instrumented), sampled once per generator:
        #   vbus = buses with a voltage-magnitude meter (fraction vbus_frac)
        #   pmu  = buses with a PMU (gives voltage magnitude AND phase angle) (fraction pmu_frac)
        #   inj  = buses whose P/Q injection is metered (all injection buses)
        self.M = dict(vbus=set(self.rng.choice(C, int(vbus_frac*C), replace=False).tolist()),
                      pmu=set(self.rng.choice(C, max(1, int(pmu_frac*C)), replace=False).tolist()),
                      inj=sorted(set(inj.tolist())))
        # Boolean mask over all branches (lines + trafos): which branches have a flow meter (fraction
        # flow_frac). Branches without a meter emit no edge feature (masked out).
        self.flow_meter = self.rng.random(len(base.line)+len(base.trafo)) < flow_frac
        # Edge index (2 x E): row 0 = from-bus, row 1 = to-bus. Lines use from/to; trafos use hv/lv.
        # The two element types are concatenated so branches share one contiguous indexing 0..E-1.
        self.ei = np.vstack([np.r_[base.line.from_bus.values, base.trafo.hv_bus.values],
                             np.r_[base.line.to_bus.values, base.trafo.lv_bus.values]]).astype(np.int32)
        # E = total branch count; nl = number of lines (first nl columns of ei are lines).
        self.E = self.ei.shape[1]; self.nl = len(base.line)
        # Series reactance per branch (line: x_per_km * length; trafo: vk% as a proxy). Kept for reference.
        self.x_react = np.r_[base.line.x_ohm_per_km.values*base.line.length_km.values, base.trafo.vk_percent.values].astype(np.float32)
        # pandapower's internal PYPOWER case (ppc) holds the numeric bus/branch arrays in ppc ordering.
        ppc = base._ppc
        # Y = nodal admittance; Yf/Yt = "from"/"to" branch-admittance matrices such that the complex
        # branch flow entering from the from-end is Sf = V_from * conj(Yf @ V).
        self._Ybus, self._Yf, self._Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
        # baseMVA = per-unit power base (multiply p.u. by this to get MW/MVAr).
        # _lut maps pandapower bus index -> ppc row index (the two orderings differ — classic footgun).
        self._bMVA = ppc["baseMVA"]; self._lut = base._pd2ppc_lookups["bus"]
        # From-bus (ppc index) of each branch, and the ppc bus count (Vc is built in ppc ordering).
        self._fb = ppc["branch"][:, 0].real.astype(int); self._nppc = ppc["bus"].shape[0]
        # Total generator MW dispatched at each bus (summing multiple gens on the same bus).
        genP = {}
        for r in base.gen.itertuples(): genP[int(r.bus)] = genP.get(int(r.bus), 0.0) + r.p_mw
        # Gen MW aligned to the load-bus ordering — lets attacks reason about net (load - gen) at a bus.
        self.load_genP = np.array([genP.get(int(b), 0.0) for b in self.load_bus])
        # PTDF: rows = branches, cols = buses; entry = sensitivity of that branch's MW flow to an
        # injection at that bus. Linear DC model, used to steer redistribution toward a target line.
        self._ptdf = makePTDF(self._bMVA, ppc["bus"], ppc["branch"])
        # Slice PTDF to (branches x load-buses): reindex ppc-order back to pandapower bus order via _lut,
        # then keep only the load-bus columns (loads are the levers an LRA/Ao attack can move).
        self._ptdf_lb = self._ptdf[:, [self._lut[b] for b in range(C)]][:, self.load_bus]
        # A reusable network instance for re-solving power flows under attacked loads (avoids rebuilds).
        self._solvenet = self.NET()
        # Buffer of recent benign records — replay attacks (Ar) copy an earlier clean snapshot from here.
        self.benign_buf = []

    # ---- emission ----
    # Draw one zero-mean Gaussian noise sample with std `s` (the meter-noise primitive).
    def _n(self, s): return self.rng.normal(0, s)

    def emit_from_state(self, X):
        # Emit a measurement graph DIRECTLY from a stored operating state X (no power-flow re-solve),
        # giving physically exact (0-error) flows before meter noise. X columns = [Pinj, Qinj, |V|, angle].
        C, SD, M = self.C, self.SD, self.M
        Pi, Qi, V, TH = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        # Rebuild the complex bus-voltage phasor vector in ppc ordering: V * e^{j*theta}.
        Vc = np.zeros(self._nppc, complex)
        for b in range(C): Vc[self._lut[b]] = V[b]*np.exp(1j*np.deg2rad(TH[b]))
        # Exact complex branch "from-end" power flow via the physics identity Sf = V_from * conj(Yf @ V),
        # scaled by baseMVA to physical units. Sf.real = P_from (MW), Sf.imag = Q_from (MVAr).
        Sf = Vc[self._fb]*np.conj(self._Yf@Vc)*self._bMVA
        # Node feature/mask buffers: columns [|V|, P_inj, Q_inj, angle]; mask=1 where a meter exists.
        nx = np.zeros((C, 4), np.float32); nm = np.zeros((C, 4), np.uint8)
        for b in range(C):
            # Voltage magnitude AND phase angle are observed at the same buses (vbus OR pmu) — angle is metered
            # wherever voltage is, so theta has the same observability as V. Absolute V noise; SD["va"] is in
            # radians so convert to degrees to match TH before adding angle noise.
            if b in M["vbus"] or b in M["pmu"]:
                nx[b, 0] = V[b]+self._n(SD["v"]); nm[b, 0] = 1
                nx[b, 3] = TH[b]+self._n(np.degrees(SD["va"])); nm[b, 3] = 1
            # Injection buses (and zero-injection junctions) emit P/Q with relative noise (+ small floor
            # so a ~0 injection still gets a tiny nonzero std). zero_inj buses carry near-0 injection.
            if b in M["inj"] or b in self.zero_inj:
                nx[b, 1] = Pi[b]+self._n(abs(Pi[b])*SD["pi"]+1e-3); nx[b, 2] = Qi[b]+self._n(abs(Qi[b])*SD["qi"]+1e-3); nm[b, 1:3] = 1
        # Edge feature/mask buffers: columns [P_from, Q_from]; mask=1 where a flow meter exists.
        ex = np.zeros((self.E, 2), np.float32); em = np.zeros((self.E, 2), np.uint8)
        for e in range(self.E):
            if self.flow_meter[e]:
                # Metered branch flow with relative noise (+ small floor) on P and Q.
                ex[e, 0] = Sf.real[e]+self._n(abs(Sf.real[e])*SD["pf"]+1e-3); ex[e, 1] = Sf.imag[e]+self._n(abs(Sf.imag[e])*SD["qf"]+1e-3); em[e] = 1
        return nx, nm, ex, em

    def emit(self, net):
        # Emit a measurement graph from a SOLVED pandapower net (attacks that re-solve a power flow). We
        # extract the solved operating state and route it through emit_from_state, so an attacked sample and
        # a benign sample are computed by the IDENTICAL measurement function. Emitting flows here from
        # res_line while benign used the Ybus identity left a ~7 MW systematic benign-vs-attack offset that
        # was not the attack; sharing one path removes it, so an alpha=1 (no-op) re-solve matches benign.
        Pi = net.res_bus.p_mw.values.copy(); Qi = net.res_bus.q_mvar.values.copy()
        # Shunts are modeled inside res_bus; subtract shunt draw so Pi/Qi reflect gen/load injection only,
        # matching how the stored states (and emit_from_state) define the injection.
        for i in net.shunt.index:
            b = net.shunt.at[i, "bus"]; Pi[b] -= net.res_shunt.p_mw[i]; Qi[b] -= net.res_shunt.q_mvar[i]
        V = net.res_bus.vm_pu.values; TH = net.res_bus.va_degree.values
        Xsolved = np.column_stack([Pi, Qi, V, TH])           # [N,4] = [Pinj, Qinj, |V|, theta]
        return self.emit_from_state(Xsolved)

    def solve(self, Lp, Lq, Xt=None, Lp_true=None):
        # Set new load P/Q on the reusable net and re-run AC power flow. Returns the solved net, or None
        # if it fails to converge (attacks push loads into non-convergent regions — caller skips those).
        net = self._solvenet
        net.load["p_mw"] = Lp; net.load["q_mvar"] = Lq
        # Pin the generation to the TRUE operating point's dispatch. Without this the re-solve leaves every
        # generator at its base-case setpoint and dumps the whole load change onto the slack bus, so even a
        # zero-attack re-solve drifts far from the true state (a large slack/generator residual that is NOT
        # the attack). We reconstruct each bus's true generation from the stored injection, Pgen[b] = (true
        # load at b) - (true injection Xt[b,0]), hold it fixed at the UNATTACKED dispatch, and let only the
        # slack absorb the attack's load delta — so an alpha=1 re-solve reproduces the true state and an
        # attack's residual collapses to its actual load footprint plus the minimal slack response.
        if Xt is not None:
            base_load = Lp_true if Lp_true is not None else Lp   # unattacked load -> the fixed dispatch
            Lfull = np.zeros(self.C)                             # total true load per bus (bus-indexed)
            for val, b in zip(base_load, self.load_bus): Lfull[int(b)] += val
            Pinj_true = Xt[:, 0]
            gbus = net.gen["bus"].values
            ncnt = {}
            for b in gbus: ncnt[int(b)] = ncnt.get(int(b), 0) + 1
            # gen bus injection reproduced: net.load(=Lfull+foldedgen) - gen = Xt[b,0]; split across co-located gens
            gp = np.array([(Lfull[int(b)] - Pinj_true[int(b)]) / ncnt[int(b)] for b in gbus], float)
            # Distribute the attack's net load change across generators (participation factor proportional to
            # dispatch, like AGC) instead of dumping it all on the slack. This keeps the counterfactual a
            # plausible, generation-balanced operating point — the most stealthy realization — so the attack's
            # measurement footprint is the attacked loads plus a small spread, not a large single-bus slack spike.
            dL = float(np.sum(Lp) - np.sum(base_load))          # net extra load introduced by the attack
            tot = gp.sum()
            if tot > 0 and dL != 0.0: gp = gp + dL * (gp / tot)
            net.gen["p_mw"] = gp
            net.gen["vm_pu"] = [Xt[int(b), 2] for b in gbus]     # hold each gen at its true voltage setpoint
            sb = net.ext_grid["bus"].values                      # pin the slack reference to the true voltage/angle
            net.ext_grid["vm_pu"] = [Xt[int(b), 2] for b in sb]
            net.ext_grid["va_degree"] = [Xt[int(b), 3] for b in sb]
        try: self.pp.runpp(net); return net
        except Exception: return None

    def corrupt(self, nx, ex, atk, kind, replay):
        # Measurement-level attacks (the BDD-DETECTABLE contrast families). They perturb the already-emitted
        # measurements at attacked buses `atk` and their incident branches, WITHOUT respecting power-flow
        # physics — which is exactly why bad-data detection can catch them.
        # Incident edges: any branch with either endpoint in the attacked-bus set.
        inc = [e for e in range(self.E) if self.ei[0, e] in atk or self.ei[1, e] in atk]
        for b in atk:
            # Ad = random additive corruption: large relative Gaussian noise on P/Q injection + a jolt on V.
            if kind == "Ad": nx[b, 1:3] += self.rng.normal(0, 0.3*np.abs(nx[b, 1:3])+0.05, 2); nx[b, 0] += self.rng.normal(0, 0.02)
            # As = scaling attack: multiply injections by a 1.25-1.5x gain.
            elif kind == "As": nx[b, 1:3] *= self.rng.uniform(1.25, 1.5)
            # Ar = replay: overwrite this bus's features with a stored earlier (clean) snapshot's values.
            elif kind == "Ar" and replay is not None: nx[b, :] = replay[b, :]
        for e in inc:
            # Apply the matching corruption to incident branch flow measurements (Ad additive, As scaling).
            if kind == "Ad": ex[e] += self.rng.normal(0, 0.3*np.abs(ex[e])+0.05, 2)
            elif kind == "As": ex[e] *= self.rng.uniform(1.25, 1.5)
        return nx, ex

    # ---- LRA (Yuan et al. 2011) target line + delta ----
    def _lra_for_line(self, L, Lp, rel, K, rand=False):
        # rand=True samples attacked buses from the top-2K high-PTDF candidates so the bus SET varies per
        # record (not memorizable); rand=False (target ranking) stays deterministic.
        # Load Redistribution Attack for a chosen target line L: craft a load-injection delta that is
        # (a) LOAD-CONSERVING (total load unchanged -> looks like a normal re-dispatch), (b) PER-BUS
        # BOUNDED (|delta_b| <= rel*|Lp_b|, physically plausible), and (c) steers flow on line L via PTDF.
        # pl = PTDF row for line L over load buses (each load's marginal effect on line-L flow).
        pl = self._ptdf_lb[L]; cap = rel*np.abs(Lp); score = np.abs(pl)*cap
        def pick(side):
            # Rank candidate buses on one PTDF sign side by score (how much flow-change each can buy) and
            # keep the strongest K (deterministic) or a random K of the top-2K (randomized).
            side = side[np.argsort(-score[side])]
            if len(side) == 0: return side
            top = side[:2*K]; k = min(K, len(top))
            return self.rng.choice(top, k, replace=False) if rand else top[:k]
        # Split load buses by PTDF sign: positive buses increase line-L flow, negative buses decrease it.
        # We RAISE load on the positive side and DROP it on the negative side to push flow up on line L.
        pos = pick(np.where(pl > 0)[0]); neg = pick(np.where(pl < 0)[0])
        if len(pos) == 0 or len(neg) == 0: return None
        # Per-bus caps for each side; the conserved budget is the smaller of the two side capacities so the
        # raise on `pos` can be exactly cancelled by the drop on `neg` (net load change = 0).
        up, dn = cap[pos].copy(), cap[neg].copy(); budget = min(up.sum(), dn.sum())
        if budget <= 0: return None
        # Scale each side to hit exactly `budget` MW moved, preserving per-bus proportions.
        up *= budget/up.sum(); dn *= budget/dn.sum()
        # Assemble the full load-delta vector: +up on positive buses, -dn on negative buses (sums to 0).
        d = np.zeros_like(Lp); d[pos] = up; d[neg] = -dn
        # Return (delta, attacked-bus indices, achieved line-L flow change = -sum(PTDF*delta)).
        return d, np.r_[pos, neg], float(-np.sum(pl*d))

    def _pick_lra_target(self, rel, K, n_targets=15):
        # Rank lines by achievable conserving-redistribution flow change and keep the top-`n_targets` as a
        # target POOL. Varying the target per attack (below) diversifies the attacked-bus set so LRA is not
        # trivially localizable (a single fixed target lets a model just memorize those buses).
        # Use base-case loads to evaluate each line's attack potential once, up front.
        bl = self.base.load.p_mw.values
        # For every line, compute its best conserving delta and the flow change it achieves.
        pot = [(L, self._lra_for_line(L, bl, rel, K)) for L in range(self.nl)]
        pot = [(L, r) for L, r in pot if r is not None]
        # Sort by |achieved flow change| descending — the most attackable lines first.
        pot.sort(key=lambda x: -abs(x[1][2]))
        # Keep the top-n as the candidate target pool.
        self._Lcands = [L for L, _ in pot[:min(n_targets, len(pot))]]
        # Sign of each candidate line's base flow (fallback +1) so the attack pushes flow in the direction
        # that WORSENS the existing loading (masking a real overload rather than relieving it).
        self._sgn = {L: (float(np.sign(self.base.res_line.p_from_mw.values[L])) or 1.0) for L in self._Lcands}
        # Default/primary target = most attackable line.
        self._Ltgt = self._Lcands[0]

    def lra_delta(self, Lp, rel, K):
        L = int(self.rng.choice(self._Lcands))                # random target line per attack
        r = self._lra_for_line(L, Lp, rel, K, rand=True)      # + randomized bus subset -> not memorizable
        # Apply the line's base-flow sign so the redistribution masks (not relieves) its overload; if no
        # feasible delta exists, return a zero delta and empty attacked-bus set (record stays effectively benign).
        return (r[0]*self._sgn[L], r[1]) if r is not None else (np.zeros_like(Lp), np.array([], int))
