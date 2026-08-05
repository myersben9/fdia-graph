"""Dataset generation engine (attack simulation + realistic measurement emission).

Refactored from the validated reference generator into a knob-driven class. Requires pandapower
(pip install 'fdia-graph[generate]'). Benign records are emitted EXACTLY from a stored operating state
(0-error AC flows, no re-solve); only attacks re-solve a power flow. Attack families:
  Aq   stealthy load scaling: bounded per-bus load rescale + AC re-solve with AGC-balanced generation,
       yielding a fully power-flow-consistent counterfactual (stealthy). OUR contribution — the AC/physical
       realization of the state-consistent attack concept (cf. Boyaci et al. 2022 "Ao"; Liu et al. 2011 FDIA)
  At   (ramp) temporal creeping load surge that ramps up then back down over a multi-timestep sequence (stealthy);
       ramp attack of Haghshenas, Hasnat & Naeini, "A Temporal Graph Neural Network for Cyber Attack Detection
       and Localization in Smart Grids", IEEE ISGT 2023
  Al   (LRA) targeted masked-overload, Yuan/Li/Ren IEEE T-SG 2011 (stealthy)
  Ad/As/Ar  measurement-level corruption (BDD-detectable contrast set)
"""
import numpy as np

# Integer label for each attack family. These IDs are written into the per-bus label tensor `y`
# (0 = clean bus, >0 = attacked bus of that family) and consumed downstream by dataset.py.
# "Ao"/"SLS" are backward-compatible aliases for Aq (id 1) so older scripts/datasets still resolve.
FAM_ID = {"benign": 0, "Aq": 1, "SLS": 1, "Ao": 1, "Ad": 2, "As": 3, "Ar": 4,
          "At": 5, "ramp": 5, "Al": 6, "LRA": 6}  # At=ramp, Al=LRA (with descriptive aliases)
# Maps a bus-count knob (14/118/300) to the pandapower.networks factory name for that IEEE case.
_CASE = {14: "case14", 118: "case118", 300: "case300"}


# ---------------------------------------------------------------------------------------------------------
# N-1 LINE OUTAGE SUPPORT
# A switching event moves the operating manifold: with a line open, the same bus injections produce a
# different voltage/flow state, so a subspace prior (or any model) fitted on the intact network is being
# asked to extrapolate. To measure that we need records generated under a genuinely different topology,
# which means taking the branch out BEFORE anything derived (Ybus, PTDF, the base operating point, the
# emitted measurements) is computed — a post-hoc mask on an intact-network dataset would not do it.
# ---------------------------------------------------------------------------------------------------------
def _line_id(net, outage):
    """Map a line NAME or index to the pandapower line index, with a clear error if it names nothing."""
    if isinstance(outage, str):
        hit = net.line.index[net.line["name"].astype(str) == outage]
        if len(hit) == 0:
            raise ValueError(f"no line named {outage!r} in this case")
        if len(hit) > 1:
            raise ValueError(f"line name {outage!r} is ambiguous ({len(hit)} matches); pass an index")
        return int(hit[0])
    idx = int(outage)
    if idx not in net.line.index:
        raise ValueError(f"line index {idx} is not in this case (lines are {net.line.index.min()}..{net.line.index.max()})")
    return idx


def _n_islands(net):
    """Number of connected components over the IN-SERVICE network (1 == still one connected grid).

    create_nxgraph drops out-of-service branches by default and keeps every bus as a node, so a bus left
    with no live connection shows up as its own component. This is the cheap screen that has to run BEFORE
    generating: pandapower will happily "converge" on an islanded case by silently marking the stranded
    buses isolated (bus type 4) and returning a result, so a solver failure is NOT the signal that a
    contingency was infeasible.
    """
    import networkx as nx
    from pandapower import topology as top
    return int(nx.number_connected_components(top.create_nxgraph(net)))


def line_outage_candidates(system, top_n=5, seed_flow_from=None):
    """Rank single-line N-1 contingencies by base-case active power flow, keeping the network connected.

    Returns (accepted, rejected). `accepted` is the `top_n` highest-flow lines whose removal leaves one
    connected, solvable, PTDF-well-posed network, each a dict with the line index, terminal buses, name and
    signed base-case from-end MW flow. `rejected` lists the higher-flow lines that were screened out and
    why, so a caller can report honestly which contingencies exist but cannot be generated.

    Highest-flow single-line outages are the standard N-1 screening choice (Moshtagh et al. use exactly
    this ranking), which makes the scenario set defensible by citation rather than by taste.
    """
    import pandapower as pp, pandapower.networks as pn
    from pandapower.pypower.makePTDF import makePTDF
    NET = getattr(pn, _CASE[int(system)])
    base = seed_flow_from if seed_flow_from is not None else NET()
    if "p_from_mw" not in base.res_line or base.res_line.empty:
        pp.runpp(base)
    flow = base.res_line["p_from_mw"].to_numpy()
    lut0 = base._pd2ppc_lookups["bus"]
    order = np.argsort(-np.abs(np.nan_to_num(flow)))       # highest |MW| first
    accepted, rejected = [], []
    for pos in order:
        if len(accepted) >= top_n:
            break
        idx = int(base.line.index[pos])
        if not bool(base.line.at[idx, "in_service"]):
            continue                                       # already open in the intact case: not a contingency
        net = NET()
        net.line.at[idx, "in_service"] = False
        why = None
        if _n_islands(net) != 1:
            why = f"removing it splits the grid into {_n_islands(net)} islands"
        else:
            try:
                pp.runpp(net)
            except Exception as e:
                why = f"post-contingency AC power flow does not converge ({type(e).__name__})"
            else:
                n_iso = int((net._ppc["bus"][:, 1].real == 4).sum())
                if n_iso:
                    why = f"leaves {n_iso} isolated bus(es)"
                elif not np.array_equal(lut0, net._pd2ppc_lookups["bus"]):
                    why = "changes the ppc bus ordering (not comparable to the base shard)"
                else:
                    try:
                        makePTDF(net._ppc["baseMVA"], net._ppc["bus"], net._ppc["branch"])
                    except Exception:
                        why = "PTDF is singular (the DC network is islanded)"
        _nm = base.line.at[idx, "name"]
        rec = dict(line=idx, pos=int(pos), from_bus=int(base.line.at[idx, "from_bus"]),
                   to_bus=int(base.line.at[idx, "to_bus"]),
                   name=(f"line{idx}" if _nm is None or str(_nm) in ("None", "nan", "") else str(_nm)),
                   base_flow_mw=float(flow[pos]))
        if why:
            rejected.append({**rec, "reason": why})
        else:
            accepted.append(rec)
    return accepted, rejected


class FdiaGenerator:
    def __init__(self, system, seed=123, vbus_frac=0.6, pmu_frac=0.2, flow_frac=0.90, outage=None):
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
        # Per-quantity measurement noise std-devs (relative for flows/injections, absolute for V/angle).
        # ASPROU ACCURACY-CLASS MEASUREMENT-CHAIN MODEL (Asprou, Kyriakides & Albu 2013 / TIM 2014; the model
        # Falas et al. 2025 delegates to). Each std = manufacturer MAX uncertainty / sqrt(3) (uniform-distribution
        # convention, the paper's eqs 16-21), for CLASS-0.2 instrument transformers (the paper's high-accuracy case):
        #   P/Q: conventional measurement-device max +-3% (Table III)  -> sigma = 3%/sqrt3 ~= 1.73%   (pi/qi/pf/qf)
        #   |V|: VT class-0.2 magnitude error 0.2% (Table I)            -> sigma = 0.2%/sqrt3 ~= 0.12% (abs, V~1 pu)
        #   ang: VT class-0.2 phase displacement 10 arc-min = 0.167 deg -> sigma = 0.167/sqrt3 ~= 0.096 deg
        #        (the VT phase displacement dominates the 0.01 deg PMU-device term). va is stored in radians.
        self.SD = dict(pf=0.017, qf=0.017, v=0.0012, pi=0.017, qi=0.017, va=0.00168)
        # ACCURACY vs PRECISION split. A meter's accuracy-class error is mostly SYSTEMATIC (calibration/ratio
        # offset — constant across scans), with only a small RANDOM repeatability jitter scan-to-scan. Modeling
        # all of SD as independent per-scan noise would make 1-minute traces unrealistically jittery (and drown
        # the temporal channel). So we split: a per-meter BIAS drawn ONCE (0.968*SD, constant over time) + a
        # small per-scan JITTER (0.25*SD). RSS(0.968,0.25)=1.0, so the total error still equals the Asprou class.
        _JIT = 0.25
        self.SDj = {k: v * _JIT for k, v in self.SD.items()}          # per-scan random jitter std
        self._sd_bias = {k: v * (1.0 - _JIT * _JIT) ** 0.5 for k, v in self.SD.items()}  # per-meter bias std
        # Grab the network factory (e.g. pn.case118) for this case.
        self.NET = getattr(pn, _CASE[self.C])
        # N-1 CONTINGENCY. `outage` (a line index or name, None = intact network) takes that line out of
        # service BEFORE the base power flow, so every derived quantity below — Ybus, Yf/Yt, PTDF,
        # edge_status, the base operating point, and therefore every measurement this generator ever emits —
        # describes the post-contingency network. The branch ROW is kept in ppc with status 0, so
        # edge_index, E and the meter plan are identical to the intact case and the two shards are directly
        # comparable: exactly one thing changed.
        self.outage = None; self.outage_pos = -1; self.outage_name = ""
        self.outage_from_bus = -1; self.outage_to_bus = -1; self.outage_base_flow_mw = float("nan")
        base = self.NET()
        if outage is not None:
            self.outage = _line_id(base, outage)
            # Base-case (INTACT) flow on the line we are about to open, recorded so the shard can say how
            # large a contingency it represents.
            intact = self.NET(); pp.runpp(intact)
            self.outage_base_flow_mw = float(intact.res_line.at[self.outage, "p_from_mw"])
            self.outage_pos = int(base.line.index.get_loc(self.outage))
            # IEEE cases from pandapower.networks carry no line names (the column is None), so fall back to
            # a readable "line<idx>" tag rather than storing the string "None" in the shard attrs.
            _nm = base.line.at[self.outage, "name"]
            self.outage_name = f"line{self.outage}" if _nm is None or str(_nm) in ("None", "nan", "") else str(_nm)
            self.outage_from_bus = int(base.line.at[self.outage, "from_bus"])
            self.outage_to_bus = int(base.line.at[self.outage, "to_bus"])
            base.line.at[self.outage, "in_service"] = False
            # Refuse an islanding contingency up front. pandapower would otherwise "converge" with the
            # stranded buses silently marked isolated and hand back a state that is not a power flow
            # solution of anything, and makePTDF would raise a singular matrix mid-build.
            if _n_islands(base) != 1:
                raise ValueError(f"line {self.outage} outage splits the grid into {_n_islands(base)} islands; "
                                 f"screen with line_outage_candidates() before generating")
        # Solve the (possibly post-contingency) AC power flow to get this topology's base operating point.
        pp.runpp(base)
        if self.outage is not None:
            n_iso = int((base._ppc["bus"][:, 1].real == 4).sum())
            if n_iso:
                raise ValueError(f"line {self.outage} outage leaves {n_iso} isolated bus(es)")
        self.base = base
        C = self.C
        # load_bus = the bus of EVERY load element, aligned 1:1 with the net.load table so the AC re-solve
        # (solve() writes net.load["p_mw"] = Lp) and the PTDF-over-load-buses arrays all stay the same length.
        _lb = base.load
        self.load_bus = _lb["bus"].values
        # ATTACKABLE subset = positions in load_bus with real ACTIVE-power load (|p_mw| > 0). A few buses (e.g.
        # IEEE-300 141, 183) carry a reactive-only load (p_mw=0, q_mvar!=0): pandapower lists them as loads, but our
        # attacks scale/redistribute ACTIVE power, so attacking one leaves no P footprint yet still gets a y=1 label
        # (the "no-load bus getting attacked" case). We keep them in the full load table for the physics but exclude
        # them from attack TARGET selection (generation.py) and from the LRA candidate set (_lra_for_line).
        self._attackable_mask = _lb["p_mw"].abs().values > 0.0
        self.attackable_pos = np.where(self._attackable_mask)[0]
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
        # DEPRECATED as of v0.5.0, retained so existing code keeps loading. This mixes UNITS: lines
        # carry series reactance in ohms while transformers carry vk_percent, a short-circuit voltage
        # percentage. On IEEE-300 that puts transformer entries three orders of magnitude above line
        # entries for reasons that are not physical, and 128 of 411 branches are transformers. Use the
        # per-unit edge_* arrays below instead.
        self.x_react = np.r_[base.line.x_ohm_per_km.values*base.line.length_km.values, base.trafo.vk_percent.values].astype(np.float32)
        # pandapower's internal PYPOWER case (ppc) holds the numeric bus/branch arrays in ppc ordering.
        ppc = base._ppc

        # FULL BRANCH AND BUS PHYSICS, per unit, exactly the quantities makeYbus itself consumes, so a
        # model reading them has the same information the state estimator has. ppc branch rows are
        # ordered lines-then-transformers, which is the same order as self.ei, verified by the branch
        # count matching len(line)+len(trafo) on all three systems.
        # BR_G (column 23) is a pandapower extension absent from stock PYPOWER and carries transformer
        # iron losses. Omitting it reconstructs Ybus exactly on IEEE-14, which has none, and wrongly on
        # IEEE-118 and IEEE-300, which have 4 and 18. That is the kind of error that looks correct on
        # the system people test with first, so it is called out here.
        # float64, not float32. These arrays are a few hundred entries, so the storage cost is a few
        # kilobytes, while float32 rounding degrades the Ybus reconstruction from ~1e-14 to ~1e-4 on
        # IEEE-300. An exact reconstruction is the whole claim, so precision wins over a trivial saving.
        _br = ppc["branch"]
        _tap = _br[:, 8].real.astype(np.float64).copy()
        _tap[_tap == 0] = 1.0                       # PYPOWER reads a zero tap entry as unity
        self.edge_r = _br[:, 2].real.astype(np.float64)      # series resistance, p.u.
        self.edge_x = _br[:, 3].real.astype(np.float64)      # series reactance, p.u.
        self.edge_b = _br[:, 4].real.astype(np.float64)      # charging susceptance, p.u.
        self.edge_g = _br[:, 23].real.astype(np.float64)     # charging conductance, p.u. (iron losses)
        self.edge_tap = _tap                                 # transformer turns ratio, 1.0 for lines
        self.edge_shift = _br[:, 9].real.astype(np.float64)  # phase shift, degrees
        self.edge_status = _br[:, 10].real.astype(np.float64)  # 1 in service, 0 out
        self.edge_is_trafo = np.r_[np.zeros(self.nl), np.ones(self.E - self.nl)].astype(np.float64)
        # Shunts sit on the Ybus DIAGONAL, so they are a BUS property and were not expressible in an
        # edge-only schema. Stored in MW and MVAr at 1.0 p.u. voltage, matching the ppc convention.
        self.bus_shunt_g = ppc["bus"][:, 4].real.astype(np.float64)
        self.bus_shunt_b = ppc["bus"][:, 5].real.astype(np.float64)
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
        # The contingency has to be applied here too, or attacked records would be solved on the INTACT
        # network while benign records came from the post-contingency one.
        self._solvenet = self.NET()
        if self.outage is not None:
            self._solvenet.line.at[self.outage, "in_service"] = False
        # Per-meter SYSTEMATIC BIAS, drawn ONCE here so it is CONSTANT across every scan (a fixed calibration
        # offset per meter). Relative for P/Q & flows (a fixed fraction of the reading), absolute for V/angle.
        # This is the slow/systematic part of the accuracy-class error; the small per-scan jitter (self.SDj) is
        # added fresh at emit time. Together they realize the class total while keeping 1-min traces smooth.
        sb = self._sd_bias
        self.bias_pi = self.rng.normal(0, sb["pi"], self.C); self.bias_qi = self.rng.normal(0, sb["qi"], self.C)
        self.bias_v = self.rng.normal(0, sb["v"], self.C);   self.bias_va = self.rng.normal(0, sb["va"], self.C)
        self.bias_pf = self.rng.normal(0, sb["pf"], self.E); self.bias_qf = self.rng.normal(0, sb["qf"], self.E)
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
        # Each reading = true + CONSTANT per-meter bias (self.bias_*, drawn once) + per-scan JITTER (self.SDj).
        # V-magnitude & flow biases are relative to the reading; V/angle biases are absolute (VT ratio error /
        # phase displacement). va bias/jitter are in radians -> converted to degrees to match TH.
        SDj = self.SDj
        for b in range(C):
            # Voltage magnitude AND phase angle are observed at the same buses (vbus OR pmu).
            if b in M["vbus"] or b in M["pmu"]:
                nx[b, 0] = V[b] + self.bias_v[b] + self._n(SDj["v"]); nm[b, 0] = 1
                nx[b, 3] = TH[b] + np.degrees(self.bias_va[b]) + self._n(np.degrees(SDj["va"])); nm[b, 3] = 1
            # Injection buses (and zero-injection junctions) emit P/Q: relative bias (fraction of reading) +
            # relative per-scan jitter (+ small floor so a ~0 injection still gets a tiny nonzero std).
            if b in M["inj"] or b in self.zero_inj:
                nx[b, 1] = Pi[b]*(1.0+self.bias_pi[b]) + self._n(abs(Pi[b])*SDj["pi"]+1e-3)
                nx[b, 2] = Qi[b]*(1.0+self.bias_qi[b]) + self._n(abs(Qi[b])*SDj["qi"]+1e-3); nm[b, 1:3] = 1
        # Edge feature/mask buffers: columns [P_from, Q_from]; mask=1 where a flow meter exists.
        ex = np.zeros((self.E, 2), np.float32); em = np.zeros((self.E, 2), np.uint8)
        for e in range(self.E):
            if self.flow_meter[e]:
                # Metered branch flow: relative per-meter bias + relative per-scan jitter on P and Q.
                ex[e, 0] = Sf.real[e]*(1.0+self.bias_pf[e]) + self._n(abs(Sf.real[e])*SDj["pf"]+1e-3)
                ex[e, 1] = Sf.imag[e]*(1.0+self.bias_qf[e]) + self._n(abs(Sf.imag[e])*SDj["qf"]+1e-3); em[e] = 1
        return nx, nm, ex, em

    def state_from_net(self, net):
        # Pull the operating state [N,4] = [Pinj, Qinj, |V|, theta] out of a SOLVED net, in the same
        # convention the stored pool uses.
        Pi = net.res_bus.p_mw.values.copy(); Qi = net.res_bus.q_mvar.values.copy()
        # Shunts are modeled inside res_bus; subtract shunt draw so Pi/Qi reflect gen/load injection only,
        # matching how the stored states (and emit_from_state) define the injection.
        for i in net.shunt.index:
            b = net.shunt.at[i, "bus"]; Pi[b] -= net.res_shunt.p_mw[i]; Qi[b] -= net.res_shunt.q_mvar[i]
        V = net.res_bus.vm_pu.values; TH = net.res_bus.va_degree.values
        return np.column_stack([Pi, Qi, V, TH])              # [N,4] = [Pinj, Qinj, |V|, theta]

    def emit(self, net):
        # Emit a measurement graph from a SOLVED pandapower net (attacks that re-solve a power flow). We
        # extract the solved operating state and route it through emit_from_state, so an attacked sample and
        # a benign sample are computed by the IDENTICAL measurement function. Emitting flows here from
        # res_line while benign used the Ybus identity left a ~7 MW systematic benign-vs-attack offset that
        # was not the attack; sharing one path removes it, so an alpha=1 (no-op) re-solve matches benign.
        return self.emit_from_state(self.state_from_net(net))

    def resolve_states(self, X):
        """Re-solve a pool of operating points [T,N,4] under THIS generator's topology.

        A stored state carries the injections AND the voltages that the INTACT network produced for them.
        Under a contingency the same loads give a different voltage/flow state, so emitting an intact-network
        state through a post-contingency Ybus would fabricate measurements that satisfy no power flow at all.
        Re-solving fixes that: the loads (and the generator dispatch reconstructed from the stored state) are
        held at exactly the values the base case had at that timestamp, and only the topology differs — which
        is the whole point, since a load profile that shifted between scenarios would confound topology with
        load level and make the comparison meaningless.

        Returns (Xnew [T,N,4], ok [T] bool). Rows where the power flow did not converge, or came back
        non-finite, are left as-is and flagged False rather than quietly dropped, so the caller can intersect
        the converged sets across scenarios and keep one common timestamp axis.
        """
        X = np.asarray(X, dtype=np.float64)
        out = X.copy(); ok = np.zeros(len(X), bool)
        for t in range(len(X)):
            Xt = X[t]
            # Base active load at each load element = stored injection + the generation folded onto that bus.
            Lp = Xt[self.load_bus, 0] + self.load_genP
            Lq = Xt[self.load_bus, 1].copy()
            # Lp_true == Lp: no attack, so this is the alpha=1 no-op re-solve. On the INTACT topology it
            # reproduces the stored state (that identity is the check that the pinning is right); on a
            # contingency topology it is the post-contingency state for the same operating conditions.
            net = self.solve(Lp, Lq, Xt=Xt, Lp_true=Lp)
            if net is None:
                continue
            s = self.state_from_net(net)
            if not np.isfinite(s).all():
                continue
            out[t] = s; ok[t] = True
        return out, ok

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
        # Restrict to ATTACKABLE (active-load) buses so a reactive-only bus is never redistributed onto / labelled.
        pos = pick(np.where((pl > 0) & self._attackable_mask)[0]); neg = pick(np.where((pl < 0) & self._attackable_mask)[0])
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
        # For every line, compute its best conserving delta and the flow change it achieves. An outaged line
        # is skipped explicitly: its PTDF row is zero so it would rank last anyway, but its base-case flow is
        # NaN, and a NaN reaching self._sgn would silently poison every LRA delta on that line.
        pot = [(L, self._lra_for_line(L, bl, rel, K)) for L in range(self.nl) if L != self.outage_pos]
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
