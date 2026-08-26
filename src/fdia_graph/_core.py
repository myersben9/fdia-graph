"""Dataset generation engine (attack simulation + realistic measurement emission).

Knob-driven class refactored from the reference generator. Requires pandapower
('fdia-graph[generate]'). Benign records emit EXACTLY from a stored state (0-error AC flows, no
re-solve); only attacks re-solve. Attack families:
  Aq   stealthy load scaling: bounded per-bus rescale + AGC-balanced AC re-solve (power-flow-consistent
       counterfactual). Our AC realization of the state-consistent attack (cf. Boyaci 2022 "Ao"; Liu 2011)
  At   (ramp) temporal creeping load surge up then down over a sequence (Haghshenas/Hasnat/Naeini ISGT 2023)
  Al   (LRA) targeted masked-overload (Yuan/Li/Ren IEEE T-SG 2011)
  Ad/As/Ar  measurement-level corruption (BDD-detectable contrast set)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Integer family label written into the per-bus label tensor `y` (0=clean, >0=attacked of that family).
# "Ao"/"SLS" are back-compat aliases for Aq (id 1); "ramp"=At, "LRA"=Al.
FAM_ID = {"benign": 0, "Aq": 1, "SLS": 1, "Ao": 1, "Ad": 2, "As": 3, "Ar": 4,
          "At": 5, "ramp": 5, "Al": 6, "LRA": 6}
# Bus-count knob (14/118/300) -> pandapower.networks factory name.
# Bus-count -> pandapower.networks builder. Transmission systems only (>=110 kV, meshed). A system is
# load()-able only once its pool + registry entry ship; listing it here just lets the generator build it.
_CASE = {14: "case14", 30: "case30", 57: "case57", 89: "case89pegase", 118: "case118",
         145: "case145", 200: "case_illinois200", 300: "case300"}


# N-1 LINE OUTAGE SUPPORT. The branch must be taken out BEFORE anything derived (Ybus, PTDF, base
# operating point, measurements) is computed — a post-hoc mask on an intact-network dataset won't do.
def _line_id(net: Any, outage: Union[str, int]) -> int:
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


def _n_islands(net: Any) -> int:
    """Number of connected components over the IN-SERVICE network (1 == still one connected grid).

    Cheap islanding screen to run BEFORE generating: pandapower "converges" on an islanded case by
    silently marking stranded buses isolated (bus type 4), so solver failure is NOT the infeasibility signal.
    """
    import networkx as nx
    from pandapower import topology as top
    return int(nx.number_connected_components(top.create_nxgraph(net)))


def line_outage_candidates(system: Union[int, str], top_n: int = 5,
                           seed_flow_from: Any = None) -> Tuple[List[Dict], List[Dict]]:
    """Rank single-line N-1 contingencies by base-case active power flow, keeping the network connected.

    Returns (accepted, rejected). `accepted` = the `top_n` highest-flow lines whose removal leaves one
    connected, solvable, PTDF-well-posed network (dict: line index, terminal buses, name, signed from-end
    MW flow). `rejected` = higher-flow lines screened out, with reason. Highest-flow ranking is the standard
    N-1 screening choice (Moshtagh et al.).
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
            continue                                       # already open: not a contingency
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
    def __init__(self, system: Union[int, str], seed: int = 123, vbus_frac: float = 0.6, pmu_frac: float = 0.2,
                 flow_frac: float = 0.90, outage: Optional[Union[int, str]] = None) -> None:
        # pandapower is heavy/optional: import lazily so it's only needed when actually generating.
        import pandapower as pp, pandapower.networks as pn
        # makeYbus -> complex nodal admittance Y; makePTDF -> linear line-flow sensitivities (for LRA targeting).
        from pandapower.pypower.makeYbus import makeYbus
        from pandapower.pypower.makePTDF import makePTDF
        self.pp = pp
        self.C = int(system); self.rng = np.random.default_rng(seed)
        # ASPROU accuracy-class measurement noise stds (Asprou/Kyriakides/Albu TIM 2014; Falas et al. 2025 uses it).
        # Each std = manufacturer max uncertainty / sqrt(3), class-0.2 instrument transformers. Relative for
        # flows/injections, absolute for V/angle. P/Q ~1.73%; |V| ~0.12%; angle ~0.096 deg (va stored in radians).
        self.SD = dict(pf=0.017, qf=0.017, v=0.0012, pi=0.017, qi=0.017, va=0.00168)
        # Accuracy-class error is mostly SYSTEMATIC (constant calibration offset), with a small per-scan jitter.
        # Treating all of SD as per-scan noise would over-jitter 1-min traces and drown the temporal channel.
        # Split: per-meter BIAS drawn once (0.968*SD) + per-scan JITTER (0.25*SD); RSS = 1.0 keeps the class total.
        _JIT = 0.25
        self.SDj = {k: v * _JIT for k, v in self.SD.items()}          # per-scan random jitter std
        self._sd_bias = {k: v * (1.0 - _JIT * _JIT) ** 0.5 for k, v in self.SD.items()}  # per-meter bias std
        self.NET = getattr(pn, _CASE[self.C])
        # N-1 CONTINGENCY. `outage` (line index/name, None=intact) opens the line BEFORE the base power flow, so
        # every derived quantity (Ybus, Yf/Yt, PTDF, edge_status, base state, all measurements) is post-contingency.
        # The branch ROW is kept in ppc with status 0, so edge_index/E/meter plan match intact — exactly one thing
        # changed, shards directly comparable.
        self.outage = None; self.outage_pos = -1; self.outage_name = ""
        self.outage_from_bus = -1; self.outage_to_bus = -1; self.outage_base_flow_mw = float("nan")
        base = self.NET()
        if outage is not None:
            self.outage = _line_id(base, outage)
            # Record the INTACT flow on the line to be opened, so the shard reports the contingency's size.
            intact = self.NET(); pp.runpp(intact)
            self.outage_base_flow_mw = float(intact.res_line.at[self.outage, "p_from_mw"])
            self.outage_pos = int(base.line.index.get_loc(self.outage))
            # IEEE cases carry no line names (None), so fall back to a "line<idx>" tag.
            _nm = base.line.at[self.outage, "name"]
            self.outage_name = f"line{self.outage}" if _nm is None or str(_nm) in ("None", "nan", "") else str(_nm)
            self.outage_from_bus = int(base.line.at[self.outage, "from_bus"])
            self.outage_to_bus = int(base.line.at[self.outage, "to_bus"])
            base.line.at[self.outage, "in_service"] = False
            # Refuse an islanding contingency up front (else pandapower "converges" on a non-solution and
            # makePTDF hits a singular matrix mid-build).
            if _n_islands(base) != 1:
                raise ValueError(f"line {self.outage} outage splits the grid into {_n_islands(base)} islands; "
                                 f"screen with line_outage_candidates() before generating")
        # Solve the (possibly post-contingency) AC power flow for the base operating point.
        pp.runpp(base)
        if self.outage is not None:
            n_iso = int((base._ppc["bus"][:, 1].real == 4).sum())
            if n_iso:
                raise ValueError(f"line {self.outage} outage leaves {n_iso} isolated bus(es)")
        self.base = base
        C = self.C
        # load_bus = bus of EVERY load element, aligned 1:1 with net.load so the re-solve and PTDF-over-load
        # arrays stay the same length.
        _lb = base.load
        self.load_bus = _lb["bus"].values
        # ATTACKABLE = load_bus positions with real ACTIVE load (|p_mw|>0). Reactive-only loads (e.g. IEEE-300
        # buses 141, 183: p_mw=0, q_mvar!=0) stay in the physics table but are excluded from attack target
        # selection and the LRA candidate set (attacking one leaves no P footprint yet still gets a y=1 label).
        self._attackable_mask = _lb["p_mw"].abs().values > 0.0
        self.attackable_pos = np.where(self._attackable_mask)[0]
        # Every bus with some injection element (gen, load, ext_grid, shunt).
        inj = np.unique(np.r_[base.gen.bus.values, base.load.bus.values, base.ext_grid.bus.values, base.shunt.bus.values])
        # Zero-injection buses: pure junctions with net injection exactly 0 (strong constraint); still emit a
        # near-zero injection measurement there.
        self.zero_inj = [b for b in range(C) if b not in set(inj)]
        # Sparse-metering plan (sampled once): vbus=voltage-magnitude meters, pmu=|V|+angle meters, inj=metered
        # P/Q injection buses (all injection buses).
        self.M = dict(vbus=set(self.rng.choice(C, int(vbus_frac*C), replace=False).tolist()),
                      pmu=set(self.rng.choice(C, max(1, int(pmu_frac*C)), replace=False).tolist()),
                      inj=sorted(set(inj.tolist())))
        # Per-branch flow-meter mask (fraction flow_frac); unmetered branches emit no edge feature.
        self.flow_meter = self.rng.random(len(base.line)+len(base.trafo)) < flow_frac
        # Edge index (2 x E): row0=from-bus, row1=to-bus (lines use from/to, trafos hv/lv). Concatenated so
        # branches share one contiguous 0..E-1 indexing.
        self.ei = np.vstack([np.r_[base.line.from_bus.values, base.trafo.hv_bus.values],
                             np.r_[base.line.to_bus.values, base.trafo.lv_bus.values]]).astype(np.int32)
        self.E = self.ei.shape[1]; self.nl = len(base.line)  # nl = number of lines (first nl cols of ei)
        # DEPRECATED (v0.5.0), retained for loading. Mixes UNITS: line reactance in ohms vs trafo vk_percent,
        # putting trafo entries ~3 orders of magnitude above lines on IEEE-300. Use the per-unit edge_* arrays.
        self.x_react = np.r_[base.line.x_ohm_per_km.values*base.line.length_km.values, base.trafo.vk_percent.values].astype(np.float32)
        ppc = base._ppc  # pandapower's PYPOWER case: numeric bus/branch arrays in ppc ordering

        # Full per-unit branch/bus physics — exactly the quantities makeYbus consumes, so a model has the same
        # info the state estimator does. ppc branch rows are ordered lines-then-transformers, matching self.ei.
        # BR_G (column 23) is a pandapower extension (absent from stock PYPOWER) carrying transformer iron
        # losses. Omitting it reconstructs Ybus exactly on IEEE-14 (no trafos) but WRONGLY on IEEE-118/300 (4/18)
        # — an error that looks correct on the first system people test.
        # float64, not float32: float32 rounding degrades Ybus reconstruction from ~1e-14 to ~1e-4 on IEEE-300,
        # and exact reconstruction is the whole claim.
        _br = ppc["branch"]
        _tap = _br[:, 8].real.astype(np.float64).copy()
        _tap[_tap == 0] = 1.0                       # PYPOWER reads a zero tap entry as unity
        self.edge_r = _br[:, 2].real.astype(np.float64)      # series resistance, p.u.
        self.edge_x = _br[:, 3].real.astype(np.float64)      # series reactance, p.u.
        self.edge_b = _br[:, 4].real.astype(np.float64)      # charging susceptance, p.u.
        self.edge_g = _br[:, 23].real.astype(np.float64)     # charging conductance, p.u. (iron losses)
        # Series admittance g_s + j b_s = 1/(r + jx): the admittance form of the branch (Ybus off-diagonal
        # magnitude), so the edge set carries admittance directly, not just impedance. edge_b/edge_g are the
        # branch shunt (line charging); together they give the full pi-model admittance per edge.
        _z = self.edge_r + 1j * self.edge_x
        _ys = np.zeros_like(_z, dtype=complex); _nz = np.abs(_z) > 1e-12
        _ys[_nz] = 1.0 / _z[_nz]                              # zero-impedance branches -> 0 (no series path)
        self.edge_gs = np.real(_ys).astype(np.float64)       # series conductance, p.u.
        self.edge_bs = np.imag(_ys).astype(np.float64)       # series susceptance, p.u. (negative for inductive)
        self.edge_tap = _tap                                 # transformer turns ratio, 1.0 for lines
        self.edge_shift = _br[:, 9].real.astype(np.float64)  # phase shift, degrees
        self.edge_status = _br[:, 10].real.astype(np.float64)  # 1 in service, 0 out
        self.edge_is_trafo = np.r_[np.zeros(self.nl), np.ones(self.E - self.nl)].astype(np.float64)
        # Shunts sit on the Ybus DIAGONAL (a BUS property, not expressible edge-only). Stored in MW/MVAr at
        # 1.0 p.u. voltage, per ppc convention.
        self.bus_shunt_g = ppc["bus"][:, 4].real.astype(np.float64)
        self.bus_shunt_b = ppc["bus"][:, 5].real.astype(np.float64)
        # Y = nodal admittance; Yf/Yt = from/to branch-admittance matrices (from-end flow Sf = V_from*conj(Yf@V)).
        self._Ybus, self._Yf, self._Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
        # _lut maps pandapower bus index -> ppc row index (orderings differ — classic footgun).
        self._bMVA = ppc["baseMVA"]; self._lut = base._pd2ppc_lookups["bus"]
        # From-bus (ppc index) per branch, and ppc bus count (Vc is built in ppc ordering).
        self._fb = ppc["branch"][:, 0].real.astype(int); self._nppc = ppc["bus"].shape[0]
        # Total generator MW per bus (summing co-located gens).
        genP = {}
        for r in base.gen.itertuples(): genP[int(r.bus)] = genP.get(int(r.bus), 0.0) + r.p_mw
        # Gen MW aligned to load-bus ordering, so attacks can reason about net (load - gen) per bus.
        self.load_genP = np.array([genP.get(int(b), 0.0) for b in self.load_bus])
        # PTDF (branches x buses): DC sensitivity of each branch's MW flow to a bus injection; steers LRA.
        self._ptdf = makePTDF(self._bMVA, ppc["bus"], ppc["branch"])
        # Slice PTDF to (branches x load-buses): reindex ppc->pandapower via _lut, keep load-bus columns.
        self._ptdf_lb = self._ptdf[:, [self._lut[b] for b in range(C)]][:, self.load_bus]
        # Reusable net for re-solving under attacked loads. Apply the contingency here too, else attacked
        # records solve on the INTACT network while benign came from the post-contingency one.
        self._solvenet = self.NET()
        if self.outage is not None:
            self._solvenet.line.at[self.outage, "in_service"] = False
        # Per-meter SYSTEMATIC BIAS drawn ONCE (constant across scans; relative for P/Q & flows, absolute for
        # V/angle). The slow part of the accuracy-class error; per-scan jitter (self.SDj) is added fresh at emit.
        sb = self._sd_bias
        self.bias_pi = self.rng.normal(0, sb["pi"], self.C); self.bias_qi = self.rng.normal(0, sb["qi"], self.C)
        self.bias_v = self.rng.normal(0, sb["v"], self.C);   self.bias_va = self.rng.normal(0, sb["va"], self.C)
        self.bias_pf = self.rng.normal(0, sb["pf"], self.E); self.bias_qf = self.rng.normal(0, sb["qf"], self.E)
        # Buffer of recent benign records — replay attacks (Ar) copy an earlier clean snapshot from here.
        self.benign_buf = []

    # ---- attack targeting ----
    def centrality_probs(self, strength: float = 1.5) -> np.ndarray:
        """Sampling probability over attackable positions, biased toward structurally CRITICAL buses.

        Combines z-scored degree, closeness and betweenness centrality into one composite criticality score
        (Doostinia et al. IEEE T-IA 2025). `strength` is the exponential tilt: strength=0 recovers the uniform
        draw exactly, larger values concentrate attacks on central load buses. Aligned to self.attackable_pos,
        cached per strength.
        """
        key = round(float(strength), 4)
        cache = getattr(self, "_cent_cache", None)
        if cache is None:
            cache = self._cent_cache = {}
        if key in cache:
            return cache[key]
        import networkx as nx
        G = nx.Graph(); G.add_nodes_from(range(self.C))
        G.add_edges_from(zip(self.ei[0].tolist(), self.ei[1].tolist()))
        dc = nx.degree_centrality(G); cc = nx.closeness_centrality(G); bc = nx.betweenness_centrality(G, normalized=True)

        def _z(dct: Dict) -> np.ndarray:
            v = np.array([dct[b] for b in range(self.C)], float)
            sd = v.std()
            return (v - v.mean()) / (sd if sd > 1e-12 else 1.0)

        comp = _z(dc) + _z(cc) + _z(bc)                     # composite criticality per bus (higher = more central)
        score = comp[self.load_bus[self.attackable_pos]]    # criticality of each attackable load bus
        # Rank-normalize to [0,1] before the exp tilt so the bias is BOUNDED: a raw exp of the z-score explodes
        # on hubs (one bus takes ~all mass). rank in [0,1] gives a most-vs-least ratio of exactly e^strength.
        r = score.argsort().argsort().astype(float)
        rank = r / max(1, len(r) - 1)
        p = np.exp(strength * rank); p = p / p.sum()
        cache[key] = p
        return p

    # ---- emission ----
    # Draw one zero-mean Gaussian noise sample with std `s` (the meter-noise primitive).
    def _n(self, s: float) -> float: return self.rng.normal(0, s)

    def emit_from_state(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Emit a measurement graph DIRECTLY from a stored state X (no re-solve): exact 0-error flows before
        # meter noise. X columns = [Pinj, Qinj, |V|, angle].
        C, SD, M = self.C, self.SD, self.M
        Pi, Qi, V, TH = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        # Rebuild the complex bus-voltage phasor vector in ppc ordering: V * e^{j*theta}.
        Vc = np.zeros(self._nppc, complex)
        for b in range(C): Vc[self._lut[b]] = V[b]*np.exp(1j*np.deg2rad(TH[b]))
        # Exact from-end flow via Sf = V_from*conj(Yf@V), scaled to physical units (Sf.real=MW, Sf.imag=MVAr).
        Sf = Vc[self._fb]*np.conj(self._Yf@Vc)*self._bMVA
        # Node buffers: cols [|V|, P_inj, Q_inj, angle]; mask=1 where metered.
        nx = np.zeros((C, 4), np.float32); nm = np.zeros((C, 4), np.uint8)
        # Each reading = true + constant per-meter bias + per-scan jitter. V-mag/flow biases relative, V/angle
        # biases absolute. va bias/jitter are radians -> degrees to match TH.
        SDj = self.SDj
        for b in range(C):
            if b in M["vbus"] or b in M["pmu"]:     # |V| and angle observed at the same buses
                nx[b, 0] = V[b] + self.bias_v[b] + self._n(SDj["v"]); nm[b, 0] = 1
                nx[b, 3] = TH[b] + np.degrees(self.bias_va[b]) + self._n(np.degrees(SDj["va"])); nm[b, 3] = 1
            # Injection/zero-injection buses emit P/Q: relative bias + jitter (+small floor so ~0 injection
            # still gets a nonzero std).
            if b in M["inj"] or b in self.zero_inj:
                nx[b, 1] = Pi[b]*(1.0+self.bias_pi[b]) + self._n(abs(Pi[b])*SDj["pi"]+1e-3)
                nx[b, 2] = Qi[b]*(1.0+self.bias_qi[b]) + self._n(abs(Qi[b])*SDj["qi"]+1e-3); nm[b, 1:3] = 1
        # Edge buffers: cols [P_from, Q_from]; mask=1 where a flow meter exists.
        ex = np.zeros((self.E, 2), np.float32); em = np.zeros((self.E, 2), np.uint8)
        for e in range(self.E):
            if self.flow_meter[e]:                  # metered branch flow: relative bias + jitter on P and Q
                ex[e, 0] = Sf.real[e]*(1.0+self.bias_pf[e]) + self._n(abs(Sf.real[e])*SDj["pf"]+1e-3)
                ex[e, 1] = Sf.imag[e]*(1.0+self.bias_qf[e]) + self._n(abs(Sf.imag[e])*SDj["qf"]+1e-3); em[e] = 1
        return nx, nm, ex, em

    def state_from_net(self, net: Any) -> np.ndarray:
        # Pull operating state [N,4]=[Pinj, Qinj, |V|, theta] from a SOLVED net, matching the stored pool.
        Pi = net.res_bus.p_mw.values.copy(); Qi = net.res_bus.q_mvar.values.copy()
        # Shunts live in res_bus; subtract shunt draw so Pi/Qi reflect gen/load injection only (matching the
        # stored states and emit_from_state).
        for i in net.shunt.index:
            b = net.shunt.at[i, "bus"]; Pi[b] -= net.res_shunt.p_mw[i]; Qi[b] -= net.res_shunt.q_mvar[i]
        V = net.res_bus.vm_pu.values; TH = net.res_bus.va_degree.values
        return np.column_stack([Pi, Qi, V, TH])              # [N,4] = [Pinj, Qinj, |V|, theta]

    def emit(self, net: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Emit from a SOLVED net (re-solving attacks) by routing its state through emit_from_state, so
        # attacked and benign samples use the IDENTICAL measurement path. Emitting flows from res_line here
        # (while benign uses the Ybus identity) left a ~7 MW systematic benign-vs-attack offset; sharing one
        # path removes it, so an alpha=1 no-op re-solve matches benign.
        return self.emit_from_state(self.state_from_net(net))

    def resolve_states(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Re-solve a pool of operating points [T,N,4] under THIS generator's topology.

        A stored state carries the injections AND the voltages the INTACT network produced. Under a
        contingency the same loads give a different state, so emitting an intact state through a
        post-contingency Ybus would fabricate measurements satisfying no power flow. Re-solving holds the
        loads and reconstructed dispatch at the base-case values and changes only the topology (else a
        shifted load profile would confound topology with load level).

        Returns (Xnew [T,N,4], ok [T] bool). Non-converged or non-finite rows are left as-is and flagged
        False (not dropped), so the caller can intersect converged sets across scenarios on one timestamp axis.
        """
        X = np.asarray(X, dtype=np.float64)
        out = X.copy(); ok = np.zeros(len(X), bool)
        for t in range(len(X)):
            Xt = X[t]
            # Base active load per element = stored injection + generation folded onto that bus.
            Lp = Xt[self.load_bus, 0] + self.load_genP
            Lq = Xt[self.load_bus, 1].copy()
            # Lp_true==Lp: alpha=1 no-op re-solve. Reproduces the stored state on the intact topology (the
            # pinning check); yields the post-contingency state on a contingency topology.
            net = self.solve(Lp, Lq, Xt=Xt, Lp_true=Lp)
            if net is None:
                continue
            s = self.state_from_net(net)
            if not np.isfinite(s).all():
                continue
            out[t] = s; ok[t] = True
        return out, ok

    def solve(self, Lp: np.ndarray, Lq: np.ndarray, Xt: Optional[np.ndarray] = None,
              Lp_true: Optional[np.ndarray] = None) -> Optional[Any]:
        # Set new load P/Q on the reusable net and re-run AC power flow. Returns the solved net, or None on
        # non-convergence (attacks can push loads into non-convergent regions — caller skips those).
        net = self._solvenet
        net.load["p_mw"] = Lp; net.load["q_mvar"] = Lq
        # Pin generation to the TRUE dispatch. Otherwise the re-solve leaves gens at base setpoints and dumps
        # the load change onto the slack, so even a zero-attack re-solve drifts far from the true state (a
        # residual that is NOT the attack). Reconstruct each bus's true gen from the stored injection, hold it
        # at the UNATTACKED dispatch, and let the slack (plus AGC spread below) absorb the delta.
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
            # Spread the attack's net load change across gens (AGC-like, proportional to dispatch) instead of
            # onto the slack alone: keeps a plausible, generation-balanced (stealthy) counterfactual whose
            # footprint is the attacked loads plus a small spread, not a single-bus slack spike.
            dL = float(np.sum(Lp) - np.sum(base_load))          # net extra load introduced by the attack
            tot = gp.sum()
            if tot > 0 and dL != 0.0: gp = gp + dL * (gp / tot)
            net.gen["p_mw"] = gp
            net.gen["vm_pu"] = [Xt[int(b), 2] for b in gbus]     # hold each gen at its true voltage setpoint
            sb = net.ext_grid["bus"].values                      # pin slack reference to true voltage/angle
            net.ext_grid["vm_pu"] = [Xt[int(b), 2] for b in sb]
            net.ext_grid["va_degree"] = [Xt[int(b), 3] for b in sb]
        try: self.pp.runpp(net); return net
        except Exception: return None

    def corrupt(self, nx: np.ndarray, ex: np.ndarray, atk: np.ndarray, kind: str,
                replay: Optional[np.ndarray], floor: float = 0.02,
                cap: float = 0.20) -> Tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
        # Measurement-level attacks (BDD-DETECTABLE contrast families): perturb already-emitted measurements
        # at attacked buses `atk` and incident branches WITHOUT respecting power-flow physics — which is why
        # bad-data detection catches them. Plausibility band [floor, cap] keeps each tamper above the noise
        # floor (not a within-noise no-op) and below the literature cap (realistic FDIA). `weak` flags a
        # record whose realized change fell inside the floor (only Ar can, since it replays the grid) so
        # make() can reject and redraw it.
        inc = [e for e in range(self.E) if self.ei[0, e] in atk or self.ei[1, e] in atk]
        weak = False; mags = []   # mags = realized per-bus |delta|/|base| on the P/Q injection channels

        def band_shift(cur: np.ndarray) -> np.ndarray:
            # additive perturbation with per-channel |delta|/|cur| drawn UNIFORMLY over [floor, cap], random
            # sign. In-band draw (vs clipping a big Gaussian) keeps Ad spread across the band, not piled at cap.
            base = np.abs(cur) + 1e-6
            rel = self.rng.uniform(floor, cap, cur.shape)
            sign = np.where(self.rng.random(cur.shape) < 0.5, -1.0, 1.0)
            return sign * rel * base

        for b in atk:
            base = np.abs(nx[b, 1:3]) + 1e-6
            if kind == "Ad":
                sh = band_shift(nx[b, 1:3]); nx[b, 1:3] += sh; nx[b, 0] += self.rng.normal(0, 0.02)
                mags.append(float(np.max(np.abs(sh)/base)))
            elif kind == "As":
                gain = self.rng.uniform(1.0 + floor, 1.0 + cap)          # gain inside the plausibility band
                nx[b, 1:3] *= gain; mags.append(abs(gain - 1.0))
            elif kind == "Ar" and replay is not None:
                cur = nx[b, 1:3].copy(); nx[b, :] = replay[b, :]
                m = float(np.max(np.abs(nx[b, 1:3] - cur) / base)); mags.append(m)
                if m < floor or m > cap: weak = True                    # replay outside the plausibility band -> reject
        for e in inc:
            if kind == "Ad": ex[e] += band_shift(ex[e])
            elif kind == "As": ex[e] *= self.rng.uniform(1.0 + floor, 1.0 + cap)
        return nx, ex, weak, np.array(mags, float)

    # ---- LRA (Yuan et al. 2011) target line + delta ----
    def _lra_for_line(self, L: int, Lp: np.ndarray, rel: float, K: int, rand: bool = False,
                      floor: float = 0.02) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        # Load Redistribution Attack for target line L: a load-injection delta that is LOAD-CONSERVING (total
        # unchanged -> looks like normal re-dispatch), PER-BUS BOUNDED (|delta_b| <= rel*|Lp_b|), and steers
        # line-L flow via PTDF. rand=True picks buses from the top-2K high-PTDF candidates (varies per record,
        # not memorizable); rand=False is deterministic ranking.
        pl = self._ptdf_lb[L]; cap = rel*np.abs(Lp); score = np.abs(pl)*cap  # pl = line-L PTDF row over load buses
        def pick(side: np.ndarray) -> np.ndarray:
            # Rank one PTDF-sign side by score, keep strongest K (or random K of top-2K if randomized).
            side = side[np.argsort(-score[side])]
            if len(side) == 0: return side
            top = side[:2*K]; k = min(K, len(top))
            return self.rng.choice(top, k, replace=False) if rand else top[:k]
        # Raise load on the positive PTDF side, drop on the negative side, to push flow up on line L.
        # Restrict to ATTACKABLE (active-load) buses so a reactive-only bus is never redistributed onto / labelled.
        pos = pick(np.where((pl > 0) & self._attackable_mask)[0]); neg = pick(np.where((pl < 0) & self._attackable_mask)[0])
        if len(pos) == 0 or len(neg) == 0: return None
        # Both sides scale to a common `budget` (MW moved) so net load change = 0. Always moving the max budget
        # pins the smaller side at cap and piles Al at 20%; instead draw the budget at random within the range
        # keeping both sides' deviation in [floor, rel] (spreads Al across the band, still load-conserving).
        ps, ns = cap[pos].sum(), cap[neg].sum()
        if min(ps, ns) <= 0: return None
        lo = (floor/rel) * max(ps, ns)        # smallest budget keeping the larger side above the floor
        hi = min(ps, ns)                      # largest budget within the per-bus caps
        if lo >= hi: return None              # line too lopsided for a plausible in-band budget -> reject
        budget = float(self.rng.uniform(lo, hi))
        up = cap[pos] * (budget/ps); dn = cap[neg] * (budget/ns)
        d = np.zeros_like(Lp); d[pos] = up; d[neg] = -dn
        # Return (delta, attacked-bus indices, achieved line-L flow change = -sum(PTDF*delta)).
        return d, np.r_[pos, neg], float(-np.sum(pl*d))

    def _pick_lra_target(self, rel: float, K: int, n_targets: int = 15) -> None:
        # Rank lines by achievable conserving-redistribution flow change; keep top-`n_targets` as a target
        # POOL. Varying the target per attack diversifies the attacked-bus set so LRA is not trivially
        # memorizable. Evaluate on base-case loads once, up front.
        bl = self.base.load.p_mw.values
        # Skip the outaged line explicitly: its PTDF row is zero (ranks last anyway) but its base-case flow is
        # NaN, and a NaN reaching self._sgn would poison every LRA delta on that line.
        pot = [(L, self._lra_for_line(L, bl, rel, K)) for L in range(self.nl) if L != self.outage_pos]
        pot = [(L, r) for L, r in pot if r is not None]
        pot.sort(key=lambda x: -abs(x[1][2]))                 # most attackable lines first
        self._Lcands = [L for L, _ in pot[:min(n_targets, len(pot))]]
        # Sign of each candidate's base flow (fallback +1) so the attack WORSENS existing loading (masks a real
        # overload rather than relieving it).
        self._sgn = {L: (float(np.sign(self.base.res_line.p_from_mw.values[L])) or 1.0) for L in self._Lcands}
        self._Ltgt = self._Lcands[0]                          # default/primary target = most attackable line

    def lra_delta(self, Lp: np.ndarray, rel: float, K: int, floor: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        L = int(self.rng.choice(self._Lcands))                # random target line per attack
        r = self._lra_for_line(L, Lp, rel, K, rand=True, floor=floor)   # + randomized bus subset -> not memorizable
        # Apply the base-flow sign so redistribution masks (not relieves) the overload; no feasible delta ->
        # zero delta and empty attacked-bus set (record stays effectively benign).
        return (r[0]*self._sgn[L], r[1]) if r is not None else (np.zeros_like(Lp), np.array([], int))
