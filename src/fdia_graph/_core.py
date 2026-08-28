"""Dataset generation engine (attack simulation + realistic measurement emission).

Requires pandapower ('fdia-graph[generate]'). Benign records emit EXACTLY from a stored state (0-error AC
flows, no re-solve); only attacks re-solve. Attack families:
  Aq   stealthy load scaling: bounded per-bus rescale + AGC-balanced AC re-solve (cf. Boyaci 2022 "Ao")
  At   (ramp) temporal creeping load surge up then down (Haghshenas/Hasnat/Naeini ISGT 2023)
  Al   (LRA) targeted masked-overload (Yuan/Li/Ren IEEE T-SG 2011)
  Ad/As/Ar  measurement-level corruption (BDD-detectable contrast set)

FdiaGenerator is split by concern across three mixins: state setup lives here (__init__), while
  _measurement.py (MeasurementMixin)  emit meter readings from a state / net
  _physics.py     (PhysicsMixin)      AC re-solve under new loads
  _attacks.py     (AttackMixin)       corrupt() + load-redistribution deltas
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ._attacks import AttackMixin
from ._measurement import MeasurementMixin
from ._physics import PhysicsMixin

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
    import pandapower as pp
    import pandapower.networks as pn
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


class FdiaGenerator(MeasurementMixin, PhysicsMixin, AttackMixin):
    def __init__(self, system: Union[int, str], seed: int = 123, vbus_frac: float = 0.6, pmu_frac: float = 0.2,
                 flow_frac: float = 0.90, outage: Optional[Union[int, str]] = None) -> None:
        # pandapower is heavy/optional: import lazily so it's only needed when actually generating.
        import pandapower as pp
        import pandapower.networks as pn
        # makeYbus -> complex nodal admittance Y; makePTDF -> linear line-flow sensitivities (for LRA targeting).
        from pandapower.pypower.makeYbus import makeYbus
        from pandapower.pypower.makePTDF import makePTDF
        self.pp = pp
        self.C = int(system)
        self.rng = np.random.default_rng(seed)
        # Measurement noise stds (accuracy-class model). |V|/angle are the class-0.2/sqrt(3) IT figures;
        # P/Q use a larger ~1.7% power-measurement std. Relative for flows/injections, absolute for V/angle.
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
        self.outage = None
        self.outage_pos = -1
        self.outage_name = ""
        self.outage_from_bus = -1
        self.outage_to_bus = -1
        self.outage_base_flow_mw = float("nan")
        base = self.NET()
        if outage is not None:
            self.outage = _line_id(base, outage)
            # Record the INTACT flow on the line to be opened, so the shard reports the contingency's size.
            intact = self.NET()
            pp.runpp(intact)
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
        self.E = self.ei.shape[1]
        self.nl = len(base.line)  # nl = number of lines (first nl cols of ei)
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
        _ys = np.zeros_like(_z, dtype=complex)
        _nz = np.abs(_z) > 1e-12
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
        self._bMVA = ppc["baseMVA"]
        self._lut = base._pd2ppc_lookups["bus"]
        # From-bus (ppc index) per branch, and ppc bus count (Vc is built in ppc ordering).
        self._fb = ppc["branch"][:, 0].real.astype(int)
        self._nppc = ppc["bus"].shape[0]
        # Total generator MW per bus (summing co-located gens).
        genP = {}
        for r in base.gen.itertuples():
            genP[int(r.bus)] = genP.get(int(r.bus), 0.0) + r.p_mw
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
        self.bias_pi = self.rng.normal(0, sb["pi"], self.C)
        self.bias_qi = self.rng.normal(0, sb["qi"], self.C)
        self.bias_v = self.rng.normal(0, sb["v"], self.C)
        self.bias_va = self.rng.normal(0, sb["va"], self.C)
        self.bias_pf = self.rng.normal(0, sb["pf"], self.E)
        self.bias_qf = self.rng.normal(0, sb["qf"], self.E)
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
        G = nx.Graph()
        G.add_nodes_from(range(self.C))
        G.add_edges_from(zip(self.ei[0].tolist(), self.ei[1].tolist()))
        dc = nx.degree_centrality(G)
        cc = nx.closeness_centrality(G)
        bc = nx.betweenness_centrality(G, normalized=True)

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
        p = np.exp(strength * rank)
        p = p / p.sum()
        cache[key] = p
        return p
