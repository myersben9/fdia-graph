"""Dataset generation engine (attack simulation + realistic measurement emission).

Requires pandapower ('fdia-graph[generate]'). Benign records emit EXACTLY from a stored state (0-error AC
flows, no re-solve); only attacks re-solve. Attack families:
  Aq   stealthy load scaling: bounded per-bus rescale + AGC-balanced AC re-solve (cf. Boyaci 2022 "Ao")
  At   (ramp) temporal creeping load surge up then down (Haghshenas/Hasnat/Naeini ISGT 2023)
  Al   (LRA) targeted masked-overload (Yuan/Li/Ren IEEE T-SG 2011)
  Ad/As/Ar  measurement-level corruption (BDD-detectable contrast set)

FdiaGenerator is split by concern across three mixins: state setup lives here (__init__), while
  measurement.py (MeasurementMixin)   emit meter readings from a state / net
  physics.py     (PhysicsMixin)      AC re-solve under new loads
  attacks.py     (AttackMixin)       corrupt() + load-redistribution deltas
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np

from ..formulas.network import BranchModel, series_admittance
from ..formulas.noise import bias_jitter_split
from ..models.assets import LineCandidate  # noqa: F401  re-exported: defined here before the models package
from ..registry import system_id
from .attacks import AttackMixin
from .base import INTACT, MeterBias, MeterPlan, Outage
from .measurement import MeasurementMixin
from .physics import PhysicsMixin

# Integer family label written into the per-bus label tensor `y` (0=clean, >0=attacked of that family).
# "Ao"/"SLS" are back-compat aliases for Aq (id 1); "ramp"=At, "LRA"=Al.
FAM_ID = {
    "benign": 0,
    "Aq": 1,
    "SLS": 1,
    "Ao": 1,
    "Ad": 2,
    "As": 3,
    "Ar": 4,
    "At": 5,
    "ramp": 5,
    "Al": 6,
    "LRA": 6,
    "Am": 7,
}
# Bus-count knob (14/118/300) -> pandapower.networks factory name.
# Bus-count -> pandapower.networks builder. Transmission systems only (>=110 kV, meshed). A system is
# load()-able only once its pool + registry entry ship; listing it here just lets the generator build it.
_CASE = {
    14: "case14",
    30: "case30",
    57: "case57",
    89: "case89pegase",
    118: "case118",
    145: "case145",
    200: "case_illinois200",
    300: "case300",
}


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
        raise ValueError(
            f"line index {idx} is not in this case (lines are {net.line.index.min()}..{net.line.index.max()})"
        )
    return idx


def _n_islands(net: Any) -> int:
    """Number of connected components over the IN-SERVICE network (1 == still one connected grid).

    Cheap islanding screen to run BEFORE generating: pandapower "converges" on an islanded case by
    silently marking stranded buses isolated (bus type 4), so solver failure is NOT the infeasibility signal.
    """
    import networkx as nx
    from pandapower import topology as top

    return int(nx.number_connected_components(top.create_nxgraph(net)))


def line_outage_candidates(
    system: Union[int, str], top_n: int = 5, seed_flow_from: Any = None
) -> tuple[list[LineCandidate], list[LineCandidate]]:
    """Rank single-line N-1 contingencies by base-case active power flow, keeping the network connected.

    Returns (accepted, rejected). `accepted` = the `top_n` highest-flow lines whose removal leaves one
    connected, solvable, PTDF-well-posed network (dict: line index, terminal buses, name, signed from-end
    MW flow). `rejected` = higher-flow lines screened out, with reason. Highest-flow ranking is the standard
    N-1 screening choice (Moshtagh et al.). A screening aid only: N-1 timeline generation is disabled
    (`fg.generate` and `generate_timeline` take no outage; `FdiaGenerator(outage=)` stays for engine use).
    """
    import pandapower as pp
    import pandapower.networks as pn

    NET = getattr(pn, _CASE[system_id(system)])
    base = seed_flow_from if seed_flow_from is not None else NET()
    if "p_from_mw" not in base.res_line or base.res_line.empty:
        pp.runpp(base)
    flow = base.res_line["p_from_mw"].to_numpy()
    lut0 = base._pd2ppc_lookups["bus"]
    order = np.argsort(-np.abs(np.nan_to_num(flow)))  # highest |MW| first
    accepted, rejected = [], []
    for pos in order:
        if len(accepted) >= top_n:
            break
        idx = int(base.line.index[pos])
        if not bool(base.line.at[idx, "in_service"]):
            continue  # already open: not a contingency
        why = _screen_line(NET, idx, lut0)
        rec = _line_record(base, idx, int(pos), float(flow[pos]))
        if why:
            rejected.append(LineCandidate(**rec, reason=why))
        else:
            accepted.append(rec)
    return accepted, rejected


def _screen_line(NET: Any, idx: int, lut0: np.ndarray) -> Optional[str]:
    """Why opening line `idx` is not an acceptable contingency, or None when it is: the grid must stay
    one island, the post-contingency power flow must converge with no isolated bus, the ppc bus
    ordering must not change (or the shard is not comparable to the base shard), and the DC PTDF
    must be well posed."""
    import pandapower as pp
    from pandapower.pypower.makePTDF import makePTDF

    net = NET()
    net.line.at[idx, "in_service"] = False
    if _n_islands(net) != 1:
        return f"removing it splits the grid into {_n_islands(net)} islands"
    try:
        pp.runpp(net)
    except Exception as e:
        return f"post-contingency AC power flow does not converge ({type(e).__name__})"
    n_iso = int((net._ppc["bus"][:, 1].real == 4).sum())
    if n_iso:
        return f"leaves {n_iso} isolated bus(es)"
    if not np.array_equal(lut0, net._pd2ppc_lookups["bus"]):
        return "changes the ppc bus ordering (not comparable to the base shard)"
    try:
        makePTDF(net._ppc["baseMVA"], net._ppc["bus"], net._ppc["branch"])
    except Exception:
        return "PTDF is singular (the DC network is islanded)"
    return None


def _line_record(base: Any, idx: int, pos: int, base_flow_mw: float) -> LineCandidate:
    """One candidate line as the caller reports it: index, position, terminals, name, base flow."""
    _nm = base.line.at[idx, "name"]
    return LineCandidate(
        line=idx,
        pos=pos,
        from_bus=int(base.line.at[idx, "from_bus"]),
        to_bus=int(base.line.at[idx, "to_bus"]),
        name=(f"line{idx}" if _nm is None or str(_nm) in ("None", "nan", "") else str(_nm)),
        base_flow_mw=base_flow_mw,
    )


class FdiaGenerator(MeasurementMixin, PhysicsMixin, AttackMixin):
    def __init__(
        self,
        system: Union[int, str],
        seed: int = 123,
        vbus_frac: float = 0.6,
        pmu_frac: float = 0.2,
        flow_frac: float = 0.90,
        outage: Optional[Union[int, str]] = None,
        max_load_mw: Optional[float] = 2000.0,
    ) -> None:
        """Load the IEEE case, apply the optional N-1 contingency, solve the base power flow, and
        draw the meter plan and the per-meter biases from `seed`.

        `max_load_mw` keeps a load above it out of the attack targets: such a load is an area
        equivalent (IEEE-145 lumps whole regions into single loads of 4 to 58 GW), not a substation
        an attacker could shift by 5% to 20% with any power-flow solution nearby; None disables the
        cap. No load of the other seven ladder systems exceeds 1.1 GW.

        The random draws happen in a fixed order, the meter plan (voltage buses, PMU buses, flow
        meters) and then the six per-meter bias vectors, so a shard's meter plan and biases are
        identical across topologies at a given seed; the contingency consumes no randomness.
        """
        # pandapower is heavy/optional: import lazily so it's only needed when actually generating.
        import pandapower as pp
        import pandapower.networks as pn

        self.pp = pp
        self.C = system_id(system)  # "ieee118" and 118 both accepted, like every public entry point
        if max_load_mw is not None and not max_load_mw > 0:
            raise ValueError(f"max_load_mw is a load cap in MW (or None), got {max_load_mw!r}")
        self.max_load_mw = max_load_mw
        self.rng = np.random.default_rng(seed)
        # Measurement noise stds (accuracy-class model [ASP14]). |V|/angle are the class-0.2/sqrt(3) IT
        # figures; P/Q use a larger ~1.7% power-measurement std. Relative for flows/injections, absolute
        # for V/angle. Split into a per-scan jitter and a per-meter bias (see formulas.noise).
        self.SD = dict(pf=0.017, qf=0.017, v=0.0012, pi=0.017, qi=0.017, va=0.00168)
        self.SDj, self._bias_sd = bias_jitter_split(self.SD, jitter_frac=0.25)
        self.NET = getattr(pn, _CASE[self.C])
        self.base = self._open_case(outage)
        self._load_tables(self.base)
        self._meter_plan(self.base, vbus_frac, pmu_frac, flow_frac)
        self._edge_index(self.base)
        self._branch_physics(self.base._ppc)
        self._admittances(self.base._ppc)
        # Reusable net for re-solving under attacked loads. Apply the contingency here too, else attacked
        # records solve on the INTACT network while benign came from the post-contingency one.
        self._solve_net = self.NET()
        if self.contingency.line is not None:
            self._solve_net.line.at[self.contingency.line, "in_service"] = False
        self._meter_bias()
        # Buffer of recent benign records: replay attacks (Ar) copy an earlier clean snapshot from here.
        self.benign_buf = []

    @property
    def nl(self) -> int:
        """The line count under its 0.17 name; retires in 0.19, use `n_lines`."""
        warnings.warn("nl is deprecated and retires in 0.19: use n_lines", DeprecationWarning, stacklevel=2)
        return self.n_lines

    def _open_case(self, outage: Optional[Union[int, str]]) -> Any:
        """The pandapower case with the contingency applied and its base power flow solved; sets
        self.contingency (an Outage, INTACT when no line is opened).

        `outage` (line index or name, None = intact) opens the line BEFORE the base power flow, so every
        derived quantity (Ybus, Yf/Yt, PTDF, edge_status, base state, all measurements) is
        post-contingency. The branch ROW is kept in ppc with status 0, so edge_index, E and the meter
        plan match the intact case: exactly one thing changed, and shards are directly comparable.
        An islanding contingency is refused up front (pandapower would "converge" on a non-solution and
        makePTDF would hit a singular matrix mid-build).
        """
        pp = self.pp
        base = self.NET()
        self.contingency = INTACT
        if outage is not None:
            line = _line_id(base, outage)
            # Record the INTACT flow on the line to be opened, so the shard reports the contingency's size.
            intact = self.NET()
            pp.runpp(intact)
            # IEEE cases carry no line names (None), so fall back to a "line<idx>" tag.
            _nm = base.line.at[line, "name"]
            self.contingency = Outage(
                line=line,
                pos=int(base.line.index.get_loc(line)),
                name=(f"line{line}" if _nm is None or str(_nm) in ("None", "nan", "") else str(_nm)),
                from_bus=int(base.line.at[line, "from_bus"]),
                to_bus=int(base.line.at[line, "to_bus"]),
                base_flow_mw=float(intact.res_line.at[line, "p_from_mw"]),
            )
            base.line.at[line, "in_service"] = False
            if _n_islands(base) != 1:
                raise ValueError(
                    f"line {line} outage splits the grid into {_n_islands(base)} islands; "
                    f"screen with line_outage_candidates() before generating"
                )
        # Solve the (possibly post-contingency) AC power flow for the base operating point.
        pp.runpp(base)
        if self.contingency.line is not None:
            n_iso = int((base._ppc["bus"][:, 1].real == 4).sum())
            if n_iso:
                raise ValueError(f"line {self.contingency.line} outage leaves {n_iso} isolated bus(es)")
        return base

    def _load_tables(self, base: Any) -> None:
        """Which buses carry load, which of those can be attacked, which inject nothing, and the
        generator MW co-located with each load.

        load_bus is the bus of EVERY load element, aligned 1:1 with net.load so the re-solve and the
        PTDF-over-load arrays stay the same length. ATTACKABLE = load_bus positions with real ACTIVE
        load (|p_mw| > 0) and at most `max_load_mw`: reactive-only loads (e.g. IEEE-300 buses 141,
        183) and area-equivalent loads stay in the physics table but are excluded from target
        selection and the LRA candidate set, since attacking a reactive-only load leaves no P
        footprint yet would still get a y = 1 label, and a gigawatt step has no local solution.
        Zero-injection buses are pure junctions with net injection exactly 0 (a strong
        constraint); a near-zero injection measurement is still emitted there.
        """
        C = self.C
        _lb = base.load
        self.load_bus = _lb["bus"].values
        self.load_p0 = _lb["p_mw"].to_numpy(dtype=float)  # base active load per element, signed
        self.load_q0 = _lb["q_mvar"].to_numpy(dtype=float)  # base reactive load per element, signed
        self.slack_bus = int(
            base.ext_grid.bus.values[0]
        )  # the angle reference; a local attack never moves it
        p_load = _lb["p_mw"].abs().values
        # ... so a load on the slack bus is never a target either: it can never enter a region, and
        # a frame that scaled it would be labelled attacked with no attack in the false state
        self._attackable_mask = (p_load > 0.0) & (self.load_bus != self.slack_bus)
        if self.max_load_mw is not None:
            self._attackable_mask &= p_load <= self.max_load_mw
        self.attackable_pos = np.where(self._attackable_mask)[0]
        # The stealthy families (Aq, At, Al, Am) also skip every load on a generator bus, zero-MW
        # condensers included: a generator bus is too risky to falsify, the attacker's rule in the
        # protocol this dataset follows [BOY22]. The in-place families keep the full set.
        live_sgen = (base.sgen.p_mw.abs() > 0) | (base.sgen.q_mvar.abs() > 0)  # as for injection buses
        gen_buses = np.r_[base.gen.bus.values, base.sgen.bus.values[live_sgen.values]]
        self._stealthy_mask = self._attackable_mask & ~np.isin(self.load_bus, gen_buses)
        self.stealthy_pos = np.where(self._stealthy_mask)[0]
        # Every bus with some injection element (gen, load, ext_grid, shunt, a static generator that
        # produces anything: IEEE-89 and 300 feed buses from sgen alone; IEEE-200's are all 0 MW).
        sgen = base.sgen[(base.sgen.p_mw.abs() > 0) | (base.sgen.q_mvar.abs() > 0)]
        inj = np.unique(
            np.r_[
                base.gen.bus.values,
                base.load.bus.values,
                base.ext_grid.bus.values,
                base.shunt.bus.values,
                sgen.bus.values,
            ]
        )
        self.zero_inj = [b for b in range(C) if b not in set(inj)]
        self._injection_buses = sorted(set(inj.tolist()))
        # Base-case generator MW per bus (summing co-located gens), aligned to load-bus ordering. The
        # pools scale generation with load, so a scan's load and dispatch come from `true_load` and
        # `scan_generation`, not from this constant.
        genP: dict[int, float] = {}
        for r in base.gen.itertuples():
            genP[int(r.bus)] = genP.get(int(r.bus), 0.0) + r.p_mw
        self.load_genP = np.array([genP.get(int(b), 0.0) for b in self.load_bus])
        self._case_limits(base)

    def _case_limits(self, base: Any) -> None:
        """Per-bus base load and generation [N, 2] (P, Q) and the case's limits [WU26, eqs. 21-23]:
        bus voltage limits [N, 2] and generator P and Q limits [N, 2] summed over co-located
        generators, unbounded (±inf) where a bus has none; the slack (ext_grid) is unbounded, its
        output is the balance of every state and its limits are the pool's own."""
        C = self.C
        self.load_base = np.zeros((C, 2))
        self.gen_base = np.zeros((C, 2))
        self.p_lim = np.tile([-np.inf, np.inf], (C, 1))
        self.q_lim = np.tile([-np.inf, np.inf], (C, 1))
        for r in base.load.itertuples():
            self.load_base[int(r.bus)] += (r.p_mw, r.q_mvar)
        gens = base.gen
        self.has_gen = np.zeros(C, bool)  # a generator on the bus, zero-MW condensers included
        self.has_gen[np.unique(gens.bus.values).astype(int)] = True
        for b in np.unique(gens.bus.values):
            rows = gens[gens.bus == b]
            self.gen_base[int(b), 0] = rows.p_mw.sum()
            self.p_lim[int(b)] = (rows.min_p_mw.sum(), rows.max_p_mw.sum())
            self.q_lim[int(b)] = (rows.min_q_mvar.sum(), rows.max_q_mvar.sum())
        self.v_case = np.stack([base.bus.min_vm_pu.values, base.bus.max_vm_pu.values], axis=1).astype(float)

    def _meter_plan(self, base: Any, vbus_frac: float, pmu_frac: float, flow_frac: float) -> None:
        """The sparse metering plan, sampled once (self.meters, a MeterPlan): vbus = voltage-magnitude
        meters, pmu = |V| + angle meters, inj = metered P/Q injection buses (all injection buses), and
        a per-branch flow-meter mask with fraction flow_frac. Three draws from the seeded RNG, in this
        order."""
        C = self.C
        vbus = set(self.rng.choice(C, int(vbus_frac * C), replace=False).tolist())
        pmu = set(self.rng.choice(C, max(1, int(pmu_frac * C)), replace=False).tolist())
        flow = self.rng.random(len(base.line) + len(base.trafo)) < flow_frac
        self.meters = MeterPlan(vbus, pmu, self._injection_buses, flow)

    def _edge_index(self, base: Any) -> None:
        """Edge index (2 x E): row 0 = from-bus, row 1 = to-bus; lines use from/to, transformers hv/lv,
        concatenated so branches share one contiguous 0..E-1 indexing (lines first)."""
        base_line = base.line
        self.ei = np.vstack(
            [
                np.r_[base_line.from_bus.values, base.trafo.hv_bus.values],
                np.r_[base_line.to_bus.values, base.trafo.lv_bus.values],
            ]
        ).astype(np.int32)
        self.E = self.ei.shape[1]
        self.n_lines = len(base_line)  # lines come first in ei, transformers after
        # DEPRECATED (v0.5.0), retained for loading. Mixes UNITS: line reactance in ohms vs trafo vk_percent,
        # putting trafo entries ~3 orders of magnitude above lines on IEEE-300. Use the per-unit edge_* arrays.
        self.x_react = np.r_[
            base_line.x_ohm_per_km.values * base_line.length_km.values, base.trafo.vk_percent.values
        ].astype(np.float32)

    def _branch_physics(self, ppc: Any) -> None:
        """Full per-unit branch and bus physics, exactly the quantities makeYbus consumes, so a model
        has the same information the state estimator does.

        ppc branch rows are ordered lines-then-transformers, matching self.ei. BR_G (column 23) is a
        pandapower extension (absent from stock PYPOWER) carrying transformer iron losses: omitting it
        reconstructs Ybus exactly on IEEE-14 (no transformers) but WRONGLY on IEEE-118/300, an error
        that looks correct on the first system people test. float64, not float32: float32 rounding
        degrades the Ybus reconstruction from ~1e-14 to ~1e-4 on IEEE-300, and exact reconstruction is
        the whole claim. Shunts sit on the Ybus DIAGONAL (a BUS property), stored in MW/MVAr at 1.0 pu.
        """
        _br = ppc["branch"]
        _tap = _br[:, 8].real.astype(np.float64).copy()
        _tap[_tap == 0] = 1.0  # PYPOWER reads a zero tap entry as unity
        self.branch = BranchModel(
            r=_br[:, 2].real.astype(np.float64),  # series resistance, p.u.
            x=_br[:, 3].real.astype(np.float64),  # series reactance, p.u.
            b=_br[:, 4].real.astype(np.float64),  # charging susceptance, p.u.
            g=_br[:, 23].real.astype(np.float64),  # charging conductance, p.u. (iron losses)
            tap=_tap,  # transformer turns ratio, 1.0 for lines
            shift_deg=_br[:, 9].real.astype(np.float64),  # phase shift, degrees
            status=_br[:, 10].real.astype(np.float64),  # 1 in service, 0 out
        )
        # Series admittance g_s + j b_s = 1 / (r + jx): the admittance form of the branch, so the edge set
        # carries admittance directly, not just impedance (formulas.network.series_admittance).
        _ys = series_admittance(self.branch.r, self.branch.x)
        self.edge_gs = np.real(_ys).astype(np.float64)  # series conductance, p.u.
        self.edge_bs = np.imag(_ys).astype(np.float64)  # series susceptance, p.u. (negative for inductive)
        self.edge_is_trafo = np.r_[np.zeros(self.n_lines), np.ones(self.E - self.n_lines)].astype(np.float64)
        self.bus_shunt_g = ppc["bus"][:, 4].real.astype(np.float64)
        self.bus_shunt_b = ppc["bus"][:, 5].real.astype(np.float64)

    def _admittances(self, ppc: Any) -> None:
        """Ybus and the from/to branch-admittance matrices (from-end flow Sf = V_from * conj(Yf @ V)),
        the ppc bus lookup, and the DC PTDF that steers the load-redistribution attack.

        _ppc_row maps pandapower bus index -> ppc row index (the orderings differ, a classic footgun);
        _from_bus_ppc is the from-bus (ppc index) per branch; Vc is built in ppc ordering.
        """
        from pandapower.pypower.makePTDF import makePTDF
        from pandapower.pypower.makeYbus import makeYbus

        C = self.C
        self._Ybus, self._Yf, self._Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
        self._base_mva = ppc["baseMVA"]
        self._ppc_row = self.base._pd2ppc_lookups["bus"]
        self._from_bus_ppc = ppc["branch"][:, 0].real.astype(int)
        self._n_ppc_buses = ppc["bus"].shape[0]
        # PTDF (branches x buses): DC sensitivity of each branch's MW flow to a bus injection; sliced to
        # (branches x load-buses) by reindexing ppc -> pandapower via _ppc_row and keeping the load-bus columns.
        self._ptdf = makePTDF(self._base_mva, ppc["bus"], ppc["branch"])
        self._ptdf_load_buses = self._ptdf[:, [self._ppc_row[b] for b in range(C)]][:, self.load_bus]

    def _meter_bias(self) -> None:
        """The per-meter SYSTEMATIC bias, drawn ONCE (self.bias, a MeterBias): constant across scans,
        relative for P/Q and flows, absolute for V and angle, the slow part of the accuracy-class error.
        The per-scan jitter (self.SDj) is added fresh at emit. Six draws from the seeded RNG, after the
        meter plan, in this order."""
        sb, C, E = self._bias_sd, self.C, self.E
        self.bias = MeterBias(
            pi=self.rng.normal(0, sb["pi"], C),
            qi=self.rng.normal(0, sb["qi"], C),
            v=self.rng.normal(0, sb["v"], C),
            va=self.rng.normal(0, sb["va"], C),
            pf=self.rng.normal(0, sb["pf"], E),
            qf=self.rng.normal(0, sb["qf"], E),
        )

    def centrality_probs(self, strength: float = 1.5) -> np.ndarray:
        """Sampling probability over attackable positions, biased toward structurally CRITICAL buses.

        Combines z-scored degree, closeness and betweenness centrality into one composite criticality score
        (Doostinia et al. IEEE T-IA 2025). `strength` is the exponential tilt: strength=0 recovers the uniform
        draw exactly, larger values concentrate attacks on central load buses. Aligned to self.attackable_pos,
        cached per strength.
        """
        key = round(float(strength), 4)
        cache = getattr(self, "_centrality_cache", None)
        if cache is None:
            cache = self._centrality_cache = {}
        if key in cache:
            return cache[key]
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(self.C))
        G.add_edges_from(zip(self.ei[0].tolist(), self.ei[1].tolist()))
        dc = nx.degree_centrality(G)
        cc = nx.closeness_centrality(G)
        bc = nx.betweenness_centrality(G, normalized=True)

        def _z(dct: dict) -> np.ndarray:
            v = np.array([dct[b] for b in range(self.C)], float)
            sd = v.std()
            return (v - v.mean()) / (sd if sd > 1e-12 else 1.0)

        comp = _z(dc) + _z(cc) + _z(bc)  # composite criticality per bus (higher = more central)
        score = comp[self.load_bus[self.attackable_pos]]  # criticality of each attackable load bus
        # Rank-normalize to [0,1] before the exp tilt so the bias is BOUNDED: a raw exp of the z-score explodes
        # on hubs (one bus takes ~all mass). rank in [0,1] gives a most-vs-least ratio of exactly e^strength.
        r = score.argsort().argsort().astype(float)
        rank = r / max(1, len(r) - 1)
        p = np.exp(strength * rank)
        p = p / p.sum()
        cache[key] = p
        return p
