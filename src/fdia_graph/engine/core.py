"""Dataset generation engine (attack simulation + realistic measurement emission).

Requires pandapower ('fdia-graph[generate]'). Benign records emit EXACTLY from a stored state (0-error AC
flows, no re-solve); only attacks re-solve, locally. Attack families [WU26]:
  At   the slow ramp: a load set scaled along a rise, a hold and a return, on the fewest-tamper support
  Am   the overload attack: a target branch's reported flow driven to its rating over the window

FdiaGenerator is split by concern across three mixins: state setup lives here (__init__), while
  measurement.py (MeasurementMixin)   emit meter readings from a state
  physics.py     (PhysicsMixin)      a scan's load and generation
  attacks/       (AttackMixin)       every attack: the stealthy false states, the fewest-tamper
                                     search, the ramp and overload episode designs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from ..formulas.network import BranchModel, series_admittance
from ..formulas.noise import PMU_CURRENT_CLASS, bias_jitter_split
from ..models.config import GeneratorOptions
from ..registry import system_id
from .attacks import AttackMixin
from .base import (  # noqa: F401  ACCURACY_CLASS, POWER_NOISE_FLOOR_MW re-exported
    ACCURACY_CLASS,
    INTACT,
    POWER_NOISE_FLOOR_MW,
    MeterBias,
    MeterPlan,
    Outage,
)
from .measurement import MeasurementMixin
from .physics import PhysicsMixin

if TYPE_CHECKING:
    from .pp_types import PandapowerNet, PpcTables

# Integer family label written into the per-bus label tensor `y` (0=clean, >0=attacked of that family);
# the generator makes At and Am (the codes of every released family are in models.choices.FAMILIES).
FAM_ID = {"benign": 0, "At": 5, "Am": 7}
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


class FdiaGenerator(MeasurementMixin, PhysicsMixin, AttackMixin):
    def __init__(
        self,
        system: Union[int, str],
        seed: int = 123,
        vbus_frac: float = 0.6,
        pmu_frac: float = 0.2,
        flow_frac: float = 0.90,
        max_load_mw: Optional[float] = 2000.0,
        meter_model: str = "hybrid",
    ) -> None:
        """Load the IEEE case, solve the base power flow, and draw the meter plan and the per-meter
        biases from `seed`.

        `max_load_mw` keeps a load above it out of the attack targets: such a load is an area
        equivalent (IEEE-145 lumps whole regions into single loads of 4 to 58 GW), not a substation
        an attacker could shift by 5% to 20% with any power-flow solution nearby; None disables the
        cap. No load of the other seven ladder systems exceeds 1.1 GW.

        The random draws happen in a fixed order, the meter plan (voltage buses, PMU buses, flow
        meters), then the six per-meter bias vectors, then the branch-current channels' biases.

        `meter_model` is what the meters measure [D10]: "hybrid", a SCADA voltmeter reads
        |V| only, the angle is a PMU channel, and every PMU reads the current phasor of each
        in-service branch at its bus [WU26 eqs. 17-20].
        """
        # pandapower is heavy/optional: import lazily so it's only needed when actually generating.
        import pandapower as pp
        import pandapower.networks as pn

        self.pp = pp
        self.C = system_id(system)  # "ieee118" and 118 both accepted, like every public entry point
        opts = GeneratorOptions(max_load_mw, meter_model)
        self.max_load_mw, hybrid = opts.max_load_mw, opts.meter_model == "hybrid"
        self.rng = np.random.default_rng(seed)
        self.seed, self.scan_key, self._jitter = seed, None, self.rng
        # Measurement noise stds, the accuracy classes (ACCURACY_CLASS), split into a per-scan jitter
        # and a per-meter bias (see formulas.noise).
        self.SD = dict(ACCURACY_CLASS)
        self.SDj, self._bias_sd = bias_jitter_split(self.SD, jitter_frac=0.25)
        self.NET = getattr(pn, _CASE[self.C])
        self.base = self._open_case()
        self._load_tables(self.base)
        self._meter_plan(self.base, vbus_frac, pmu_frac, flow_frac, hybrid)
        self._edge_index(self.base)
        self._branch_physics(self.base._ppc)
        self._admittances(self.base._ppc)
        self._meter_bias()
        if hybrid:
            self._current_bias()

    def _open_case(self) -> PandapowerNet:
        """The pandapower case with its base power flow solved; the generator is the intact grid
        (self.contingency = INTACT, which the file attributes record)."""
        base = self.NET()
        self.contingency = INTACT
        self.pp.runpp(base)
        return base

    def _load_tables(self, base: PandapowerNet) -> None:
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
        self._case_limits(base)

    def _case_limits(self, base: PandapowerNet) -> None:
        """Per-bus base load and generation [N, 2] (P, Q) and the case's limits [WU26 eqs. 21-23]:
        bus voltage limits [N, 2] and generator P and Q limits [N, 2] summed over co-located
        generators, unbounded (±inf) where a bus has none; the slack (ext_grid) is unbounded, its
        output is the balance of every state and its limits are the pool's own."""
        C = self.C
        self.load_base = np.zeros((C, 2))
        self.gen_base = np.zeros((C, 2))
        self.p_lim = np.tile([-np.inf, np.inf], (C, 1))
        self.q_lim = np.tile([-np.inf, np.inf], (C, 1))
        # per-bus sums over co-located loads and generators, in row order (np.add.at is unbuffered)
        np.add.at(self.load_base, base.load.bus.to_numpy(int), base.load[["p_mw", "q_mvar"]].to_numpy(float))
        gens = base.gen
        at = gens.bus.to_numpy(int)
        self.has_gen = np.zeros(C, bool)  # a generator on the bus, zero-MW condensers included
        self.has_gen[at] = True
        np.add.at(self.gen_base[:, 0], at, gens.p_mw.to_numpy(float))
        if len(at):
            self.p_lim[at] = self.q_lim[at] = 0.0  # summed over the bus's generators below
            np.add.at(self.p_lim, at, gens[["min_p_mw", "max_p_mw"]].to_numpy(float))
            np.add.at(self.q_lim, at, gens[["min_q_mvar", "max_q_mvar"]].to_numpy(float))
        self.v_case = np.stack([base.bus.min_vm_pu.values, base.bus.max_vm_pu.values], axis=1).astype(float)

    def _meter_plan(
        self, base: PandapowerNet, vbus_frac: float, pmu_frac: float, flow_frac: float, hybrid: bool = False
    ) -> None:
        """The sparse metering plan, sampled once (self.meters, a MeterPlan): vbus = voltage-magnitude
        meters, pmu = |V| + angle meters, inj = metered P/Q injection buses (all injection buses), and
        a per-branch flow-meter mask with fraction flow_frac. Three draws from the seeded RNG, in this
        order; `hybrid` sets the D10 meter model (angles at PMUs only, PMU branch currents) without a
        draw of its own."""
        C = self.C
        vbus = set(self.rng.choice(C, int(vbus_frac * C), replace=False).tolist())
        pmu = set(self.rng.choice(C, max(1, int(pmu_frac * C)), replace=False).tolist())
        flow = self.rng.random(len(base.line) + len(base.trafo)) < flow_frac
        self.meters = MeterPlan(vbus, pmu, self._injection_buses, flow, hybrid, hybrid)

    def _edge_index(self, base: PandapowerNet) -> None:
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

    def _branch_physics(self, ppc: PpcTables) -> None:
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

    def _admittances(self, ppc: PpcTables) -> None:
        """Ybus and the from/to branch-admittance matrices (from-end flow Sf = V_from * conj(Yf @ V))
        and the ppc bus lookup.

        _ppc_row maps pandapower bus index -> ppc row index (the orderings differ, a classic footgun);
        _from_bus_ppc is the from-bus (ppc index) per branch; Vc is built in ppc ordering.
        """
        from pandapower.pypower.makeYbus import makeYbus

        self._Ybus, self._Yf, self._Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
        self._base_mva = ppc["baseMVA"]
        self._ppc_row = self.base._pd2ppc_lookups["bus"]
        self._from_bus_ppc = ppc["branch"][:, 0].real.astype(int)
        self._ppc_branch = ppc["branch"]  # the ppc branch matrix, for the flow derivatives
        self._n_ppc_buses = ppc["bus"].shape[0]

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

    def _current_bias(self) -> None:
        """The PMU branch-current channels' systematic bias (hybrid meters only): relative, one per
        channel, drawn once after the six v0.8.3 bias vectors from the accuracy class
        `PMU_CURRENT_CLASS` split like the others (`bias_jitter_split`); the per-scan jitter part is
        kept in `self._i_jitter`."""
        jit, bias = bias_jitter_split({"i": PMU_CURRENT_CLASS}, jitter_frac=0.25)
        self._i_jitter = jit["i"]
        self.bias = self.bias._replace(i=self.rng.normal(0, bias["i"], (self.E, 4)))
