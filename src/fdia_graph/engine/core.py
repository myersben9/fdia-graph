"""The dataset generation engine: attack simulation and realistic meter readings on an IEEE case.

Requires pandapower ('fdia-graph[generate]'). A benign scan is emitted exactly from a stored state (no
re-solve); only attacks re-solve, locally. The attack families are At (the slow ramp) and Am (the
overload attack) [WU26].

FdiaGenerator sets up the state here (`__init__`) and gets its methods from three mixins:
physics.py (a scan's load and generation), measurement.py (meter readings from a state) and
attacks/ (every attack).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from ..constants.grid import IEEE_CASES
from ..constants.meters import ACCURACY_CLASS, JITTER_FRAC, PMU_CURRENT_CLASS
from ..formulas.network import series_admittance
from ..formulas.noise import bias_jitter_split
from ..models.config import GeneratorOptions
from ..models.grid import INTACT, BranchModel, MeterBias, MeterPlan
from ..registry import system_id
from .attacks import AttackMixin

if TYPE_CHECKING:
    from ..models.plan import GeneratorSpec
    from .pandapower_types import PandapowerNet, PpcTables


class FdiaGenerator(AttackMixin):
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
        """Load the IEEE case, solve the base power flow, and draw from `seed`, in this fixed order: the
        meter plan (voltage buses, PMU buses, flow meters), six per-meter bias vectors, the current biases.
        With `meter_model` "hybrid" only PMUs read angles, and they read branch currents [WU26 eqs. 17-20]."""
        # pandapower is heavy and optional: imported only when generating
        import pandapower as pp
        import pandapower.networks as pn

        self.pp = pp
        self.C = system_id(system)  # "ieee118" and 118 both accepted, like every public entry point
        # `max_load_mw` keeps a larger load out of the attack targets (None: no cap): it is an area
        # equivalent (IEEE-145 has loads of 4 to 58 GW), with no power-flow solution near a 5% to 20% shift.
        # No load of the other seven ladder systems exceeds 1.1 GW.
        opts = GeneratorOptions(max_load_mw, meter_model)
        self.max_load_mw, hybrid = opts.max_load_mw, opts.meter_model == "hybrid"
        self.rng = np.random.default_rng(seed)
        self.seed, self.scan_key, self._jitter = seed, None, self.rng
        # the accuracy classes, split into a per-scan jitter and a per-meter bias (formulas.noise)
        self.SD = dict(ACCURACY_CLASS)
        self.SDj, self._bias_sd = bias_jitter_split(self.SD, jitter_frac=JITTER_FRAC)
        self.NET = getattr(pn, IEEE_CASES[self.C])
        self.base = self._open_case()
        self._load_tables(self.base)
        self._meter_plan(self.base, vbus_frac, pmu_frac, flow_frac, hybrid)
        self._edge_index(self.base)
        self._branch_physics(self.base._ppc)
        self._admittances(self.base._ppc)
        self._meter_bias()
        if hybrid:
            self._current_bias()

    @classmethod
    def from_spec(cls, spec: GeneratorSpec) -> FdiaGenerator:
        """The generator of `spec`: its case, meter plan and ratings."""
        m = spec.meters
        g = cls(spec.system, spec.seed, max_load_mw=spec.max_load_mw, meter_model=m.meter_model, **m.coverage)
        g._line_ratings, g._rating_delta = spec.ratings, spec.rating_delta
        return g

    def _open_case(self) -> PandapowerNet:
        """The pandapower case with its base power flow solved; the generator is the intact grid
        (self.contingency = INTACT, which the file attributes record)."""
        base = self.NET()
        self.contingency = INTACT
        self.pp.runpp(base)
        return base

    def _load_tables(self, base: PandapowerNet) -> None:
        """Which buses carry load, which loads can be attacked, and which buses inject nothing.
        `load_bus` has one entry per load element, aligned with `net.load`. A zero-injection bus is a
        pure junction (net injection exactly 0); a near-zero injection reading is still emitted there."""
        C = self.C
        _lb = base.load
        self.load_bus = _lb["bus"].values
        self.load_p0 = _lb["p_mw"].to_numpy(dtype=float)  # base active load per element, signed
        self.load_q0 = _lb["q_mvar"].to_numpy(dtype=float)  # base reactive load per element, signed
        # the angle reference; a local attack never moves it
        self.slack_bus = int(base.ext_grid.bus.values[0])
        p_load = _lb["p_mw"].abs().values
        # Attackable: active load above zero, off the slack bus, at most `max_load_mw`. A reactive-only
        # load (IEEE-300 buses 141, 183) would leave no P footprint yet be labelled attacked; a load on the
        # slack bus can never enter a region; a gigawatt area-equivalent load has no local solution.
        self._attackable_mask = (p_load > 0.0) & (self.load_bus != self.slack_bus)
        if self.max_load_mw is not None:
            self._attackable_mask &= p_load <= self.max_load_mw
        self.attackable_pos = np.where(self._attackable_mask)[0]
        # The stealthy families also skip every load on a generator bus, zero-MW condensers included: a
        # generator bus is too risky to falsify, the attacker's rule in the protocol followed here [BOY22].
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
        """Per-bus base load and generation [N, 2] (P, Q) and the case's limits [WU26 eqs. 21-23]: bus
        voltage limits and generator P and Q limits [N, 2], summed over co-located generators, ±inf where
        a bus has none. The slack is unbounded: its output is the balance of every state."""
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
        """The sparse meter plan, sampled once (`self.meters`): voltmeter buses, PMU buses, and a flow-meter
        mask over the branches; every injection bus is metered. Three draws from the seeded RNG, in this
        order. `hybrid` (angles at PMUs only, PMU branch currents) adds no draw."""
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

    def _branch_physics(self, ppc: PpcTables) -> None:
        """The per-unit branch and bus quantities makeYbus consumes, so a model has what the state
        estimator has. Rows are lines then transformers, like `self.ei`. Bus shunts are MW/MVAr at 1.0 pu."""
        # BR_G (column 23) is a pandapower extension carrying transformer iron losses: without it Ybus is
        # exact on IEEE-14 (no transformers) but wrong on IEEE-118 and 300. float64, not float32: float32
        # degrades the Ybus reconstruction from ~1e-14 to ~1e-4 on IEEE-300.
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
        # series admittance g_s + j b_s = 1 / (r + jx), so the edge set carries admittance, not just impedance
        _ys = series_admittance(self.branch.r, self.branch.x)
        self.edge_gs = np.real(_ys).astype(np.float64)  # series conductance, p.u.
        self.edge_bs = np.imag(_ys).astype(np.float64)  # series susceptance, p.u. (negative for inductive)
        self.edge_is_trafo = np.r_[np.zeros(self.n_lines), np.ones(self.E - self.n_lines)].astype(np.float64)
        self.bus_shunt_g = ppc["bus"][:, 4].real.astype(np.float64)
        self.bus_shunt_b = ppc["bus"][:, 5].real.astype(np.float64)

    def _admittances(self, ppc: PpcTables) -> None:
        """Ybus, the from and to branch-admittance matrices (from-end flow Sf = V_from * conj(Yf @ V)) and
        the bus lookup: `_ppc_row` maps a pandapower bus index to its ppc row (the orderings differ), and
        `_from_bus_ppc` is each branch's from bus as a ppc row."""
        from pandapower.pypower.makeYbus import makeYbus

        self._Ybus, self._Yf, self._Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
        self._base_mva = ppc["baseMVA"]
        self._ppc_row = self.base._pd2ppc_lookups["bus"]
        self._from_bus_ppc = ppc["branch"][:, 0].real.astype(int)
        self._ppc_branch = ppc["branch"]  # the ppc branch matrix, for the flow derivatives
        self._n_ppc_buses = ppc["bus"].shape[0]

    def _meter_bias(self) -> None:
        """The per-meter systematic bias (`self.bias`), constant across scans: relative for P, Q and
        flows, absolute for |V| and angle. Six draws from the seeded RNG, after the meter plan, in this
        order."""
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
        channel, drawn once after the six bias vectors, from the accuracy class `PMU_CURRENT_CLASS`.
        The per-scan jitter part is kept in `self._i_jitter`."""
        jit, bias = bias_jitter_split({"i": PMU_CURRENT_CLASS}, jitter_frac=JITTER_FRAC)
        self._i_jitter = jit["i"]
        self.bias = self.bias._replace(i=self.rng.normal(0, bias["i"], (self.E, 4)))
