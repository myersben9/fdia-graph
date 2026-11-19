"""The dataset generation engine: attack simulation and realistic meter readings on a test system.

The grid is a `Case` with its `Network`, the meters a `MeterPlan` with its `Measurement` and
`NoiseModel`, the same classes for an IEEE case and the New York grid. An IEEE case
(`Network.from_ieee`) requires pandapower ('fdia-graph[generate]'), which solves its base power
flow once. A benign scan is emitted exactly from a stored state (no re-solve); only attacks
re-solve, locally. The attack families are At (the slow ramp) and Am (the overload attack) [WU26].

FdiaGenerator holds the state (the grid, the meters, the random stream) and one object of each
part (`engine.base.Part`), named after its class: `emitter` (measurement.py, meter readings from a
state), `scan_loads` (physics.py, a scan's load and generation) and the parts of attacks/ (`area`,
`false_state`, `ramp_frames`, `minimizer`, `overload`, `episode_designs`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..constants.grid import BusType
from ..constants.meters import FLOW_KINDS, MeterKind
from ..models.config import GeneratorOptions, MeterSettings
from ..models.frames import LayerSet
from ..sensing.measure import Measurement
from ..sensing.meters import MeterPlan
from ..sensing.noise import NoiseModel
from ..systems.network import Network
from .attacks.area import AttackArea
from .attacks.episodes import EpisodeDesigns
from .attacks.false_state import FalseState
from .attacks.minimize import Minimizer
from .attacks.overload import Overload
from .attacks.ramp_frames import RampFrames
from .measurement import Emitter
from .physics import LoadSplit, ScaledSplit, ScanLoads

if TYPE_CHECKING:
    from ..models.plan import GeneratorSpec


class FdiaGenerator:
    def __init__(
        self,
        net: Network,
        seed: int = 123,
        meters: MeterSettings = MeterSettings(),
        max_load_mw: Optional[float] = 2000.0,
        plan: Optional[MeterPlan] = None,
        spec: LayerSet = LayerSet(),
        split: Optional[LoadSplit] = None,
    ) -> None:
        """The generator of any grid: its `Network`, and the meter plan of `meters` drawn from `seed`
        (`MeterPlan.from_coverage`: the meters, then their biases). Only PMUs read angles, and they
        read branch currents [WU26 eqs. 17-20]. A `plan` is used as given and nothing is drawn. A
        scan is laid out in the layers of `spec` (a plan that meters both ends of a branch needs `LayerSet(both_ends=True)`).
        `split` says how the pool's injections divide into load and generation (None: a scaled pool)."""
        options = GeneratorOptions(max_load_mw)
        self.split = ScaledSplit.of_case(net.case) if split is None else split
        self._open(net, seed, spec, options.max_load_mw)
        self.use_plan(MeterPlan.from_coverage(self.case, meters, self.rng) if plan is None else plan)

    def _open(self, net: Network, seed: int, spec: LayerSet, max_load_mw: Optional[float]) -> None:
        """The grid, the generator's random stream and the tables of the case."""
        self.spec = spec
        # `max_load_mw` keeps a larger load out of the attack targets (None: no cap): it is an area
        # equivalent (IEEE-145 has loads of 4 to 58 GW), with no power-flow solution near a 5% to 20% shift.
        # No load of the other seven ladder systems exceeds 1.1 GW.
        self.max_load_mw = max_load_mw
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        # the timestep whose jitter the next emission draws (a timeline walk sets it), or None: the
        # jitter then comes from `rng` like every other draw
        self.scan_key: Optional[int] = None
        self.net, self.case = net, net.case
        self.C = len(self.case.bus_id)
        self.ei = np.vstack([self.case.f, self.case.t]).astype(np.int32)  # [2, E]: from bus, to bus
        self.E = self.ei.shape[1]
        # the parts, each one job in one class: a call `g.<part>.<method>` names the class of the method
        self.emitter, self.scan_loads, self.area = Emitter(self), ScanLoads(self), AttackArea(self)
        self.false_state, self.ramp_frames = FalseState(self), RampFrames(self)
        self.minimizer, self.overload = Minimizer(self), Overload(self)
        self.episode_designs = EpisodeDesigns(self)
        self._load_tables()
        self._case_limits()

    def use_plan(self, plan: MeterPlan) -> None:
        """The meter plan with its measurement function, its noise and the masks of its meters."""
        self.plan, self.meas, self.noise = plan, Measurement(self.net, plan), NoiseModel(plan)
        self.masks = self.emitter.layers(np.ones(len(plan.kind), np.uint8))
        self.pmu_buses = plan.buses(MeterKind.VA)
        self.flow_metered = self.masks.edge[:, FLOW_KINDS.index(MeterKind.PF)] > 0

    @classmethod
    def from_spec(cls, spec: GeneratorSpec) -> FdiaGenerator:
        """The generator of `spec`: its case, meter plan and ratings."""
        g = cls(spec.net, spec.seed, spec.meters, spec.max_load_mw, split=spec.split)
        g.overload.ratings, g.overload.rating_delta = spec.ratings, spec.rating_delta
        return g

    def _load_tables(self) -> None:
        """Which buses carry load, which loads can be attacked, and which buses inject nothing. The
        load table has one row per loaded bus, in bus order. A zero-injection bus is a pure junction
        (net injection exactly 0); a near-zero injection reading is still emitted there."""
        case, base = self.case, self.split.base_load
        self.load_bus = np.flatnonzero((base[:, 0] != 0) | (base[:, 1] != 0))
        self.load_p0 = base[self.load_bus, 0]  # base active load per row, signed (MW)
        self.load_q0 = base[self.load_bus, 1]  # base reactive load per row, signed (MVAr)
        # the angle reference; a local attack never moves it
        self.slack_bus = int(np.flatnonzero(case.bus_type == BusType.REF)[0])
        p_load = np.abs(self.load_p0)
        # Attackable: active load above zero, off the slack bus, at most `max_load_mw`, at one of the
        # grid's own buses. A reactive-only load (IEEE-300 buses 141, 183) would leave no P footprint yet
        # be labelled attacked; a load on the slack bus can never enter a region; a gigawatt
        # area-equivalent load has no local solution; a neighbour's equivalent bus is not the grid's.
        attackable = (p_load > 0.0) & (self.load_bus != self.slack_bus) & case.own_bus[self.load_bus]
        if self.max_load_mw is not None:
            attackable &= p_load <= self.max_load_mw
        self.attackable_pos = np.where(attackable)[0]
        # A unit on a PQ bus is a static generator (IEEE-89 and 300 feed buses from them alone); one
        # that produces nothing (all of IEEE-200's) injects nothing.
        idle = self.split.idle_units(case)
        # The stealthy families also skip every load on a generator bus, zero-MW condensers included: a
        # generator bus is too risky to falsify, the attacker's rule in the protocol followed here [BOY22].
        self.stealthy_pos = np.where(attackable & ~np.isin(self.load_bus, case.unit_bus[~idle]))[0]
        # every bus with a load, a shunt or a unit that is not idle injects; the rest are junctions
        shunted = np.flatnonzero((case.gs != 0) | (case.bs != 0))
        injecting = np.unique(np.r_[self.load_bus, shunted, case.unit_bus[~idle]])
        self.zero_inj = np.setdiff1d(np.arange(self.C), injecting).tolist()

    def _case_limits(self) -> None:
        """The case's limits [WU26 eqs. 21-23]: bus voltage limits and generator P and Q limits [N, 2],
        summed over co-located generators, ±inf where a bus has none. The slack is unbounded: its output
        is the balance of every state."""
        case, C = self.case, self.C
        self.p_lim = np.tile([-np.inf, np.inf], (C, 1))
        self.q_lim = np.tile([-np.inf, np.inf], (C, 1))
        gens = np.flatnonzero(case.bus_type[case.unit_bus] == BusType.PV)  # the units that hold a voltage
        at = case.unit_bus[gens]
        self.has_gen = np.zeros(C, bool)  # a generator on the bus, zero-MW condensers included
        self.has_gen[at] = True
        if len(at):
            self.p_lim[at] = self.q_lim[at] = 0.0  # summed over the bus's generators below
            np.add.at(self.p_lim, at, np.stack([case.pmin[gens], case.pmax[gens]], axis=1))
            np.add.at(self.q_lim, at, np.stack([case.qmin[gens], case.qmax[gens]], axis=1))
        self.v_case = np.stack([case.vmin, case.vmax], axis=1)
