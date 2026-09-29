"""The static description of a system: the per-branch pi model and the admittance matrices built
from it, which meters exist, the constant meter bias, and the N-1 contingency a generator was
built with. Nothing here changes from one scan to the next."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import torch

# One column of a measurement array, a view with no copy: numpy, or torch for a record bundle.
Column = Union[np.ndarray, "torch.Tensor"]


class NodeColumns(NamedTuple):
    """The four columns of `node_x`, `node_m` and `clean`, in the order every shard, stream and
    pool stores them. `NodeColumns.of(a)` gives named views of the last axis of any such array;
    `NODE` (a `NodeIndex`, the same fields as column indices) is for code that indexes."""

    v: Column  # |V|, voltage magnitude (pu)
    p_inj: Column  # P_inj, active injection (MW, or pu on baseMVA)
    q_inj: Column  # Q_inj, reactive injection (MVAr, or pu)
    theta: Column  # voltage angle (deg, or rad)

    @classmethod
    def of(cls, a: Column) -> NodeColumns:
        return cls(a[..., 0], a[..., 1], a[..., 2], a[..., 3])


class NodeIndex(NamedTuple):
    """Where each `NodeColumns` field sits on the last axis: `node_x[..., NODE.theta]`."""

    v: int
    p_inj: int
    q_inj: int
    theta: int


class EdgeColumns(NamedTuple):
    """The two columns of `edge_x`, `edge_m`, `edge_clean` and `edge_clean_full`."""

    p_from: Column  # P_from, active flow leaving the from end (MW, or pu)
    q_from: Column  # Q_from, reactive flow leaving the from end (MVAr, or pu)

    @classmethod
    def of(cls, a: Column) -> EdgeColumns:
        return cls(a[..., 0], a[..., 1])


class EdgeIndex(NamedTuple):
    """Where each `EdgeColumns` field sits on the last axis."""

    p_from: int
    q_from: int


class CurrentColumns(NamedTuple):
    """The four columns of `pmu_i` and `pmu_i_m`: the branch-current phasor a PMU reads at each end of
    a branch [WU26, eqs. 19-20], the real and imaginary part, per unit on the base current. At the
    from end the current leaving the from bus into the branch (`Yf V`), at the to end the current
    leaving the to bus into the branch (`Yt V`); a column is metered only where a PMU sits at that end."""

    re_from: Column
    im_from: Column
    re_to: Column
    im_to: Column

    @classmethod
    def of(cls, a: Column) -> CurrentColumns:
        return cls(a[..., 0], a[..., 1], a[..., 2], a[..., 3])


class CurrentIndex(NamedTuple):
    """Where each `CurrentColumns` field sits on the last axis."""

    re_from: int
    im_from: int
    re_to: int
    im_to: int


class BranchColumns(NamedTuple):
    """The eight columns of `edge_attr`, the static per-unit branch physics."""

    r: Column
    x: Column
    b: Column
    g: Column
    gs: Column
    bs: Column
    tap: Column
    shift: Column

    @classmethod
    def of(cls, a: Column) -> BranchColumns:
        return cls(*(a[..., i] for i in range(8)))


class BranchIndex(NamedTuple):
    """Where each `BranchColumns` field sits on the last axis."""

    r: int
    x: int
    b: int
    g: int
    gs: int
    bs: int
    tap: int
    shift: int


NODE = NodeIndex(0, 1, 2, 3)  # column indices: node_x[..., NODE.theta]
EDGE = EdgeIndex(0, 1)
BRANCH = BranchIndex(0, 1, 2, 3, 4, 5, 6, 7)
CURRENT = CurrentIndex(0, 1, 2, 3)  # pmu_i[..., CURRENT.re_to]


class BranchModel(NamedTuple):
    """The per-branch pi model [MP19], one entry per branch, per unit; the shapes a shard stores."""

    r: np.ndarray  # series resistance
    x: np.ndarray  # series reactance
    b: np.ndarray  # charging susceptance
    g: np.ndarray  # charging conductance (transformer iron losses)
    tap: np.ndarray  # turns ratio, 1 (or 0, read as 1) for lines
    shift_deg: np.ndarray  # phase shift in degrees
    status: Optional[np.ndarray] = None  # 1 in service, 0 out; None = all in service


class Admittances(NamedTuple):
    """The nodal and branch admittance matrices of one topology, complex per unit."""

    ybus: np.ndarray  # [N, N]
    yf: np.ndarray  # [E, N] from-end: I_f = Yf @ V
    yt: np.ndarray  # [E, N] to-end: I_t = Yt @ V


class MeterPlan(NamedTuple):
    """The sparse metering plan, sampled once per generator: which buses carry a voltage-magnitude
    meter, which carry a PMU (|V| and angle), which carry P/Q injection meters, and which branches
    carry a flow meter. Under the hybrid meter model (the plan's D10) a voltmeter bus reads |V| only
    and each PMU also reads the current phasor of every in-service branch at its bus; the v0.8.3
    model (both flags off) writes an angle at every voltmeter bus and no currents."""

    vbus: set[int]
    pmu: set[int]
    inj: list[int]
    flow: np.ndarray  # [E] bool
    angle_at_pmu_only: bool = False  # hybrid: the voltage angle is a PMU channel
    pmu_currents: bool = False  # hybrid: branch-current phasors at every PMU bus [WU26, eqs. 19-20]


class MeterBias(NamedTuple):
    """The per-meter SYSTEMATIC bias drawn once (constant across scans): relative for P/Q
    injections and flows, absolute for |V| and angle (radians)."""

    pi: np.ndarray  # [N]
    qi: np.ndarray  # [N]
    v: np.ndarray  # [N]
    va: np.ndarray  # [N]
    pf: np.ndarray  # [E]
    qf: np.ndarray  # [E]
    i: Optional[np.ndarray] = None  # [E, 4] relative, the PMU branch-current channels (hybrid meters only)


class Outage(NamedTuple):
    """The N-1 contingency a generator was built with: the pandapower line id (None = intact), its
    branch position, its name, its terminals, and its intact-case active flow (the contingency's
    size)."""

    line: Optional[int]
    pos: int
    name: str
    from_bus: int
    to_bus: int
    base_flow_mw: float


INTACT = Outage(None, -1, "", -1, -1, float("nan"))


class PseudoLinks(NamedTuple):
    """Where [WU26, eq. (3)] can place a pseudo voltage phasor: one row per metered branch end whose
    bus has a PMU and whose far bus does not (`formulas.estimation.pmu_pseudo_links`). The
    admittances are the branch's pi-model entries seen from the metered end, taps and shifts
    included: I_near = y_nn V_near + y_nf V_far."""

    near: np.ndarray  # [J] the PMU bus
    far: np.ndarray  # [J] the bus the pseudo-measurement lands on
    edge: np.ndarray  # [J] the branch
    end: np.ndarray  # [J] 0 when the PMU reads the from end, 1 the to end
    y_nn: np.ndarray  # [J] complex, Yf[e, from] or Yt[e, to]
    y_nf: np.ndarray  # [J] complex, Yf[e, to] or Yt[e, from]


class PseudoVoltages(NamedTuple):
    """The pseudo voltage phasors of [WU26, eq. (3)] and their first-order propagated noise
    (`formulas.estimation.pmu_pseudo_voltages`), NaN at the buses no link reaches."""

    v: np.ndarray  # [..., N] |V| pu
    theta: np.ndarray  # [..., N] angle rad
    var_v: np.ndarray  # [..., N] variance of |V|, pu^2
    var_theta: np.ndarray  # [..., N] variance of the angle, rad^2
    reached: np.ndarray  # [N] bool, a link lands on the bus
