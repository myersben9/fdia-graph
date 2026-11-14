"""The interchange check: what the operator reads across the ties against what the neighbours' own meters read
(N62, N67). The New York ends of the ties carry the attack's false interchange, the neighbours' ends do not, so the
gap between them (`tie_gap`) leaves its benign noise once an attack moves the interchange, and `interchange_alarm` names the
first scan it does. `seen_interfaces` is the interface flow the operator reads from its own scans."""

from __future__ import annotations

import numpy as np

from fdia_graph.engine.sensing.meters import JITTER_FRAC, KINDS, MeterPlan
from fdia_graph.generation.nygrid.case import Case
from fdia_graph.generation.nygrid.interfaces import Interface
from fdia_graph.generation.nygrid.network import Network
from fdia_graph.operator.day import OperatorDay

NEIGHBOUR_SEED = 20190101  # the neighbours' tie meters: one bias each, the same every day


def tie_meters(case: Case, plan: MeterPlan) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ties to a neighbour with a P-flow meter at their New York end: the branches, which end is New
    York's (0 from, 1 to) and the row of that meter in the plan."""
    pf_k = KINDS.index("pf")
    ny_f, ny_t = case.is_ny[case.f], case.is_ny[case.t]
    out = []
    for b in np.flatnonzero(ny_f != ny_t):
        end = 0 if ny_f[b] else 1
        rows = np.flatnonzero((plan.kind == pf_k) & (plan.branch == b) & (plan.end == end))
        if len(rows):
            out.append((int(b), end, int(rows[0])))
    a = np.array(out, int).reshape(-1, 3)
    return a[:, 0], a[:, 1], a[:, 2]


def neighbour_reads(
    plan: MeterPlan, rows: np.ndarray, far: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """The neighbours' readings `[T, K]` of the flows `far` at their ends of the ties, through meters of the
    same accuracy class as the New York end (`rows` in the plan): a constant bias drawn from a fixed
    stream, so the same every day, and a fresh jitter per scan from `rng`, split as in M2."""
    sr, sa = plan.sigma_rel[rows], plan.sigma_abs[rows]
    rel = sr > 0
    bias = np.random.default_rng(NEIGHBOUR_SEED).normal(
        0.0, np.sqrt(1.0 - JITTER_FRAC**2) * np.where(rel, sr, sa)
    )
    jitter = JITTER_FRAC * sr * np.abs(far) + np.where(rel, sa, JITTER_FRAC * sa)
    biased = np.where(rel, far * (1.0 + bias), far + bias)
    return biased + rng.normal(0.0, 1.0, far.shape) * jitter


def tie_gap(day: OperatorDay, z: np.ndarray) -> np.ndarray:
    """`[T]` the net interchange the operator reads (the P-flow meters at the New York end of every tie
    to a neighbour, as flow leaving New York) less what the neighbours meter at their own end, turned to
    the same direction (N62). Ties are metered at both ends and the readings exchanged, so a gap beyond
    the two meters' noise shows that the reported interchange is wrong. The neighbours' jitter is drawn
    from a stream keyed by the day, so the attacked and the benign call see the same neighbour readings."""
    c, plan = day.case, day.plan
    br, end, rows = tie_meters(c, plan)
    pf, _, pt, _ = Network(c).flows(day.vm, day.va)
    far = np.where(end == 0, -pt[:, br], -pf[:, br])  # the neighbour's end, in the New York direction
    day_key = int(np.asarray(day.time)[0]) if len(day.time) else 0
    neighbour = neighbour_reads(plan, rows, far, np.random.default_rng([NEIGHBOUR_SEED, day_key]))
    return np.nan_to_num(z[:, rows]).sum(axis=1) - neighbour.sum(axis=1)


def seen_interfaces(plan: MeterPlan, interfaces: list[Interface], z: np.ndarray) -> np.ndarray:
    """`[T, I]` each interface's flow as the operator reads it: the signed sum of its branches' P-flow
    meters, at the from end, or the to end with its sign turned where the from end has none."""
    pf = KINDS.index("pf")
    out = np.zeros((len(z), len(interfaces)))
    for i, it in enumerate(interfaces):
        for b, s in zip(it.branches, it.signs):
            mine = np.flatnonzero((plan.kind == pf) & (plan.branch == b))
            rows = {int(plan.end[r]): int(r) for r in mine}
            if 0 in rows:
                out[:, i] += s * z[:, rows[0]]
            else:
                out[:, i] -= s * z[:, rows[1]]
    return out


def interchange_alarm(
    day: OperatorDay, tie_a: np.ndarray, tie_b: np.ndarray, calib: tuple[float, float] | None = None
) -> int | None:
    """The first scan at or after the onset whose read interchange is off the neighbours' meters by
    more than four standard deviations of the gap's benign noise (N62). `calib` is that noise's mean and
    standard deviation (MW), calibrated on other days; by default the benign
    day's own, which is in sample."""
    if day.onset < 0:
        return None
    b = tie_b[np.isfinite(tie_b)]
    mean, sd = (float(np.mean(b)), float(np.std(b))) if calib is None else calib
    off = np.abs(tie_a - mean) > 4.0 * sd
    hits = np.flatnonzero(off[day.onset :])
    return day.onset + int(hits[0]) if len(hits) else None
