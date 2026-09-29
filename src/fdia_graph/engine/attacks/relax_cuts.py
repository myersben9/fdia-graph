"""The valid cuts that tighten the certifier's relaxation (docs/plans/RELAX_CERTIFIER_PLAN.md,
section 2.1), each family a function of the relaxation it tightens:

- **bounds:** optimization-based bound tightening of each area bus's W_ii and of how far its voltage
  can move (|V_i - V_i^true| <= rho_i), then the angle bounds these imply, as wedges on V_i and on
  each W_ij.
- **qc:** the QC relaxation [CHV16]: per pair of area buses, the voltage product and the cosine and
  sine of the angle difference, each with its convex envelope, and W_ij as their McCormick products.
- **cycle:** one angle per area bus, each pair's angle difference the difference of its ends' angles
  (so the angle differences sum to zero around every cycle of the area), and V_i tied to its bus's
  angle by the same envelopes.

Every cut holds at every rank-one point (an AC state) inside the bounds, so an attack the search can
return satisfies all of them (tests/test_certify.py puts the search's attack into each family).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ...formulas.relax import angle_bound, cos_envelope, mccormick, polygon_radius, sin_envelope

if TYPE_CHECKING:
    from .certify import _Relaxation

DIRECTIONS = (
    8  # directions of the polygon that bounds each voltage's move (rho within 1 / cos(pi / 8) = 1.08)
)
BOUND_SLACK = 1e-6  # widens every tightened bound, so the solver's tolerance never cuts a feasible point
RIGHT_ANGLE = math.pi / 2 - 1e-3  # angle bounds at or beyond this give no usable envelope


def tighten(relax: _Relaxation, t: int, cutoff: int, time_limit: float = 10.0) -> bool:
    """Tighten how far each area bus's voltage can move at snapshot t (rho) by bounding the move in
    DIRECTIONS directions over the mixed-integer relaxation with at most `cutoff` devices, each solve
    stopped at `time_limit` seconds and read by SCIP's dual bound (valid at any stop). False when
    the relaxation is infeasible: no attack in the area tampers `cutoff` devices or fewer."""
    import cvxpy as cp

    L = relax.layout
    c = cp.Parameter(L.n)
    prob = relax.build([t], cutoff=cutoff, objective=c)[0]
    V0 = relax.V[t][L.area]
    rho = relax.rho[t].copy()
    phi = 2.0 * np.pi * np.arange(DIRECTIONS) / DIRECTIONS
    for a in range(len(L.area)):
        rows = [_unit(L.n, L.e[a], -np.cos(p)) + _unit(L.n, L.f[a], -np.sin(p)) for p in phi]
        values = [_minimum(prob, c, row, time_limit) for row in rows]
        if any(math.isinf(v) and v > 0 for v in values):
            return False
        # max u . (V - V0) <= -(dual bound of min -u . V) - u . V0 per direction
        h = np.array([-v - (np.cos(p) * V0[a].real + np.sin(p) * V0[a].imag) for v, p in zip(values, phi)])
        rho[a] = min(rho[a], polygon_radius(h, DIRECTIONS) + BOUND_SLACK)
        relax.rho[t] = rho
    return True


def _unit(n: int, k: int, value: float) -> np.ndarray:
    row = np.zeros(n)
    row[k] = value
    return row


def _minimum(prob, c, row: np.ndarray, time_limit: float) -> float:
    """A valid lower bound on min row . x over the problem: SCIP's dual bound (+inf when infeasible,
    -inf when it proves nothing)."""
    from cvxpy.error import SolverError

    from .certify import _bounding_scip

    c.value = row
    solver = _bounding_scip()
    try:
        prob.solve(solver=solver, scip_params={"limits/time": float(time_limit)})
    except SolverError:  # stopped with no feasible point found: the dual bound still holds
        pass
    if solver.status == "infeasible":
        return math.inf
    return solver.dual_bound if math.isfinite(solver.dual_bound) else -math.inf


def _bounds(relax: _Relaxation, t: int) -> dict[str, np.ndarray]:
    """Snapshot t's bounds on the polar quantities: |V| per area bus, the angle move delta per area
    bus, and per pair of area buses the largest angle-difference move m (delta_i + delta_j)."""
    L = relax.layout
    V0 = relax.V[t][L.area]
    wlo, whi = relax.box[t]
    rho = relax.rho[t]
    v_lo = np.maximum(np.sqrt(wlo), np.abs(V0) - rho)
    v_hi = np.minimum(np.sqrt(whi), np.abs(V0) + rho)
    delta = angle_bound(rho, V0)
    return {"v_lo": v_lo, "v_hi": v_hi, "delta": delta, "m": delta[L.pair_a] + delta[L.pair_b]}


def _rotated_pair(relax: _Relaxation, x, t: int, k: int) -> tuple:
    """W_ij e^{-j phi0} of pair k (phi0 its true angle difference), linear in x: rank one gives
    |V_i| |V_j| e^{j dphi}, dphi the move of the angle difference."""
    L = relax.layout
    phi0 = relax.x0[t][L.re[k]] + 1j * relax.x0[t][L.im[k]]
    cos0, sin0 = math.cos(np.angle(phi0)), math.sin(np.angle(phi0))
    re, im = x[L.re[k]], x[L.im[k]]
    return cos0 * re + sin0 * im, cos0 * im - sin0 * re


def _rotated_bus(relax: _Relaxation, x, t: int, a: int) -> tuple:
    """V_i e^{-j theta0} of area bus a, linear in x: rank one gives |V_i| e^{j dtheta}."""
    L = relax.layout
    th0 = float(np.angle(relax.V[t][L.area[a]]))
    e, f = x[L.e[a]], x[L.f[a]]
    return math.cos(th0) * e + math.sin(th0) * f, math.cos(th0) * f - math.sin(th0) * e


def angle_cuts(relax: _Relaxation, x, t: int) -> list:
    """The wedges the tightened bounds imply: each V_i within delta_i of its true angle and each W_ij
    within m_ij of its true angle difference, with the real part at least the smallest magnitude times
    cos of the bound."""
    import cvxpy as cp

    L, bd = relax.layout, _bounds(relax, t)
    cons = []
    for a in range(len(L.area)):
        if bd["delta"][a] < RIGHT_ANGLE:
            u, p = _rotated_bus(relax, x, t, a)
            cons += [cp.abs(p) <= math.tan(bd["delta"][a]) * u, u >= bd["v_lo"][a] * math.cos(bd["delta"][a])]
    for k in range(len(L.pair_a)):
        if bd["m"][k] < RIGHT_ANGLE:
            u, p = _rotated_pair(relax, x, t, k)
            low = bd["v_lo"][L.pair_a[k]] * bd["v_lo"][L.pair_b[k]] * math.cos(bd["m"][k])
            cons += [cp.abs(p) <= math.tan(bd["m"][k]) * u, u >= low]
    return cons


def qc(relax: _Relaxation, x, t: int, cycle: bool) -> list:
    """The QC relaxation of snapshot t: |V| per area bus with W_ii >= |V|^2 and the secant; per pair
    the product |V_i| |V_j|, cos and sin of the angle-difference move in their envelopes and W_ij
    their McCormick products. With `cycle`, the moves are differences of bus angles and V_i is its
    bus's |V| times the envelopes of its angle."""
    import cvxpy as cp

    L, bd = relax.layout, _bounds(relax, t)
    na, npair = len(L.area), len(L.pair_a)
    aux = {name: cp.Variable(na) for name in ("vm", "dth", "cth", "sth")}
    aux.update({name: cp.Variable(npair) for name in ("vv", "dphi", "cs", "sn")})
    relax.aux[t] = aux
    lo, hi, w = bd["v_lo"], bd["v_hi"], x[L.w]
    cons = [aux["vm"] >= lo, aux["vm"] <= hi, cp.square(aux["vm"]) <= w]
    cons += [w <= cp.multiply(lo + hi, aux["vm"]) - lo * hi]
    cons += [cp.norm(cp.vstack([x[L.e], x[L.f]]), axis=0) <= aux["vm"]]
    usable = [k for k in range(npair) if bd["m"][k] < RIGHT_ANGLE]
    for k in usable:
        cons += _pair(relax, x, t, k, bd)
    cons += _rectangular(relax, x, t)
    if cycle:
        cons += _cycle(relax, x, t, bd, usable)
    return cons


def _rectangular(relax: _Relaxation, x, t: int) -> list:
    """W_ij = V_i conj(V_j) in rectangular coordinates, each product relaxed to its McCormick envelope
    over the box of e and f (the true value +- rho, inside +- |V|max): Re W_ij = e_i e_j + f_i f_j,
    Im W_ij = f_i e_j - e_i f_j. It needs no angle bound."""
    import cvxpy as cp

    L = relax.layout
    x0, rho = relax.x0[t], relax.rho[t]
    vmax = np.sqrt(relax.box[t][1])
    lo_e, hi_e = np.maximum(x0[L.e] - rho, -vmax), np.minimum(x0[L.e] + rho, vmax)
    lo_f, hi_f = np.maximum(x0[L.f] - rho, -vmax), np.minimum(x0[L.f] + rho, vmax)
    npair = len(L.pair_a)
    prods = {name: cp.Variable(npair) for name in ("ee", "ff", "fe", "ef")}
    relax.aux.setdefault(t, {}).update({"rect_" + name: v for name, v in prods.items()})
    cons = [x[L.re] == prods["ee"] + prods["ff"], x[L.im] == prods["fe"] - prods["ef"]]
    for k in range(npair):
        a, b = int(L.pair_a[k]), int(L.pair_b[k])
        pairs = {
            "ee": (x[L.e[a]], x[L.e[b]], (lo_e[a], hi_e[a]), (lo_e[b], hi_e[b])),
            "ff": (x[L.f[a]], x[L.f[b]], (lo_f[a], hi_f[a]), (lo_f[b], hi_f[b])),
            "fe": (x[L.f[a]], x[L.e[b]], (lo_f[a], hi_f[a]), (lo_e[b], hi_e[b])),
            "ef": (x[L.e[a]], x[L.f[b]], (lo_e[a], hi_e[a]), (lo_f[b], hi_f[b])),
        }
        for name, (u, v, ub, vb) in pairs.items():
            cons += _product(prods[name][k], u, v, ub, vb)
    return cons


def _trig(angle, cosine, sine, m: float) -> list:
    """cos and sin of an angle in [-m, m] inside their envelopes, and on the unit disc."""
    import cvxpy as cp

    c, cos_m = cos_envelope(m)
    slope, offset = sin_envelope(m)
    return [
        angle >= -m,
        angle <= m,
        cosine <= 1 - c * cp.square(angle),
        cosine >= cos_m,
        sine <= slope * (angle - m / 2) + offset,
        sine >= slope * (angle + m / 2) - offset,
        cp.abs(sine) <= math.sin(m),
        cp.norm(cp.hstack([cosine, sine])) <= 1,
    ]


def _product(p, a, b, a_box: tuple[float, float], b_box: tuple[float, float]) -> list:
    """p = a b relaxed to its McCormick envelope."""
    out = []
    for sense, ca, cb, c0 in mccormick(a_box, b_box):
        rhs = ca * a + cb * b + c0
        out.append(p >= rhs if sense == ">=" else p <= rhs)
    return out


def _pair(relax: _Relaxation, x, t: int, k: int, bd: dict[str, np.ndarray]) -> list:
    """Pair k's QC constraints: vv = |V_i| |V_j|, W' = vv (cos dphi + j sin dphi)."""
    L, aux, m = relax.layout, relax.aux[t], float(bd["m"][k])
    a, b = int(L.pair_a[k]), int(L.pair_b[k])
    va, vb = (bd["v_lo"][a], bd["v_hi"][a]), (bd["v_lo"][b], bd["v_hi"][b])
    vv_box = (va[0] * vb[0], va[1] * vb[1])
    re, im = _rotated_pair(relax, x, t, k)
    vv, cs, sn = aux["vv"][k], aux["cs"][k], aux["sn"][k]
    cons = _trig(aux["dphi"][k], cs, sn, m) + _product(vv, aux["vm"][a], aux["vm"][b], va, vb)
    cons += _product(re, vv, cs, vv_box, (math.cos(m), 1.0))
    return cons + _product(im, vv, sn, vv_box, (-math.sin(m), math.sin(m)))


def _cycle(relax: _Relaxation, x, t: int, bd: dict[str, np.ndarray], usable: list[int]) -> list:
    """Bus angles: each pair's move is the difference of its ends' moves (zero sum around every cycle),
    and V_i e^{-j theta0} = |V_i| (cos dtheta + j sin dtheta) by the same envelopes."""
    L, aux = relax.layout, relax.aux[t]
    cons = [aux["dphi"][k] == aux["dth"][int(L.pair_a[k])] - aux["dth"][int(L.pair_b[k])] for k in usable]
    for a in range(len(L.area)):
        m = float(bd["delta"][a])
        if m >= RIGHT_ANGLE:
            continue
        u, p = _rotated_bus(relax, x, t, a)
        va = (bd["v_lo"][a], bd["v_hi"][a])
        cons += _trig(aux["dth"][a], aux["cth"][a], aux["sth"][a], m)
        cons += _product(u, aux["vm"][a], aux["cth"][a], va, (math.cos(m), 1.0))
        cons += _product(p, aux["vm"][a], aux["sth"][a], va, (-math.sin(m), math.sin(m)))
    return cons


def lift(relax: _Relaxation, t: int, V: np.ndarray) -> None:
    """Give snapshot t's QC variables the values of the AC voltages V (ppc order), the point they
    relax (for the validity tests)."""
    L, aux = relax.layout, relax.aux.get(t)
    if aux is None:
        return
    va, v0 = V[L.area], relax.V[t][L.area]
    dth = np.angle(va * np.conj(v0))
    dphi = dth[L.pair_a] - dth[L.pair_b]
    values = {"vm": np.abs(va), "dth": dth, "cth": np.cos(dth), "sth": np.sin(dth)}
    values.update(vv=np.abs(va[L.pair_a]) * np.abs(va[L.pair_b]), dphi=dphi, cs=np.cos(dphi), sn=np.sin(dphi))
    e, f = va.real, va.imag
    a, b = L.pair_a, L.pair_b
    values.update(rect_ee=e[a] * e[b], rect_ff=f[a] * f[b], rect_fe=f[a] * e[b], rect_ef=e[a] * f[b])
    for name, var in aux.items():
        var.value = values[name]
