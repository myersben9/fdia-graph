"""The AC network model: branch admittances, bus injections and branch flows [AE04, ch. 2], [MP19].

This is the one implementation of the measurement function h(x) that the generator, the loader
and the state estimator share: `ac_measurement` assembles the estimator's h from `bus_injections`
and `branch_flows`, and `ac_jacobian` is its closed-form Jacobian. The estimator's torch twin
(fdia_graph.se.base.SEBase._h_t) is kept for callers that differentiate through it;
tests/test_formulas.py pins it to the functions here.

Every function keeps the exact floating-point expression the callers used before it existed, so
released shards and streams reproduce bit for bit: `branch_flows` evaluates V_f * conj(Yf @ V)
for one voltage vector and V[:, f] * conj(V @ Yf.T) for a stack, which are the two orders the
generator and the loader always used.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from scipy.sparse import spmatrix

    # Ybus, Yf, Yt: pandapower's makeYbus builds them sparse; a dense matrix works the same
    Admittance = Union[np.ndarray, spmatrix]

from ..models.grid import (  # noqa: F401  re-exported: defined here before the models package
    Admittances,
    BranchModel,
)


def complex_voltages(vm: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
    """Bus voltage phasors V = |V| e^{j theta} [AE04, eq. 2.1].

    vm        : [..., N] voltage magnitude (pu)
    theta_deg : [..., N] voltage angle (degrees)
    returns   : [..., N] complex128
    """
    return vm * np.exp(1j * np.deg2rad(theta_deg))


def series_admittance(r: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Series admittance of a branch, y_s = g_s + j b_s = 1 / (r + j x) [MP19, branch model].

    A branch with no impedance (|r + jx| below 1e-12) has no series path and contributes 0.
    r, x    : [E] per-unit resistance and reactance
    returns : [E] complex128
    """
    z = np.asarray(r, np.float64) + 1j * np.asarray(x, np.float64)
    ys = np.zeros_like(z, dtype=np.complex128)
    nz = np.abs(z) > 1e-12
    ys[nz] = 1.0 / z[nz]
    return ys


def branch_admittances(
    branch: BranchModel,
    edge_index: np.ndarray,
    n_bus: int,
    bus_shunt_g: Optional[np.ndarray] = None,
    bus_shunt_b: Optional[np.ndarray] = None,
    base_mva: float = 100.0,
) -> Admittances:
    """Ybus [N, N], Yf [E, N] and Yt [E, N] from the per-branch pi model [MP19, makeYbus].

    Series admittance y_s, charging admittance (g + j b) split half per end, tap ratio and phase
    shift on the from side, in-service status; bus shunts (MW and MVAr at 1 pu) on the Ybus
    diagonal as (g_sh + j b_sh) / base_mva. With a complex bus voltage vector V, `V[f] * conj(Yf @ V)`
    is the from-end branch flow and `V * conj(Ybus @ V)` the bus injection, both per unit.

        y_tt = y_s + (g + j b) / 2
        y_ff = y_tt / (tap * conj(tap)),  y_ft = -y_s / conj(tap),  y_tf = -y_s / tap

    branch     : the per-branch physics, a BranchModel
    edge_index : [2, E] from-bus and to-bus of every branch, in the bus order of the outputs
    returns    : Admittances(ybus, yf, yt), complex128; unpacks as (Ybus, Yf, Yt)
    """
    E = len(branch.r)
    stat = np.ones(E) if branch.status is None else np.asarray(branch.status, np.float64)
    ys = stat * series_admittance(branch.r, branch.x)
    bc = stat * (np.asarray(branch.g, np.float64) + 1j * np.asarray(branch.b, np.float64))
    tap = np.asarray(branch.tap, np.float64)
    t = np.where(tap == 0.0, 1.0, tap) * np.exp(1j * np.pi / 180 * np.asarray(branch.shift_deg, np.float64))
    ytt = ys + bc / 2
    yff = ytt / (t * np.conj(t))
    yft = -ys / np.conj(t)
    ytf = -ys / t
    f, to = np.asarray(edge_index)
    rows = np.arange(E)
    Yf = np.zeros((E, n_bus), np.complex128)  # from-end: I_f = Yf @ V
    Yf[rows, f] += yff
    Yf[rows, to] += yft
    Yt = np.zeros((E, n_bus), np.complex128)  # to-end: I_t = Yt @ V
    Yt[rows, f] += ytf
    Yt[rows, to] += ytt
    Y = np.zeros((n_bus, n_bus), np.complex128)
    np.add.at(Y, (f, f), yff)
    np.add.at(Y, (f, to), yft)
    np.add.at(Y, (to, f), ytf)
    np.add.at(Y, (to, to), ytt)
    if bus_shunt_g is not None or bus_shunt_b is not None:
        gs = np.zeros(n_bus) if bus_shunt_g is None else np.asarray(bus_shunt_g, np.float64)
        bs = np.zeros(n_bus) if bus_shunt_b is None else np.asarray(bus_shunt_b, np.float64)
        d = np.arange(n_bus)
        Y[d, d] += (gs + 1j * bs) / base_mva
    return Admittances(Y, Yf, Yt)


def _dense(Y: Admittance) -> np.ndarray:
    """An admittance matrix as a dense array; a scipy sparse matrix is expanded, a dense one kept."""
    todense = getattr(Y, "todense", None)
    return np.asarray(todense() if todense is not None else Y)


def bus_injections(V: np.ndarray, Ybus: Admittance, base_mva: float = 1.0) -> np.ndarray:
    """Complex bus injections S = V ∘ conj(Ybus V) [AE04, eq. 2.6], generation positive.

    V       : [N] or [T, N] complex bus voltages
    Ybus    : [N, N] nodal admittance (dense or scipy sparse)
    returns : same leading shape as V, complex; times base_mva for MW and MVAr
    """
    if V.ndim == 1:
        return V * np.conj(Ybus @ V) * base_mva
    return V * np.conj((Ybus @ V.T).T) * base_mva


def branch_flows(V: np.ndarray, Yf: Admittance, from_bus: np.ndarray, base_mva: float = 1.0) -> np.ndarray:
    """Complex from-end branch flows S_f = V_f ∘ conj(Yf V) [AE04, eq. 2.8].

    V        : [N] one voltage vector, or [T, N] a stack of them
    Yf       : [E, N] from-end branch admittance (dense or scipy sparse)
    from_bus : [E] the from bus of every branch, indexing V's bus axis
    returns  : [E] or [T, E] complex; times base_mva for MW and MVAr

    The two evaluation orders below are the ones the generator and the loader used before this
    function existed and are kept so their outputs stay bit-identical (for a scipy sparse Yf,
    `V @ Yf.T` is evaluated by scipy as `(Yf @ V.T).T`, the generator's batched expression).
    """
    if V.ndim == 1:
        return V[from_bus] * np.conj(Yf @ V) * base_mva
    return V[:, from_bus] * np.conj(V @ Yf.T) * base_mva


def branch_currents(V: np.ndarray, Yf: Admittance, Yt: Admittance) -> np.ndarray:
    """The branch-current phasors at both ends of every branch [WU26, eqs. 19-20], the channels a PMU
    reads: I_f = Yf V (leaving the from bus into the branch) and I_t = Yt V (leaving the to bus),
    in the `CURRENT` column order.

    V       : [n] one complex voltage vector (per unit, ppc order), or [T, n] a stack of them
    Yf, Yt  : [E, n] from- and to-end branch admittances (dense or scipy sparse)
    returns : [E, 4] or [T, E, 4] real, per unit on the base current: Re I_f, Im I_f, Re I_t, Im I_t
    """
    If, It = (Yf @ V, Yt @ V) if V.ndim == 1 else ((V @ Yf.T), (V @ Yt.T))
    return np.stack([np.real(If), np.imag(If), np.real(It), np.imag(It)], axis=-1)


def ac_measurement(
    vm: np.ndarray,
    theta: np.ndarray,
    Ybus: Admittance,
    Yf: Admittance,
    from_bus: np.ndarray,
    lut: np.ndarray,
    n_ppc: int,
) -> np.ndarray:
    """The AC measurement function h(x) of the estimator, unmasked, load-positive injections
    [AE04, eqs. 2.6 and 2.8].

        h = [ |V|, −Re S, −Im S, θ, Re S_f, Im S_f ]   with S = V ∘ conj(Y V), S_f = V_f ∘ conj(Y_f V)

    vm, theta : [n, N] voltage magnitude (pu) and angle (rad) at the N pandapower buses
    Ybus, Yf  : [n_ppc, n_ppc] and [E, n_ppc] admittances in ppc bus order
    from_bus  : [E] from bus of every branch, ppc order
    lut       : [N] the ppc index of every pandapower bus; unmapped ppc buses sit at zero voltage
    n_ppc     : number of ppc buses
    returns   : [n, 4N + 2E] in per unit and radians, the order the estimator masks
    """
    n = vm.shape[0]
    Vc = np.zeros((n, n_ppc), np.complex128)
    Vc[:, lut] = vm * np.exp(1j * theta)
    Sb = bus_injections(Vc, Ybus)  # generation positive; the shard's injections are load positive
    Sf = branch_flows(Vc, Yf, from_bus)
    return np.concatenate([vm, -Sb.real[:, lut], -Sb.imag[:, lut], theta, Sf.real, Sf.imag], axis=1)


def ac_jacobian(
    vm: np.ndarray,
    theta: np.ndarray,
    Ybus: Admittance,
    Yf: Admittance,
    from_bus: np.ndarray,
    lut: np.ndarray,
    n_ppc: int,
) -> np.ndarray:
    """The measurement Jacobian H = ∂h/∂[θ, |V|] of `ac_measurement` at one state, in closed
    form [AE04, ch. 2], written with the complex bus-voltage derivatives of [MP19, dSbus_dV and
    dSbr_dV]:

        ∂S/∂θ   = j diag(V) conj(diag(I) − Y diag(V)),      I = Y V
        ∂S/∂|V| = diag(V) conj(Y diag(V/|V|)) + conj(diag(I)) diag(V/|V|)
        ∂S_f/∂θ   = j (conj(diag(I_f)) C_f diag(V) − diag(V_f) conj(Y_f diag(V))),   I_f = Y_f V
        ∂S_f/∂|V| = diag(V_f) conj(Y_f diag(V/|V|)) + conj(diag(I_f)) C_f diag(V/|V|)

    vm, theta : [N] one state, pandapower bus order
    returns   : [4N + 2E, 2N] rows in the order of `ac_measurement`, columns [θ (all N) | |V| (all N)];
                the estimator drops the slack angle column
    """
    N, E = len(lut), len(from_bus)
    V = np.zeros(n_ppc, np.complex128)
    V[lut] = vm * np.exp(1j * theta)
    Yb, Yff = _dense(Ybus), _dense(Yf)
    Vnorm = np.where(np.abs(V) > 0, V / np.where(np.abs(V) > 0, np.abs(V), 1.0), 0.0)
    I = Yb @ V
    dS_dVa = 1j * (V[:, None] * np.conj(np.diag(I) - Yb * V[None, :]))
    dS_dVm = V[:, None] * np.conj(Yb * Vnorm[None, :]) + np.conj(I)[:, None] * np.diag(Vnorm)
    If = Yff @ V
    Cf = np.zeros((E, n_ppc))
    Cf[np.arange(E), from_bus] = 1.0
    Vf = V[from_bus]
    dSf_dVa = 1j * (np.conj(If)[:, None] * (Cf * V[None, :]) - Vf[:, None] * np.conj(Yff * V[None, :]))
    dSf_dVm = Vf[:, None] * np.conj(Yff * Vnorm[None, :]) + np.conj(If)[:, None] * (Cf * Vnorm[None, :])
    eye, zero = np.eye(N), np.zeros((N, N))
    rows = [
        np.concatenate([zero, eye], axis=1),  # |V|
        np.concatenate([-dS_dVa.real[np.ix_(lut, lut)], -dS_dVm.real[np.ix_(lut, lut)]], axis=1),  # −P
        np.concatenate([-dS_dVa.imag[np.ix_(lut, lut)], -dS_dVm.imag[np.ix_(lut, lut)]], axis=1),  # −Q
        np.concatenate([eye, zero], axis=1),  # θ
        np.concatenate([dSf_dVa.real[:, lut], dSf_dVm.real[:, lut]], axis=1),  # P_f
        np.concatenate([dSf_dVa.imag[:, lut], dSf_dVm.imag[:, lut]], axis=1),  # Q_f
    ]
    return np.concatenate(rows, axis=0)


def subnetwork(
    edge_index: np.ndarray, seeds: np.ndarray, hops: int, n_bus: int
) -> tuple[np.ndarray, np.ndarray]:
    """The attacker's local subnetwork around `seeds` [WU26]: the interior, every bus within
    `hops` branches of a seed, and its boundary, the buses one branch outside the interior.

    edge_index : [2, E] from and to bus of every branch
    seeds      : the buses the attack is built around (the target line's ends, the attacked loads)
    hops       : how many branches the interior reaches from a seed
    returns    : (interior [k] sorted, boundary [b] sorted), disjoint
    """
    adj: list[set[int]] = [set() for _ in range(n_bus)]
    for a, b in zip(edge_index[0], edge_index[1]):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    interior = {int(b) for b in seeds}
    frontier = set(interior)
    for _ in range(hops):
        frontier = {j for i in frontier for j in adj[i]} - interior
        interior |= frontier
    boundary = {j for i in interior for j in adj[i]} - interior
    return np.array(sorted(interior), int), np.array(sorted(boundary), int)


def _local_mismatch(
    Yb: np.ndarray, V: np.ndarray, I_: np.ndarray, S_target: np.ndarray
) -> tuple[np.ndarray, float]:
    """The interior injection mismatch [Re; Im] of S_target - S(V) and its largest entry."""
    mis = S_target - (V * np.conj(Yb @ V))[I_]
    f = np.concatenate([np.real(mis), np.imag(mis)])
    return f, float(np.max(np.abs(f)))


def _backtrack(
    Yb: np.ndarray, V: np.ndarray, I_: np.ndarray, S_target: np.ndarray, step: np.ndarray, norm: float
) -> Optional[np.ndarray]:
    """The Newton step, halved until it lowers the mismatch and keeps every magnitude positive;
    the full step is tried first, so an iteration the plain method accepts is unchanged. None when
    no fraction of the step helps (the target has no solution near this state)."""
    k = len(I_)
    vm0, va0 = np.abs(V[I_]), np.angle(V[I_])
    for _ in range(_BACKTRACK_HALVINGS):
        vm, va = vm0 + step[k:], va0 + step[:k]
        if np.all(vm > 0):
            Vtry = V.copy()
            Vtry[I_] = vm * np.exp(1j * va)
            if _local_mismatch(Yb, Vtry, I_, S_target)[1] < norm:
                return Vtry
        step = step / 2
    return None


_BACKTRACK_HALVINGS = 8  # step fractions tried per Newton iteration: 1, 1/2, ... 1/128


def _interior_jacobian(Yb: np.ndarray, V: np.ndarray, I_: np.ndarray) -> np.ndarray:
    """The [2k, 2k] Jacobian of the interior injections in [θ_I, |V|_I]: the entries of the full
    ac_jacobian derivatives at (i, j) in the interior depend only on Y_ij, V_i, V_j and the current
    into i, so the block is built from Y_II directly instead of the n x n matrices sliced."""
    Y_II = Yb[np.ix_(I_, I_)]
    V_I = V[I_]
    I_I = (Yb @ V)[I_]
    Vn_I = V_I / np.abs(V_I)
    A = 1j * (V_I[:, None] * np.conj(np.diag(I_I) - Y_II * V_I[None, :]))
    B = V_I[:, None] * np.conj(Y_II * Vn_I[None, :]) + np.conj(I_I)[:, None] * np.diag(Vn_I)
    return np.block([[np.real(A), np.real(B)], [np.imag(A), np.imag(B)]])


def local_ac_solve(
    Ybus: Admittance,
    V: np.ndarray,
    interior: np.ndarray,
    S_target: np.ndarray,
    iters: int = 50,
    tol: float = 1e-9,
) -> Optional[np.ndarray]:
    """The false state of a local attacker [WU26]: the interior bus voltages that give the target
    injections there, with every other bus voltage held at its true value.

        S_i(V) = V_i conj(Σ_j Y_ij V_j) = S_target_i   for i in the interior, V_j fixed elsewhere

    solved by Newton on [θ_I, |V|_I] with the closed-form injection derivatives of `ac_jacobian`,
    each step backtracked (halved) until the mismatch drops [AE04, ch. 2], since a load step of
    gigawatts inside a fixed-boundary region (IEEE-145) throws the plain step past a solution the
    damped one still reaches. Meters that depend only on the fixed voltages keep their true value,
    so the attack touches exactly the interior's and the boundary's meters and is consistent with
    a full AC state.

    Ybus     : [n, n] nodal admittance (dense or scipy sparse), per unit
    V        : [n] true complex bus voltages
    interior : the buses whose voltages may change
    S_target : [len(interior)] target complex injections at those buses, per unit, generation positive
    returns  : [n] the false voltages, or None when no step lowers the mismatch or `iters` run out
    """
    Yb = _dense(Ybus)
    V = np.array(V, np.complex128, copy=True)
    I_ = np.asarray(interior, int)
    for _ in range(iters):
        f, norm = _local_mismatch(Yb, V, I_, S_target)
        if norm < tol:
            return V
        J = _interior_jacobian(Yb, V, I_)
        try:
            step = np.linalg.solve(J, f)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(step)):
            return None
        Vnext = _backtrack(Yb, V, I_, S_target, step, norm)
        if Vnext is None:
            return None
        V = Vnext
    return None


def _flow_residual(
    Yb: np.ndarray,
    yf: np.ndarray,
    f: int,
    V: np.ndarray,
    fixed: np.ndarray,
    S_fixed: np.ndarray,
    target: float,
) -> np.ndarray:
    """[Re; Im] of S_fixed - S(V) at the fixed buses, then |S_f(V)| - target on the goal branch."""
    mis = S_fixed - (V * np.conj(Yb @ V))[fixed]
    flow = abs(V[f] * np.conj(yf @ V))
    return np.concatenate([np.real(mis), np.imag(mis), [flow - target]])


def _flow_jacobian(
    Yb: np.ndarray, yf: np.ndarray, f: int, V: np.ndarray, I_: np.ndarray, fixed_rows: np.ndarray
) -> np.ndarray:
    """The Jacobian of `_flow_residual` in [θ_I, |V|_I]: the fixed buses' injection rows (negated,
    the residual is target minus injection) and the goal branch's apparent-power row, from the
    from-end flow S_f = V_f conj(y_f . V):

        dS_f/dθ_j   = 1j S_f [j = f] - 1j V_f conj(y_fj V_j)
        dS_f/d|V|_j = (V_f/|V_f|) conj(y_f . V) [j = f] + V_f conj(y_fj V_j / |V_j|)
        d|S_f|      = (P_f dP_f + Q_f dQ_f) / |S_f|
    """
    k = len(I_)
    Jinj = _interior_jacobian(Yb, V, I_)
    rows = np.concatenate([fixed_rows, k + fixed_rows])
    Iff = yf @ V
    Sf = V[f] * np.conj(Iff)
    VI = V[I_]
    dth = -1j * V[f] * np.conj(yf[I_] * VI)
    dvm = V[f] * np.conj(yf[I_] * VI / np.abs(VI))
    at_f = np.flatnonzero(I_ == f)
    if len(at_f):
        dth[at_f] += 1j * Sf
        dvm[at_f] += (V[f] / abs(V[f])) * np.conj(Iff)
    mag = max(abs(Sf), 1e-12)
    dmag = np.concatenate([np.real(np.conj(Sf) * dth), np.real(np.conj(Sf) * dvm)]) / mag
    return np.vstack([-Jinj[rows], dmag[None, :]])


def local_flow_solve(
    Ybus: Admittance,
    yf_row: np.ndarray,
    from_bus: int,
    V: np.ndarray,
    interior: np.ndarray,
    fixed: np.ndarray,
    S_fixed: np.ndarray,
    target: float,
    iters: int = 50,
    tol: float = 1e-9,
) -> Optional[np.ndarray]:
    """The false state of a local attacker who drives one branch's apparent flow to `target` [WU26,
    eqs. 24-25]: the interior voltages move, the buses of `fixed` keep their injections, the others of
    the interior (the loads the attacker pretends) are free, and every bus outside the interior keeps
    its true voltage.

        S_i(V) = S_fixed_i   for i in `fixed`,      |V_f conj(y_f . V)| = target

    With fewer equations than the 2k unknowns [θ_I, |V|_I], the solve takes the least-norm
    Gauss-Newton step (the smallest voltage change that removes the residual to first order, numpy's
    minimum-norm least squares), each step halved until the residual drops, as `local_ac_solve`
    backtracks. From the true state this reaches the solution nearest to it.

    Ybus     : [n, n] nodal admittance, per unit
    yf_row   : [n] the goal branch's row of Yf (from-end current), per unit
    from_bus : the goal branch's from-end bus (the end its flow is metered at)
    V        : [n] true complex bus voltages
    interior : the buses whose voltages may change
    fixed    : the interior buses whose injection is held (a subset of `interior`)
    S_fixed  : [len(fixed)] their injections, per unit, generation positive
    target   : the goal branch's apparent flow, per unit
    returns  : [n] the false voltages, or None when no step lowers the residual or `iters` run out
    """
    Yb = _dense(Ybus)
    yf = np.asarray(_dense(yf_row)).ravel()
    V = np.array(V, np.complex128, copy=True)
    I_ = np.asarray(interior, int)
    fx = np.asarray(fixed, int)
    fixed_rows = np.array([int(np.flatnonzero(I_ == b)[0]) for b in fx], int)
    for _ in range(iters):
        r = _flow_residual(Yb, yf, from_bus, V, fx, S_fixed, target)
        norm = float(np.max(np.abs(r)))
        if norm < tol:
            return V
        J = _flow_jacobian(Yb, yf, from_bus, V, I_, fixed_rows)
        step = np.linalg.lstsq(J, -r, rcond=None)[0]  # the minimum-norm solution of J step = -r
        if not np.all(np.isfinite(step)):
            return None
        residual = partial(_flow_residual, Yb, yf, from_bus, fixed=fx, S_fixed=S_fixed, target=target)
        Vnext = _flow_backtrack(residual, V, I_, step, norm)
        if Vnext is None:
            return None
        V = Vnext
    return None


def _flow_backtrack(
    residual: Callable[[np.ndarray], np.ndarray], V: np.ndarray, I_: np.ndarray, step: np.ndarray, norm: float
) -> Optional[np.ndarray]:
    """The Gauss-Newton step on [θ_I, |V|_I], halved until the residual's largest entry drops below
    `norm` with every magnitude positive; None when no fraction of it helps."""
    k = len(I_)
    vm0, va0 = np.abs(V[I_]), np.angle(V[I_])
    for _ in range(_BACKTRACK_HALVINGS):
        vm, va = vm0 + step[k:], va0 + step[:k]
        if np.all(vm > 0):
            Vtry = V.copy()
            Vtry[I_] = vm * np.exp(1j * va)
            if np.max(np.abs(residual(Vtry))) < norm:
                return Vtry
        step = step / 2
    return None
