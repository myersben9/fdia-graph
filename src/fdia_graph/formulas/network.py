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
from typing import TYPE_CHECKING, NamedTuple, Optional, Union

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
    YV: np.ndarray, V: np.ndarray, I_: np.ndarray, S_target: np.ndarray
) -> tuple[np.ndarray, float]:
    """The interior injection mismatch [Re; Im] of S_target - S(V) and its largest entry, from the
    bus currents YV = Ybus V of the same V."""
    mis = S_target - (V * np.conj(YV))[I_]
    f = np.concatenate([np.real(mis), np.imag(mis)])
    return f, float(np.max(np.abs(f)))


def _backtrack(
    Yb: np.ndarray, V: np.ndarray, I_: np.ndarray, S_target: np.ndarray, step: np.ndarray, norm: float
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """The Newton step, halved until it lowers the mismatch and keeps every magnitude positive;
    the full step is tried first, so an iteration the plain method accepts is unchanged. Returns the
    accepted voltages and their bus currents Ybus V (the next iteration's mismatch and Jacobian reuse
    them), or None when no fraction of the step helps (the target has no solution near this state)."""
    k = len(I_)
    vm0, va0 = np.abs(V[I_]), np.angle(V[I_])
    for _ in range(_BACKTRACK_HALVINGS):
        vm, va = vm0 + step[k:], va0 + step[:k]
        if np.all(vm > 0):
            Vtry = V.copy()
            Vtry[I_] = vm * np.exp(1j * va)
            YV = Yb @ Vtry
            if _local_mismatch(YV, Vtry, I_, S_target)[1] < norm:
                return Vtry, YV
        step = step / 2
    return None


_BACKTRACK_HALVINGS = 8  # step fractions tried per Newton iteration: 1, 1/2, ... 1/128


def _as_real_blocks(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """[[Re A, Re B], [Im A, Im B]] filled in place (what np.block builds, without its checks, which
    cost more than the fill for the small blocks of a local solve)."""
    r, k = A.shape
    J = np.empty((2 * r, 2 * k))
    J[:r, :k], J[:r, k:], J[r:, :k], J[r:, k:] = A.real, B.real, A.imag, B.imag
    return J


def _interior_jacobian(Y_II: np.ndarray, V_I: np.ndarray, I_I: np.ndarray) -> np.ndarray:
    """The [2k, 2k] Jacobian of the interior injections in [θ_I, |V|_I]: the entries of the full
    ac_jacobian derivatives at (i, j) in the interior depend only on Y_ij, V_i, V_j and the current
    I_i into i, so the block is built from the interior block Y_II of Ybus, the interior voltages
    V_I and currents I_I directly instead of the n x n matrices sliced."""
    Vn_I = V_I / np.abs(V_I)
    A = 1j * (V_I[:, None] * np.conj(np.diag(I_I) - Y_II * V_I[None, :]))
    B = V_I[:, None] * np.conj(Y_II * Vn_I[None, :]) + np.conj(I_I)[:, None] * np.diag(Vn_I)
    return _as_real_blocks(A, B)


def _injection_jacobian(
    Y_RI: np.ndarray, V_R: np.ndarray, V_I: np.ndarray, I_R: np.ndarray, D: np.ndarray
) -> np.ndarray:
    """The [2r, 2k] Jacobian of the injections at the buses R (any buses: inside the interior or on
    its edge) in the interior's [θ_I, |V|_I], from S_r = V_r conj(sum_j Y_rj V_j):

        dS_r/dθ_j   = 1j V_r conj(I_r) [r = j] - 1j V_r conj(Y_rj V_j)
        dS_r/d|V|_j = V_r conj(Y_rj V_j / |V_j|) + conj(I_r) V_r / |V_r| [r = j]

    Y_RI : [r, k] the rows of R of Ybus on the interior's columns
    V_R, V_I : the voltages of R and of the interior
    I_R  : [r] the bus currents into R, (Ybus V)_R
    D    : [r, k] 1 where the bus of R is the interior bus, else 0
    """
    A = 1j * V_R[:, None] * (np.conj(I_R)[:, None] * D - np.conj(Y_RI * V_I[None, :]))
    B = (
        V_R[:, None] * np.conj(Y_RI * (V_I / np.abs(V_I))[None, :])
        + (np.conj(I_R) * V_R / np.abs(V_R))[:, None] * D
    )
    return _as_real_blocks(A, B)


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

    Ybus     : [n, n] nodal admittance (dense or scipy sparse), per unit; pass it dense when solving
               often (a sparse one is expanded on every call)
    V        : [n] true complex bus voltages
    interior : the buses whose voltages may change
    S_target : [len(interior)] target complex injections at those buses, per unit, generation positive
    returns  : [n] the false voltages, or None when no step lowers the mismatch or `iters` run out

    The products stay the full dense Ybus V of the released generator, evaluated once per accepted
    voltage vector and reused by the next mismatch and Jacobian, so the frozen reference timeline's
    stealthy frames reproduce bit for bit (a product over the interior rows alone rounds differently
    in the last bit for a few row sets).
    """
    Yb = _dense(Ybus)
    V = np.array(V, np.complex128, copy=True)
    I_ = np.asarray(interior, int)
    YV = Yb @ V
    for _ in range(iters):
        f, norm = _local_mismatch(YV, V, I_, S_target)
        if norm < tol:
            return V
        J = _interior_jacobian(Yb[np.ix_(I_, I_)], V[I_], YV[I_])
        try:
            step = np.linalg.solve(J, f)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(step)):
            return None
        accepted = _backtrack(Yb, V, I_, S_target, step, norm)
        if accepted is None:
            return None
        V, YV = accepted
    return None


def _row_block(Y: Admittance, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rows of an admittance matrix cut down to the columns where they hold an admittance: (the
    dense block [len(rows), m], those m columns), so that Y[rows] @ V = block @ V[columns]. A branch
    or bus touches a few buses, so a product over a few rows costs their size, not the grid's, and
    stays small enough that BLAS runs it on one thread (a product of a full matrix starts threads
    whose start-up, hundreds of microseconds on a many-core machine, outweighs the arithmetic). The
    sum runs over fewer terms than the full product, so it can differ from it in the last bit: use it
    where a reading is only compared with its noise (the search's attack vector), not inside a solve.

    Y       : [n, n] or [E, n] admittance, dense or scipy sparse
    rows    : the row indices kept
    returns : (block [len(rows), m] complex, columns [m] sorted)
    """
    sub = np.asarray(_dense(Y[rows]))
    cols = np.flatnonzero(np.any(sub != 0, axis=0))
    return sub[:, cols], cols


class _FlowBlocks(NamedTuple):
    """The constant pieces of a local flow solve on one interior, cut once (`_flow_blocks`): the dense
    Ybus, the rows of the buses whose injection a solve may hold (`rows`: the interior and any edge
    bus) on the interior's columns (the Jacobian's), and the goal branches' dense from-end rows of Yf
    and their interior columns. Slices only: every product is still the full one of the released
    solve, so its answers are unchanged to the last bit."""

    Yb: np.ndarray
    rows: np.ndarray
    Y_RI: np.ndarray
    yf: np.ndarray
    yf_I: np.ndarray


class _Held(NamedTuple):
    """What one `local_flow_solve` call holds, placed in its `_FlowBlocks`: the fixed buses' rows of
    Ybus on the interior's columns, where each fixed bus is the interior bus (the Jacobian's diagonal
    term), which [P; Q] rows are held, the held magnitudes' positions in the interior, and each goal
    branch's from bus's position there (-1 outside it)."""

    Y_FI: np.ndarray
    D: np.ndarray
    keep: np.ndarray
    vm_cols: np.ndarray
    f_at: np.ndarray


def _flow_blocks(
    Ybus: Admittance, yf_rows: Admittance, interior: np.ndarray, rows: np.ndarray
) -> _FlowBlocks:
    """The `_FlowBlocks` of `local_flow_solve` on this interior for the held buses among `rows` (the
    interior and the edge buses whose injection a solve may hold) and these goal branches' Yf rows
    (`yf_row` there), for a caller that solves the same region many times (the active set of
    `FalseStateMixin.solve_flow_local` re-solves it with bounds pinned)."""
    I_ = np.asarray(interior, int)
    R = np.asarray(rows, int)
    Yb = np.asarray(_dense(Ybus))
    yf = np.atleast_2d(np.asarray(_dense(yf_rows)))
    return _FlowBlocks(Yb, R, Yb[np.ix_(R, I_)], yf, yf[:, I_])


def _held(
    blocks: _FlowBlocks,
    I_: np.ndarray,
    fixed: tuple[np.ndarray, Optional[np.ndarray]],
    vb: np.ndarray,
    f: np.ndarray,
) -> _Held:
    """`_Held` of a solve: `fixed` = (the held buses, which of P and Q each holds or None for both),
    `vb` the buses of the held magnitudes, `f` the goal branches' from buses."""
    fx, hold = fixed
    row = {int(b): i for i, b in enumerate(blocks.rows)}
    position = {int(b): i for i, b in enumerate(I_)}
    keep = (
        np.ones(2 * len(fx), bool) if hold is None else np.concatenate([hold[:, 0], hold[:, 1]]).astype(bool)
    )
    return _Held(
        blocks.Y_RI[[row[int(b)] for b in fx]],
        (fx[:, None] == I_[None, :]).astype(float),
        keep,
        np.array([position[int(b)] for b in vb], int),
        np.array([position.get(int(b), -1) for b in f], int),
    )


def _flow_residual(
    blocks: _FlowBlocks,
    held: _Held,
    goal: tuple[np.ndarray, np.ndarray],
    V: np.ndarray,
    fixed: np.ndarray,
    S_fixed: np.ndarray,
    vm_fixed: tuple[np.ndarray, np.ndarray] = (np.zeros(0, int), np.zeros(0)),
) -> tuple[np.ndarray, np.ndarray]:
    """([Re; Im] of S_fixed - S(V) at the fixed buses (only the rows `held.keep` marks, [2 len(fixed)]
    over [Re; Im]), then |S_f(V)| - target on each goal branch (`goal` = (f [L], target [L]), the
    branches' rows in `blocks`), then |V_b| - value at each held magnitude (`vm_fixed`); the bus
    currents Ybus V, which the Jacobian at the same V reuses)."""
    f, target = goal
    YV = blocks.Yb @ V
    mis = S_fixed - (V * np.conj(YV))[fixed]
    inj = np.concatenate([np.real(mis), np.imag(mis)])
    flow = np.abs(V[f] * np.conj(blocks.yf @ V))
    vb, vm = vm_fixed
    return np.concatenate([inj[held.keep], flow - target, np.abs(V[vb]) - vm]), YV


def _flow_jacobian(
    blocks: _FlowBlocks,
    held: _Held,
    f: np.ndarray,
    V: np.ndarray,
    I_: np.ndarray,
    at: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """The Jacobian of `_flow_residual` in [θ_I, |V|_I] at V: the held injection rows (`at` = (the
    buses `fixed`, inside the interior or on its edge; the bus currents Ybus V at V); the [P; Q] rows
    `held.keep`), negated since the residual is target minus injection), one apparent-power row per
    goal branch, from its from-end flow S_f = V_f conj(y_f . V), and a unit row per held magnitude
    (`held.vm_cols`, positions in the interior):

        dS_f/dθ_j   = 1j S_f [j = f] - 1j V_f conj(y_fj V_j)
        dS_f/d|V|_j = (V_f/|V_f|) conj(y_f . V) [j = f] + V_f conj(y_fj V_j / |V_j|)
        d|S_f|      = (P_f dP_f + Q_f dQ_f) / |S_f|
    """
    fixed, YV = at
    k = len(I_)
    VI = V[I_]
    Jinj = _injection_jacobian(held.Y_FI, V[fixed], VI, YV[fixed], held.D)[held.keep]
    flow_rows = []
    for y, y_I, b, at_f in zip(blocks.yf, blocks.yf_I, f, held.f_at):
        Iff = y @ V
        Sf = V[b] * np.conj(Iff)
        dth = -1j * V[b] * np.conj(y_I * VI)
        dvm = V[b] * np.conj(y_I * VI / np.abs(VI))
        if at_f >= 0:
            dth[at_f] += 1j * Sf
            dvm[at_f] += (V[b] / abs(V[b])) * np.conj(Iff)
        mag = max(abs(Sf), 1e-12)
        flow_rows.append(np.concatenate([np.real(np.conj(Sf) * dth), np.real(np.conj(Sf) * dvm)]) / mag)
    vm_rows = np.zeros((len(held.vm_cols), 2 * k))
    vm_rows[np.arange(len(held.vm_cols)), k + held.vm_cols] = 1.0  # d|V_b| / d|V|_b
    return np.vstack([-Jinj, *flow_rows, vm_rows])


def local_flow_solve(
    Ybus: Admittance,
    yf_row: np.ndarray,
    from_bus: Union[int, np.ndarray],
    V: np.ndarray,
    interior: np.ndarray,
    fixed: np.ndarray,
    S_fixed: np.ndarray,
    target: Union[float, np.ndarray],
    iters: int = 50,
    tol: float = 1e-9,
    vm_fixed: Optional[tuple[np.ndarray, np.ndarray]] = None,
    hold: Optional[np.ndarray] = None,
    blocks: Optional[_FlowBlocks] = None,
) -> Optional[np.ndarray]:
    """The false state of a local attacker who drives one or more branches' apparent flows to their
    targets [WU26, eqs. 24-25]: the interior voltages move, the buses of `fixed` keep their
    injections, the others of the interior (the free injections: the loads the attacker pretends and
    the generators whose output it pretends) are free, and every bus outside the interior keeps its
    true voltage.

        S_i(V) = S_fixed_i   for i in `fixed`,      |V_f conj(y_f . V)| = target_l for each goal l

    With fewer equations than the 2k unknowns [θ_I, |V|_I], the solve takes the least-norm
    Gauss-Newton step (the smallest voltage change that removes the residual to first order, numpy's
    minimum-norm least squares), each step halved until the residual drops, as `local_ac_solve`
    backtracks. From the true state this reaches the solution nearest to it.

    Each voltage vector tried costs one product Ybus V, whose rows the residual and, for an accepted
    vector, the next iteration's Jacobian share. The products are the full ones, not a region's rows:
    a search's answer can turn on the last bit of a solve near its convergence limit, and a product
    over fewer terms rounds differently.

    Ybus     : [n, n] nodal admittance (dense or scipy sparse), per unit
    yf_row   : [n] the goal branch's row of Yf (from-end current), or [L, n] for L goal branches
    from_bus : the goal branch's from-end bus (the end its flow is metered at), or [L]
    V        : [n] true complex bus voltages
    interior : the buses whose voltages may change
    fixed    : the buses whose injection is held: interior buses, or buses on the interior's edge
               (whose voltage is held true, and whose injection the interior's voltages move)
    S_fixed  : [len(fixed)] their injections, per unit, generation positive
    target   : the goal branch's apparent flow, per unit, or [L]
    vm_fixed : (interior buses, magnitudes) held at those voltage magnitudes, or None (a bound the
               caller enforces, e.g. a voltage limit)
    hold     : [len(fixed), 2] booleans, which of P and Q each fixed bus holds (a generator at one
               limit keeps the other component free); None holds both everywhere
    blocks   : `_flow_blocks(Ybus, yf_row, interior, rows)` when the caller already cut them for a
               `rows` holding every bus of `fixed`, else None (cut here)
    returns  : [n] the false voltages, or None when no step lowers the residual or `iters` run out
    """
    f = np.atleast_1d(np.asarray(from_bus, int))
    goal = np.atleast_1d(np.asarray(target, float))
    V = np.array(V, np.complex128, copy=True)
    I_ = np.asarray(interior, int)
    fx = np.asarray(fixed, int)
    vb, vm = (
        (np.zeros(0, int), np.zeros(0)) if vm_fixed is None else (np.asarray(vm_fixed[0], int), vm_fixed[1])
    )
    cut = _flow_blocks(Ybus, yf_row, I_, np.union1d(I_, fx)) if blocks is None else blocks
    held = _held(cut, I_, (fx, hold), vb, f)
    residual = partial(_flow_residual, cut, held, (f, goal), fixed=fx, S_fixed=S_fixed, vm_fixed=(vb, vm))
    r, YV = residual(V)
    for _ in range(iters):
        norm = float(np.max(np.abs(r)))
        if norm < tol:
            return V
        J = _flow_jacobian(cut, held, f, V, I_, (fx, YV))
        step = np.linalg.lstsq(J, -r, rcond=None)[0]  # the minimum-norm solution of J step = -r
        if not np.all(np.isfinite(step)):
            return None
        accepted = _flow_backtrack(residual, V, I_, step, norm)
        if accepted is None:
            return None
        V, (r, YV) = accepted
    return None


def _flow_backtrack(
    residual: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    V: np.ndarray,
    I_: np.ndarray,
    step: np.ndarray,
    norm: float,
) -> Optional[tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]]:
    """The Gauss-Newton step on [θ_I, |V|_I], halved until the residual's largest entry drops below
    `norm` with every magnitude positive: (the voltages, their residual and bus currents, which the
    next iteration starts from), or None when no fraction of it helps."""
    k = len(I_)
    vm0, va0 = np.abs(V[I_]), np.angle(V[I_])
    for _ in range(_BACKTRACK_HALVINGS):
        vm, va = vm0 + step[k:], va0 + step[:k]
        if np.all(vm > 0):
            Vtry = V.copy()
            Vtry[I_] = vm * np.exp(1j * va)
            tried = residual(Vtry)
            if np.max(np.abs(tried[0])) < norm:
                return Vtry, tried
        step = step / 2
    return None
