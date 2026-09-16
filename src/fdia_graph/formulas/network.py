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

from typing import Any, Optional

import numpy as np
from ..models.grid import Admittances, BranchModel  # noqa: F401  re-exported: defined here before the models package


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


def bus_injections(V: np.ndarray, Ybus: Any, base_mva: float = 1.0) -> np.ndarray:
    """Complex bus injections S = V ∘ conj(Ybus V) [AE04, eq. 2.6], generation positive.

    V       : [N] or [T, N] complex bus voltages
    Ybus    : [N, N] nodal admittance (dense or scipy sparse)
    returns : same leading shape as V, complex; times base_mva for MW and MVAr
    """
    if V.ndim == 1:
        return V * np.conj(Ybus @ V) * base_mva
    return V * np.conj((Ybus @ V.T).T) * base_mva


def branch_flows(V: np.ndarray, Yf: Any, from_bus: np.ndarray, base_mva: float = 1.0) -> np.ndarray:
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


def ac_measurement(
    vm: np.ndarray,
    theta: np.ndarray,
    Ybus: Any,
    Yf: Any,
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
    vm: np.ndarray, theta: np.ndarray, Ybus: Any, Yf: Any, from_bus: np.ndarray, lut: np.ndarray, n_ppc: int
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
    Yb = np.asarray(Ybus.todense() if hasattr(Ybus, "todense") else Ybus)
    Yff = np.asarray(Yf.todense() if hasattr(Yf, "todense") else Yf)
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
