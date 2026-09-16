"""The AC network model: branch admittances, bus injections and branch flows [AE04, ch. 2], [MP19].

This is the one implementation of the measurement function h(x) that the generator, the loader
and the state estimator share. The estimator's torch twin (fdia_graph.se.base.SEBase._h_t) exists only
because the Jacobian is taken by automatic differentiation; tests/test_formulas.py pins it to the
functions here.

Every function keeps the exact floating-point expression the callers used before it existed, so
released shards and streams reproduce bit for bit: `branch_flows` evaluates V_f * conj(Yf @ V)
for one voltage vector and V[:, f] * conj(V @ Yf.T) for a stack, which are the two orders the
generator and the loader always used.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional, Tuple

import numpy as np


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


class BranchModel(NamedTuple):
    """The per-branch pi model [MP19], one entry per branch, per unit; the shapes a shard stores."""

    r: np.ndarray  # series resistance
    x: np.ndarray  # series reactance
    b: np.ndarray  # charging susceptance
    g: np.ndarray  # charging conductance (transformer iron losses)
    tap: np.ndarray  # turns ratio, 1 (or 0, read as 1) for lines
    shift_deg: np.ndarray  # phase shift in degrees
    status: Optional[np.ndarray] = None  # 1 in service, 0 out; None = all in service


def branch_admittances(
    branch: BranchModel,
    edge_index: np.ndarray,
    n_bus: int,
    bus_shunt_g: Optional[np.ndarray] = None,
    bus_shunt_b: Optional[np.ndarray] = None,
    base_mva: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ybus [N, N], Yf [E, N] and Yt [E, N] from the per-branch pi model [MP19, makeYbus].

    Series admittance y_s, charging admittance (g + j b) split half per end, tap ratio and phase
    shift on the from side, in-service status; bus shunts (MW and MVAr at 1 pu) on the Ybus
    diagonal as (g_sh + j b_sh) / base_mva. With a complex bus voltage vector V, `V[f] * conj(Yf @ V)`
    is the from-end branch flow and `V * conj(Ybus @ V)` the bus injection, both per unit.

        y_tt = y_s + (g + j b) / 2
        y_ff = y_tt / (tap * conj(tap)),  y_ft = -y_s / conj(tap),  y_tf = -y_s / tap

    branch     : the per-branch physics, a BranchModel
    edge_index : [2, E] from-bus and to-bus of every branch, in the bus order of the outputs
    returns    : (Ybus, Yf, Yt), complex128
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
    return Y, Yf, Yt


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
