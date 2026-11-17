"""The estimator's AC measurement function h(x) and its closed-form Jacobian [AE04, ch. 2], [MP19],
assembled from the network model the generator and the loader use (`fdia_graph.formulas.network`:
`bus_injections`, `branch_flows`). The estimator's torch twin (`se.base.SEBase._h_t`) is kept for
callers that differentiate through it; analysis/tests/test_formulas.py pins it to the functions here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fdia_graph.formulas.network import branch_flows, bus_injections

if TYPE_CHECKING:
    from fdia_graph.formulas.network import Admittance


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
    Yt: Admittance,
    branch: np.ndarray,
    lut: np.ndarray,
) -> np.ndarray:
    """The measurement Jacobian H = ∂h/∂[θ, |V|] of `ac_measurement` at one state, from MATPOWER's
    complex derivatives [MP19, dSbus_dV and dSbr_dV] as pandapower ships them [AE04, ch. 2]:

        ∂S/∂θ   = j diag(V) conj(diag(I) − Y diag(V)),      I = Y V
        ∂S/∂|V| = diag(V) conj(Y diag(V/|V|)) + conj(diag(I)) diag(V/|V|)
        ∂S_f/∂θ   = j (conj(diag(I_f)) C_f diag(V) − diag(V_f) conj(Y_f diag(V))),   I_f = Y_f V
        ∂S_f/∂|V| = diag(V_f) conj(Y_f diag(V/|V|)) + conj(diag(I_f)) C_f diag(V/|V|)

    vm, theta     : [N] one state, pandapower bus order
    Ybus, Yf, Yt  : admittances in ppc bus order (makeYbus), sparse or dense
    branch        : the ppc branch matrix (its from and to bus columns)
    lut           : [N] the ppc index of every pandapower bus; unmapped ppc buses sit at zero voltage
    returns       : [4N + 2E, 2N] rows in the order of `ac_measurement`, columns [θ (all N) | |V| (all N)];
                    the estimator drops the slack angle column

    Both derivatives are generation positive; the estimator's injections are load positive, so
    the P and Q rows are negated. A bus at zero voltage has an undefined V/|V|; its columns are
    never kept, so the invalid-value warning is silenced.
    """
    from pandapower.pypower.dSbr_dV import dSbr_dV  # the [se] extra
    from pandapower.pypower.dSbus_dV import dSbus_dV
    from scipy.sparse import csr_matrix

    N = len(lut)
    V = np.zeros(np.shape(Ybus)[0], np.complex128)
    V[lut] = vm * np.exp(1j * theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        # dSbus_dV returns (∂/∂|V|, ∂/∂θ) and dSbr_dV the from end as (∂/∂θ, ∂/∂|V|)
        dVm, dVa = dSbus_dV(csr_matrix(Ybus), V)
        dfa, dfm = dSbr_dV(branch, csr_matrix(Yf), csr_matrix(Yt), V)[:2]
    dS_dVm, dS_dVa, dSf_dVa, dSf_dVm = (np.asarray(d.toarray(), np.complex128) for d in (dVm, dVa, dfa, dfm))
    ix = np.ix_(lut, lut)
    eye, zero = np.eye(N), np.zeros((N, N))
    rows = [
        np.concatenate([zero, eye], axis=1),  # |V|
        np.concatenate([-dS_dVa.real[ix], -dS_dVm.real[ix]], axis=1),  # −P
        np.concatenate([-dS_dVa.imag[ix], -dS_dVm.imag[ix]], axis=1),  # −Q
        np.concatenate([eye, zero], axis=1),  # θ
        np.concatenate([dSf_dVa.real[:, lut], dSf_dVm.real[:, lut]], axis=1),  # P_f
        np.concatenate([dSf_dVa.imag[:, lut], dSf_dVm.imag[:, lut]], axis=1),  # Q_f
    ]
    return np.concatenate(rows, axis=0)
