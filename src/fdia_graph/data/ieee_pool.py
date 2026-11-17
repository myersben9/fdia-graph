"""The pool of an IEEE case: one AC operating state per step of a load profile.

    generate_states(system, S)  ->  the pool of states [T, N, 4]

The pool's columns are [|V| (pu), P_inj (MW), Q_inj (MVAr), theta (deg)], in pandapower's
consumer-positive convention. `S` is a load-scaling vector of `data/profiles.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from fdia_graph.constants.grid import (
    IEEE_CASES,  # bus-count -> pandapower builder; single source of supported systems
)
from fdia_graph.constants.time import LOAD_JITTER_RHO
from fdia_graph.models.inputs import SupportedSystem

if TYPE_CHECKING:
    from fdia_graph.systems.pandapower_types import PandapowerNet

# per-bus scale at timestep t: clip(1 + K*S_t + jitter_t, *CLIP). K is the profile's weight, SIGMA the
# jitter's std, and the clip holds a plausible load band.
K_DEFAULT = 0.1
SIGMA_DEFAULT = 0.03
CLIP_DEFAULT = (0.7, 1.3)


def _case_buses(key: int) -> np.ndarray:
    """The buses with a load or a generator, sorted: the column order of a scale-factor row. The
    worker rebuilds the same order."""
    import pandapower.networks as pn

    base = getattr(pn, IEEE_CASES[key])()
    return np.unique(np.concatenate([base.load["bus"].to_numpy(), base.gen["bus"].to_numpy()]))


def _solve_states_chunk(key: int, sf_chunk: np.ndarray) -> list[np.ndarray]:
    """The AC operating state [N, 4] of each scale-factor row of `sf_chunk` [L, nbus] (columns in
    `_case_buses` order). A row whose power flow does not converge is skipped. Module-level, so it
    can run in a worker process."""
    import pandapower as pp
    import pandapower.networks as pn

    base = getattr(pn, IEEE_CASES[key])()
    pp.runpp(base)  # seed a valid base state
    nodelist = sorted(base.bus.index)
    base_load_p = base.load["p_mw"].to_numpy().copy()
    base_load_q = base.load["q_mvar"].to_numpy().copy()
    base_gen_p = base.gen["p_mw"].to_numpy().copy()
    load_buses = base.load["bus"].to_numpy()
    gen_buses = base.gen["bus"].to_numpy()
    all_buses = np.unique(np.concatenate([load_buses, gen_buses]))
    # each load's and generator's column in a scale-factor row (all_buses is sorted and unique)
    at_load, at_gen = np.searchsorted(all_buses, load_buses), np.searchsorted(all_buses, gen_buses)
    pos = {int(b): i for i, b in enumerate(nodelist)}
    out = []
    for sf in sf_chunk:
        sf = np.asarray(sf, np.float64)
        base.load["p_mw"] = base_load_p * sf[at_load]
        base.load["q_mvar"] = base_load_q * sf[at_load]
        base.gen["p_mw"] = base_gen_p * sf[at_gen]
        try:
            pp.runpp(base, init="flat", max_iteration=50, tolerance_mva=1e-6)
        except Exception:
            continue  # no power-flow solution: skip this timestep
        # [N, 4] = [|V|, Pinj, Qinj, theta], the column order the pool, the engine and node_x share
        z = base.res_bus.reindex(nodelist)[["vm_pu", "p_mw", "q_mvar", "va_degree"]].to_numpy().copy()
        _remove_shunt_injections(z, base, pos)
        out.append(z)
    return out


def _remove_shunt_injections(z: np.ndarray, base: PandapowerNet, pos: dict[int, int]) -> None:
    """Subtract each shunt's draw from its bus's P and Q, in place. State estimation models a shunt in
    the admittance matrix, not as an injection, so the stored injection must exclude it."""
    if not len(base.res_shunt):
        return
    keep = np.array([int(b) in pos for b in base.shunt.bus.to_numpy()], bool)
    rows = np.array([pos[int(b)] for b in base.shunt.bus.to_numpy()[keep]], int)
    # unbuffered: two shunts on one bus both subtract
    np.subtract.at(z[:, 1], rows, base.res_shunt.p_mw.to_numpy()[keep])
    np.subtract.at(z[:, 2], rows, base.res_shunt.q_mvar.to_numpy()[keep])


def _ar1_scale(
    S: np.ndarray, nbus: int, k: float, sigma: float, clip: tuple[float, float], rho: float, seed: int
) -> np.ndarray:
    """The per-bus scale factors [T, nbus]: clip(1 + k*S_t + jitter_t), with a per-bus AR(1) jitter
    jitter_t = rho*jitter_{t-1} + sqrt(1-rho^2)*sigma*eps. The AR(1) makes each load drift smoothly
    (~0.5%/min at rho=0.98, against ~4% for independent jitter) while its stationary std stays sigma."""
    T = len(S)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((T, nbus))
    jit = np.empty((T, nbus))
    jit[0] = sigma * eps[0]
    step = sigma * np.sqrt(1.0 - rho * rho)
    for t in range(1, T):
        jit[t] = rho * jit[t - 1] + step * eps[t]
    return np.clip(1.0 + k * S[:, None] + jit, clip[0], clip[1])


def generate_states(
    system: Union[int, str],
    profile: Union[np.ndarray, Sequence[float]],
    k: float = K_DEFAULT,
    sigma: float = SIGMA_DEFAULT,
    clip: tuple[float, float] = CLIP_DEFAULT,
    n: Optional[int] = None,
    seed: int = 123,
    workers: Optional[int] = None,
) -> np.ndarray:
    """A pool of AC states [T', N, 4] = [|V| (pu), P_inj (MW), Q_inj (MVAr), theta (deg)] from a profile of
    T steps: each timestep scales the loads and generator setpoints per bus (`_ar1_scale`) and solves the AC
    power flow. One that does not converge is dropped, so T' <= T and the rows no longer align with the
    profile. `workers` > 1 needs an `if __name__ == "__main__":` guard."""
    key = SupportedSystem(system, frozenset(IEEE_CASES)).number
    S = np.asarray(profile, dtype=float).ravel()
    if n is not None:
        S = S[:n]
    # the whole scale-factor matrix is built here, so the AR(1) jitter is continuous across chunks
    nbus = len(_case_buses(key))
    SF = _ar1_scale(S, nbus, k, sigma, clip, LOAD_JITTER_RHO, seed)  # [T, nbus]
    if workers and workers > 1 and len(S) >= workers:
        import multiprocessing as mp

        sf_chunks = np.array_split(SF, workers)  # contiguous rows: concatenating keeps time order
        args = [(key, sf_chunks[i]) for i in range(len(sf_chunks))]
        with mp.Pool(workers) as pool:
            parts = pool.starmap(_solve_states_chunk, args)
        out = [z for part in parts for z in part]
    else:
        out = _solve_states_chunk(key, SF)
    if not out:
        raise RuntimeError("no operating points converged; check the profile / scaling knobs")
    return np.asarray(out, dtype=np.float64)
