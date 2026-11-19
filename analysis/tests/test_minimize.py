"""The analysis toolkit's part of tests/test_minimize.py:
the tests that run an estimator, a localizer, a trust selector or an analysis formula. Of that module:
The fewest-tamper support search [WU26 eq. 12]: exact against brute force on IEEE-14, a lower
bound that never exceeds a cost, the two sigmas (the emitter's per-scan jitter, the meters'
accuracy class the search counts against), and a continuous reweighted-l1 cross-check the search is
never beaten by."""

import numpy as np

from fdia_graph.constants.channels import NodeChannel
from tests.helpers import windows


def _irls_support(g, states, goal, area, iters=20):
    """A support from iteratively reweighted l1 on the linearized local AC equations: the area's
    voltage moves that realize the target buses' load change while penalizing, reweighted, the
    injection change they cause at every other bus (the tamper proxy)."""
    from fdia_analysis.formulas.measurement import ac_jacobian

    Xt = states[0]
    N = g.C
    ends = np.stack([g.case.f, g.case.t], axis=1)  # the two branch columns the derivatives read
    vm, theta = Xt[:, NodeChannel.V], np.radians(Xt[:, NodeChannel.THETA])
    H = ac_jacobian(vm, theta, g.net.ybus, g.net.yf, g.net.yt, ends, np.arange(N))
    P, Q = (
        H[N * NodeChannel.P_INJ : N * (NodeChannel.P_INJ + 1)],
        H[N * NodeChannel.Q_INJ : N * (NodeChannel.Q_INJ + 1)],
    )
    cols = np.r_[area, N + area]  # theta and |V| of the area buses
    design = goal.designs[0]
    targets = np.unique(g.load_bus[design.targets])
    dload = np.zeros(N)
    np.add.at(
        dload,
        g.load_bus[design.targets],
        g.scan_loads.true_load(Xt)[design.targets] * (np.asarray(design.mult) - 1.0),
    )
    A = np.vstack([P[targets][:, cols], Q[targets][:, cols]])
    rhs = np.r_[dload[targets], np.zeros(len(targets))] / g.case.base_mva
    others = [b for b in range(N) if b not in set(targets.tolist())]
    w = np.ones(len(others))
    x = np.zeros(len(cols))
    for _ in range(iters):
        G = sum(
            wi * (np.outer(P[b, cols], P[b, cols]) + np.outer(Q[b, cols], Q[b, cols]))
            for wi, b in zip(w, others)
        )
        G = G + 1e-9 * np.eye(len(cols))
        K = np.block([[G, A.T], [A, np.zeros((len(A), len(A)))]])
        x = np.linalg.lstsq(K, np.r_[np.zeros(len(cols)), rhs], rcond=None)[0][: len(cols)]
        change = np.array([np.hypot(P[b, cols] @ x, Q[b, cols] @ x) for b in others])
        w = 1.0 / (change + 1e-6)
    move = np.abs(x[: len(area)]) + np.abs(x[len(area) :])
    S = np.union1d(area[move > 5e-2 * move.max()], targets)
    S, _ = g.area.grow_over_zero_injection(S)  # the region's rule, as every candidate follows it
    return S


def test_reweighted_l1_never_beats_the_search(case):
    """The continuous relaxation proposes a support; under the same constraints it never tampers fewer
    devices than exhaustive search over the area. The relaxation cannot see the per-snapshot stealth
    bound, so both sides are costed without it."""
    from fdia_graph.engine.attacks.minimize import _Window

    g, _, k = case
    compared = 0
    for states, goal, d in windows(case, 3, held=False):
        buses = np.unique(g.load_bus[d.targets])
        area = g.area.local_region(buses, k.hops)
        window = _Window(g, states, goal, k, stealth_bound=False)
        costs = [
            c
            for S in g.minimizer._supports([frozenset(int(b) for b in buses)], area)
            if (c := window.cost(S, None)) is not None
        ]
        S = _irls_support(g, states, goal, np.asarray(area))
        cost = window.cost(S, None)
        if costs and cost is not None:
            compared += 1
            assert cost >= min(costs)
    assert compared >= 1, "the reweighted-l1 support should be feasible on at least one window"
