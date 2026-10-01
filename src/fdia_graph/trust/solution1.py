"""[WU26]'s Solution 1: the trusted PMUs by row reduction of the measurement matrix (Sec. IV-D1, p. 657),
on the trusted-PMU MDP of one overload window (`trust.WuDefenseEnv`; docs/plans/WU_DEFENSE_PLAN.md,
PR C).

The paper's steps, at each dynamic attack interval: "1) Compute the Jacobian matrix and transform [...]
into a full-row-rank matrix h^T. 2) Perform Elementary Row Transformations [...] into an RREF. 3) [...]
identify the row with the fewest nonzero elements, and perform column exchanges to move all nonzero
elements in this row to the end. A tracking matrix is maintained [...]. 4) Repeat steps 2) and 3) until
the row with the minimal number of nonzero elements remains unchanged [...]. 5) [...] The row vector with
the fewest nonzero elements is multiplied by the inverse of the tracking matrix to derive the
approximate optimal attack vector. The PMUs corresponding to these measurements are protected by dynamic
trusted encryption, which adds them to the secure set h^S. The process is repeated for the remaining
nonsecure subset h^S' until the desired security level is achieved."

Here each configuration step reduces the Jacobian of the attack area's attackable channels at the step's
snapshot, over the buses the PMUs trusted so far leave free (`RrefMixin.area_jacobian`, the same matrix
the row-reduction attack uses), with the column exchanges of `formulas.trust.sparsest_rows`, and trusts
a PMU of the sparsest attack it finds. Ours, where the paper is silent: the chase is restricted to attacks
that move a target line and that a PMU still on offer reads (Solution 1 defends against the overload
attack, and securing a PMU protects nothing an attack does not touch; unrestricted, the sparsest attack
on IEEE-14 sits at bus 13 and moves neither line, and the order came out 13, 4, 6, 1); the case studies
trust one PMU per step, so of the attack's PMUs the one reading the most of its channels is trusted
(ties to the first in the candidate list); and when no open attack touches a PMU on offer, the first PMU
still on offer is trusted. The time `fit` reports is the defender's own (the reductions), not the
search that scores the schedule.
"""

from __future__ import annotations

import functools
import time

import numpy as np

from ..engine.attacks.rref import _moved_lines, _state_change
from ..formulas.trust import sparsest_rows
from .defense import WuDefenseEnv

ZERO = 1e-9  # an attack entry this small relative to the Jacobian's largest is structurally zero


class TrustedPMUs:
    """Solution 1 on `env`: `fit` walks the configuration steps and fills `order` (the actions, indices
    into `WuDefenseConfig.pmus`), `rewards` and `costs` (the environment's, per step) and `seconds`
    (the reductions' wall time)."""

    def __init__(self, env: WuDefenseEnv) -> None:
        self.env = env
        self.order: list[int] = []
        self.rewards: list[float] = []
        self.costs: list[float] = []
        self.seconds = 0.0

    def fit(self) -> TrustedPMUs:
        """Trust one PMU per configuration step, each the one Solution 1 picks, until the steps run out
        or the environment ends the episode."""
        env = self.env
        env.reset()
        self.order, self.rewards, self.costs, self.seconds = [], [], [], 0.0
        for _ in range(len(env.config.slots)):
            start = time.perf_counter()
            action = self.next_action()
            self.seconds += time.perf_counter() - start
            _, reward, done = env.step(action)
            self.order.append(action)
            self.rewards.append(reward)
            self.costs.append(env.cost)
            if done:
                break
        return self

    def next_action(self) -> int:
        """The PMU Solution 1 trusts at the environment's next step."""
        env, g = self.env, self.env.g
        offered = np.flatnonzero(env.valid())
        slots, pmus = env.config.slots, env.config.pmus
        t = int(slots[len(env.trusted)])
        pinned = {int(pmus[i]) for i in env.trusted}
        seeds, _, _ = g._goal_seeds(env.goal)
        area = g.local_region(seeds, env.k.hops)
        free = np.array(
            [int(b) for b in (area if area is not None else []) if int(b) not in pinned], dtype=np.int64
        )
        if not len(free):
            return int(offered[0])
        H, G, devices = g.area_jacobian(env.window, t, free)
        offer = {g.C + int(pmus[i]): int(i) for i in offered}  # a PMU's device id: N + its bus
        tol = ZERO * float(np.abs(H).max()) if H.size else ZERO

        threat = functools.partial(_threatens, H=H, G=G, devices=devices, offer=frozenset(offer), tol=tol)
        rows = sparsest_rows(H.T, tol, eligible=threat)
        if not len(rows):
            return int(offered[0])
        touched = devices[np.abs(rows[0]) > tol]
        reads = {i: int((touched == d).sum()) for d, i in offer.items()}
        return max(reads, key=lambda i: (reads[i], -i))


def _threatens(
    a: np.ndarray, *, H: np.ndarray, G: np.ndarray, devices: np.ndarray, offer: frozenset[int], tol: float
) -> bool:
    """Whether the attack vector a moves a target line (G, through its state change on H) and a PMU on
    offer (device ids `offer`) reads one of its channels (`devices`, one per row of H)."""
    reads = bool(set(devices[np.abs(a) > tol].tolist()) & offer)
    return reads and bool(_moved_lines(G, _state_change(H, a)))
