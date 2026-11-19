#!/usr/bin/env python
"""An attack family of your own: a slow bias on one load bus's injection meter.

A family implements one method, `design(window, rng) -> Episode | None`: given the true states of a window it
returns the attack vector to add to every scan, or None when it finds nothing to attack there. The package's
own families keep the false readings consistent with one AC state; this one does not, which is the naive kind
of attack the residual test is meant to catch. `design_explained` says why a design found nothing.

The example runs on synthetic window states, so it needs no data. In a real build you pass the family to the
source under the name of the family it takes the place of, `SearchSource(net, X, settings, families={"At":
SlowBias(...)})`, and `source.generate(path)` places, designs and writes its episodes like the package's own.
`Episode.family` must be a name in `fdia_graph.FAMILIES`, whose code a file stores. This one uses "Ad", the
meter-corruption family.

Run: python examples/nygrid/custom_attack_family.py
"""

from __future__ import annotations

import numpy as np

from fdia_graph.engine.attacks.family import AttackFamily
from fdia_graph.models.plan import Episode


class SlowBias(AttackFamily):
    """Ramp one injection meter by `rate` (its own units) per scan, on a meter drawn from the design's stream."""

    def __init__(self, meter_bus: np.ndarray, meter_owner: np.ndarray, rate: float = 2.0) -> None:
        self.meter_bus, self.meter_owner, self.rate = meter_bus, meter_owner, rate

    def design(self, window, rng):
        onset, vm, va = window
        if vm.shape[0] < 2:
            self.last_miss = "no_target"  # a hint design_explained reports: too short to ramp anything
            return None
        meter = int(rng.integers(len(self.meter_bus)))
        a = np.zeros((vm.shape[0], len(self.meter_bus)))
        a[:, meter] = self.rate * np.arange(vm.shape[0])  # grows scan by scan
        return Episode(
            family="Ad",  # a name in fdia_graph.FAMILIES: a file stores its code
            clients=np.array([self.meter_owner[meter]]),
            onset=onset,
            length=vm.shape[0],
            buses=np.array([self.meter_bus[meter]]),
            target_branch=-1,  # no target branch: not an overload attack
            a=a,
        )


def main() -> None:
    rng = np.random.default_rng(1)
    fam = SlowBias(meter_bus=np.arange(6), meter_owner=np.array([0, 0, 1, 1, 2, 2]))
    vm, va = np.ones((16, 6)), np.zeros((16, 6))  # a 16-scan window of six buses
    ep, why = fam.design_explained((100, vm, va), rng)
    print("designed:", ep is not None, "| meter moved on bus", ep.buses, "| last scan's bias", ep.a[-1].max())
    _, why = fam.design_explained((100, vm[:1], va[:1]), rng)
    print("a one-scan window:", why)
    # what the design stage of a timeline calls: the window as states [L, N, 4] = |V|, P, Q, angle
    states = np.zeros((16, 6, 4))
    states[:, :, 0] = 1.0
    designed = fam.designed(100, states, rng)
    print("as a timeline asks it: family code", designed.code)


if __name__ == "__main__":
    main()
