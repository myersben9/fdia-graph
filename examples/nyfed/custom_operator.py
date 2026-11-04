#!/usr/bin/env python
"""An operator of your own: one that ignores what it reads and never redispatches.

An operator is anything with `respond(day, *, calib=None, alarm_at=None) -> Response` (the `Operator` protocol): it
runs on an attacked day and returns its run on the attacked scans and on their benign twin, each a `Twin` of per-scan
arrays. `summarize` then reports the attack's consequence for it exactly as for the package's `NyisoOperator`. This
do-nothing operator is the baseline every dispatch is compared with.

The example builds its `Response` on a synthetic day, so it needs no data. On a real day, `operator_day(path, pool,
data)` gives the `OperatorDay` and the operator reads `day.time`, `day.unit_p`, `day.interfaces` and the scans.

Run: python examples/nyfed/custom_operator.py
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from fdia_graph.nyfed.operator import NyisoOperator, Operator, OperatorPolicy, Response, Twin, summarize


def idle_twin(T: int, units: int, buses: int, interfaces: int) -> Twin:
    """A run that moved nothing: zero dispatch, regulation and shed, every interface at its basepoint."""
    shapes = {
        "dev": (T, units),
        "reg": (T, units),
        "unrelieved": (T, interfaces),
        "iface_change": (T, interfaces),
        "congestion": (T, interfaces),
        "energy": (T,),
        "shed": (T, buses),
    }
    arrays = {f.name: np.zeros(shapes.get(f.name, (T,))) for f in fields(Twin)}
    arrays["margin_at"] = np.full(T, -1)
    arrays["pf_failed"] = np.zeros(T, bool)
    return Twin(**arrays)


class DoesNothing:
    """Never acts on what it reads, so the attack costs it nothing and the neighbours supply nothing."""

    def __init__(self, units: int, buses: int, interfaces: int) -> None:
        self.units, self.buses, self.interfaces = units, buses, interfaces

    def respond(self, day, *, calib=None, alarm_at=None) -> Response:
        T = len(day.time)
        twin = idle_twin(T, self.units, self.buses, self.interfaces)
        return Response(
            attacked=twin,
            benign=twin,
            unit_client=np.zeros(self.units, int),
            bus_client=np.zeros(self.buses, int),
            client_names=["one"],
            onset=day.onset,
            length=day.length,
            delay=0,
            over=0.0,
            tie_attacked=np.zeros(T),
            tie_benign=np.zeros(T),
            alarm=float("nan"),
            mode="idle",
        )


class _Day:  # the few fields of an OperatorDay this operator reads
    time, onset, length = np.arange(288), 100, 32


def main() -> None:
    idle = DoesNothing(units=3, buses=5, interfaces=2)
    for op in (idle, NyisoOperator(OperatorPolicy(mode="accounting"))):
        assert isinstance(op, Operator)  # both plug in wherever an operator is taken
    s = summarize(idle.respond(_Day()))
    print("dispatch moved:", s["dispatch_moved_mw"], "MW | inadvertent:", s["inadvertent_mwh"], "MWh")


if __name__ == "__main__":
    main()
