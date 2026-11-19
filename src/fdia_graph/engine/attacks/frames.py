"""An attack written onto a scan: the attack vector a = h(x_false) - h(x_true) of a false state, and
the frame that carries it.

`frame_from_state` takes a local false state (the search-based families), `frame_from_vector` an
attack vector over the meters (a family of your own). Either way the frame is the true scan, with
every meter's own noise draw, plus the attack on the meters it moves."""

from __future__ import annotations

import numpy as np

from ...constants.numerics import ZERO_TOL
from ...models.frames import Frame, MeterLayers, Scan
from ..base import Part


class AttackFrames(Part):
    """Write an attack onto the true scan: the generator's `attack_frames`."""

    def frame_from_state(
        self, x_true: np.ndarray, x_false: np.ndarray, buses: np.ndarray, dev: np.ndarray
    ) -> Frame:
        """The emitted frame of a local false state `x_false` on the true state `x_true`: the true scan (the
        benign twin and every meter's noise draw) plus the attack vector a = h(x_false) - h(x_true) on
        the meters it moves, labelled at `buses` with their designed magnitudes `dev`."""
        scan = self.g.emitter.emit_from_state(x_true)
        bnx, bex = scan.node_x, scan.edge_x
        a_node, a_edge = self.attack_vector(x_false, x_true)
        moved = changed_meters(a_node, a_edge, scan)
        nx, ex = bnx.copy(), bex.copy()
        nx[moved[0]] += a_node[moved[0]]
        ex[moved[1]] += a_edge[moved[1]]
        y = np.zeros(self.g.n_bus, np.uint8)
        y[buses] = 1
        ix, itamper = self.currents_with_attack(scan, x_false, x_true)
        # the tamper masks: the meters whose stored float32 reading changed, no fewer, no more
        changed = MeterLayers(nx != bnx, ex != bex, itamper)
        return Frame.from_scan(scan, MeterLayers(nx, ex, ix), y, (buses, dev), changed)

    def frame_from_vector(self, x_true: np.ndarray, a: np.ndarray) -> Frame:
        """The emitted frame of an attack held as a vector over the meters, a [M] = h(x_false) -
        h(x_true): the true scan plus `a` on every meter. The meters it moves beyond their noise
        (`NoiseModel.tampered`) are the tamper masks, and a bus with one is labelled."""
        scan = self.g.emitter.emit_from_state(x_true)
        tampered = self.g.noise.tampered(self.g.emitter.exact(x_true), a)
        add = self.g.emitter.layers(np.asarray(a, float))
        seen = (scan.node_x + add.node, scan.edge_x + add.edge, scan.i_x + add.current)
        y = np.zeros(self.g.n_bus, np.uint8)
        y[self.g.plan.bus[tampered]] = 1
        return Frame.from_scan(
            scan, MeterLayers(*(x.astype(np.float32) for x in seen)), y, None, self.g.emitter.layers(tampered)
        )

    def currents_with_attack(
        self, scan: Scan, x_false: np.ndarray, x_true: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The PMU branch-current readings of a stealthy frame [WU26 eqs. 19-20]: the true scan's
        currents plus the attack vector on them, and the current channels written."""
        a = self.current_attack(x_false, x_true)
        moved = (np.abs(a) > ZERO_TOL) & (scan.i_m > 0)
        ix = scan.i_x.copy()
        ix[moved] += a[moved].astype(np.float32)
        return ix, ix != scan.i_x

    def attack_vector(self, x_false: np.ndarray, x_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The attack vector a = h(x_false) - h(x_true) [WU26 eqs. 13-18] per node channel [N, 4] and per flow
        channel [E, 2], in the scan's physical units: the noiseless reading of the false state minus
        that of the true state (unmetered flows zero on both sides)."""
        flows = self.g.emitter.clean_flows_from_states(
            np.stack([x_false, x_true])
        )  # [2, E, 2], unmetered zeroed
        return (np.asarray(x_false, float) - np.asarray(x_true, float)).astype(np.float32), flows[0] - flows[
            1
        ]

    def current_attack(self, x_false: np.ndarray, x_true: np.ndarray) -> np.ndarray:
        """The attack vector on the PMU branch-current channels [E, 4] (per unit), zero where no PMU
        reads that end. The currents of a false state follow from its voltages, so the attacker must
        write them too [WU26 eqs. 19-20]."""
        cur = self.g.emitter.currents_from_states(np.stack([x_false, x_true]))
        return cur[0] - cur[1]


def changed_meters(
    a_node: np.ndarray, a_edge: np.ndarray, scan: Scan, tol: float = 1e-7
) -> tuple[np.ndarray, np.ndarray]:
    """The metered channels the attack vector moves (the tamper set): the interior's and the
    boundary's injections and voltages and every flow on a branch touching the interior."""
    node = (np.abs(a_node) > tol) & (scan.node_m > 0)
    edge = (np.abs(a_edge) > tol) & (scan.edge_m > 0)
    return node, edge
