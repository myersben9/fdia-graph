"""One attacked (or benign) scan, the per-frame physics of the timeline writer.

The stealthy families (Aq, At, Al, Am) are local false states [WU26]: the attacker scales or
redistributes loads inside a subnetwork and solves that subnetwork's power flow with the boundary
voltages held true, so only the subnetwork's meters change and the measurement vector stays
consistent with an AC state. Ad/As/Ar corrupt the emitted measurements in place; a benign scan
emits the stored state. Two switches in `FrameKnobs` remain from the record-shard writer
that shared this code until 0.18: `reject_below_floor` (reject a stealthy scan whose designed
change sits inside the noise floor; the timeline keeps every scan and lets the label say what
happened) and `with_benign` (also emit the un-attacked measurement of the same scan, the timeline's
`benign` layer).

RNG-order invariant: every released file is reproduced bit for bit from its seed, so the order of
random draws here is fixed: one emission per scan, the true one; a stealthy family adds its attack
vector to it without a draw, and the corrupt-in-place families follow the emission with the
replay-lag draw, then the corruption draws. Changing that order changes every released file.
tests/test_frozen.py holds the line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from ..formulas.attacks import generator_output, within_limits
from ..models.frames import (  # noqa: F401  re-exported: defined here before the models package
    Frame,
    FrameKnobs,
    Scan,
)
from ..models.grid import NODE

if TYPE_CHECKING:
    from .core import FdiaGenerator

# Family ids (see fdia_graph.FAMILIES): 0 benign, 1 Aq, 2 Ad, 3 As, 4 Ar, 5 At (ramp), 6 Al (LRA),
# 7 Am (multi-snapshot, timelines only).
RESOLVE_FAMILIES = (1, 5)  # stealthy: the grid is re-solved under a scaled load
RAMP_FAMILY = 5
LRA_FAMILY = 6
AM_FAMILY = 7
CORRUPT_KIND = {2: "Ad", 3: "As", 4: "Ar"}  # corrupt-in-place families and their AttackMixin.corrupt code
BENIGN_BUFFER = 300  # recent benign scans kept for the replay families (FIFO)
LRA_DRAWS = 40  # target lines an Al frame tries before giving up: a region may hold too few loads, or
# every redistribution at this operating point may push a boundary generator past its limits
AQ_HALVINGS = 3  # times an Aq load step with no local power-flow solution is halved before giving up
STEP_HALVINGS = 6  # the same for one frame of a ramp (At, Am), whose step is under the floor anyway
REPLAY_MIN_LAG = 20  # a random replay reaches at least this many benign scans back


def replay_frame(
    buffer: list[np.ndarray], tau: Optional[int], rng: np.random.Generator
) -> Optional[np.ndarray]:
    """The benign scan an Ar/As attack replays [DAT26].

    A fixed lag `tau` takes exactly that many scans back (clamped to what the buffer holds); no
    lag takes a random scan at least REPLAY_MIN_LAG back once the buffer is deep enough, else the
    oldest scan, else nothing (the attack then leaves the measurements untouched).
    """
    if tau is not None and buffer:
        return buffer[-min(tau, len(buffer))]
    if len(buffer) > REPLAY_MIN_LAG:
        return buffer[int(rng.integers(0, len(buffer) - REPLAY_MIN_LAG))]
    return buffer[0] if buffer else None


def remember_benign(g: FdiaGenerator, nx: np.ndarray) -> None:
    """Keep a benign scan for the replay families, FIFO of BENIGN_BUFFER scans."""
    g.benign_buf.append(nx.copy())
    if len(g.benign_buf) > BENIGN_BUFFER:
        g.benign_buf.pop(0)


def attack_frame(
    g: FdiaGenerator,
    Xt: np.ndarray,
    family: int,
    targets: Optional[np.ndarray],
    mult: Union[float, np.ndarray, None],
    knobs: FrameKnobs,
    interior: Optional[np.ndarray] = None,
) -> Optional[Frame]:
    """Build the scan of `family` on the stored operating point `Xt` ([N, 4] = |V|, P_inj, Q_inj, theta).

    targets index the generator's load-bus table (positions, not bus numbers); mult is the load
    multiplier of the stealthy families (a scalar for the ramp, one per target for Aq and Am);
    `interior` the attacker's subnetwork when the caller holds it over an episode (Am). Returns
    None when the scan is rejected: a non-converging local power flow, no feasible redistribution,
    or, with knobs.reject_below_floor, a designed or realized change inside the noise floor.
    """
    if family in RESOLVE_FAMILIES:
        return _resolve_frame(g, Xt, family, targets, mult, knobs)
    if family == LRA_FAMILY:
        return _lra_frame(g, Xt, knobs)
    if family == AM_FAMILY:
        return _am_frame(g, Xt, targets, mult, knobs, interior)
    if family == 0:
        return _benign_frame(g, Xt)
    return _corrupt_frame(g, Xt, family, targets, knobs)


def _stealthy_frame(g, Xt, targets, mult, interior, k: FrameKnobs) -> Optional[Frame]:
    """One local false state [WU26]: the targeted loads scaled by `mult`, the interior buses
    re-solved with the boundary voltages held true, and the attack vector a = h(x_false) - h(x_true)
    added to the true scan. Every meter keeps its own noise draw and the tampered ones (the meters
    the false state moves) are shifted by exactly what it moves them, so the measurement is a full
    AC state plus meter noise and the residual test sees noise only. A re-emission of the false
    state would draw each tampered meter's noise from the false reading instead, which a meter
    whose true reading is structurally zero (a condenser's P, a zero-injection bus) gives away.
    Needs `with_benign`: the true scan is the benign twin."""
    assert k.with_benign, "a stealthy frame needs the benign twin (with_benign=True)"
    Xa = stealthy_state(g, Xt, targets, mult, interior, k)
    if Xa is None:
        return None  # no local solution, or one outside the operating limits: the caller halves
    scan = g.emit_from_state(Xt)  # the true scan: the benign twin, and the draw every meter keeps
    bnx, bex = scan.node_x, scan.edge_x
    a_node, a_edge = _attack_vector(g, Xa, Xt)
    moved = _changed_meters(a_node, a_edge, scan)
    nx, ex = bnx.copy(), bex.copy()
    nx[moved[0]] += a_node[moved[0]]
    ex[moved[1]] += a_edge[moved[1]]
    tamper = (nx != bnx, ex != bex)  # the meters whose stored float32 reading changed, no fewer, no more
    buses = g.load_bus[targets]
    y = np.zeros(g.C, np.uint8)
    y[buses] = 1
    dev = np.abs(np.asarray(mult, float) - 1.0)
    return Frame(
        nx,
        scan.node_m,
        ex,
        scan.edge_m,
        y,
        1,
        buses,
        np.broadcast_to(dev, buses.shape).astype(float),
        bnx,
        bex,
        tamper,
    )


def stealthy_state(g, Xt, targets, mult, interior, k: FrameKnobs) -> Optional[np.ndarray]:
    """The local false state of `targets` scaled by `mult` on the interior, or None when the local
    power flow has no solution or the state breaks the operating limits [WU26, eqs. 21-23]. Spends
    no random draw, so an episode can test its design at onset and redraw."""
    Lp = (
        Xt[g.load_bus, NODE.p_inj] + g.load_genP
    )  # base active load = stored P at load buses + generator P there
    Lq = Xt[g.load_bus, NODE.q_inj].copy()
    Lp_true, Lp = Lp, Lp.copy()
    Lp[targets] *= mult
    Xa = g.solve_local(Xt, interior, Lp, Lq)
    if Xa is None:
        return None
    if k.limits is not None and not _within_limits(g, Xa, Xt, Lp - Lp_true, k.limits):
        return None
    return Xa


def is_feasible(g, Xt, targets, mult, k: FrameKnobs, interior=None) -> bool:
    """Whether the design (targets, mult) has a stealthy state on `Xt`, on `interior` or the
    region around the targets; the onset test of an episode (no random draw)."""
    if interior is None:
        interior = g.local_region(g.load_bus[targets], k.hops)
    return interior is not None and stealthy_state(g, Xt, targets, mult, interior, k) is not None


def _within_limits(g, Xa, Xt, load_delta_pos, limits) -> bool:
    """[WU26, eqs. 21-23] on a false state: `load_delta_pos` is the pretended load change per
    load-table position (MW), summed per bus for buses carrying several loads."""
    dload = np.zeros(g.C)
    np.add.at(dload, g.load_bus, load_delta_pos)
    return within_limits(Xa, Xt, generator_output(Xt, g.load_base, g.gen_base), dload, limits)


def _attack_vector(g, Xa, Xt) -> tuple[np.ndarray, np.ndarray]:
    """The attack vector a = h(x_false) - h(x_true) [WU26] per node channel [N, 4] and per flow
    channel [E, 2], in the scan's physical units: the noiseless reading of the false state minus
    that of the true state (unmetered flows zero on both sides)."""
    flows = g.clean_flows_from_states(np.stack([Xa, Xt]))  # [2, E, 2], unmetered zeroed
    return (np.asarray(Xa, float) - np.asarray(Xt, float)).astype(np.float32), flows[0] - flows[1]


def _changed_meters(
    a_node: np.ndarray, a_edge: np.ndarray, scan: Scan, tol: float = 1e-7
) -> tuple[np.ndarray, np.ndarray]:
    """The metered channels the attack vector moves (the tamper set): the interior's and the
    boundary's injections and voltages and every flow on a branch touching the interior."""
    node = (np.abs(a_node) > tol) & (scan.node_m > 0)
    edge = (np.abs(a_edge) > tol) & (scan.edge_m > 0)
    return node, edge


def _solvable_step(
    g, Xt, targets, mult, interior, k: FrameKnobs, limit: tuple[int, Optional[float]]
) -> Optional[Frame]:
    """The stealthy frame at `mult`, or at the largest halving of its step that has a local
    power-flow solution: `limit` = (halvings tried, the floor a halved step may not fall under, or
    None). A step the region cannot absorb (a large load inside a fixed boundary) keeps the frame
    attacked at the largest step that solves; the frame's magnitudes record the step used."""
    halvings, floor = limit
    frame = _stealthy_frame(g, Xt, targets, mult, interior, k)
    for _ in range(halvings):
        if frame is not None:
            break
        mult = 1.0 + (np.asarray(mult, float) - 1.0) / 2
        if floor is not None and np.max(np.abs(mult - 1.0)) < floor:
            break
        frame = _stealthy_frame(g, Xt, targets, mult, interior, k)
    return frame


def _resolve_frame(g, Xt, family, targets, mult, k: FrameKnobs) -> Optional[Frame]:
    """Aq and At: the targeted loads scaled, the subnetwork within `hops` of them re-solved locally.
    A step without a local solution is halved: an Aq step at most AQ_HALVINGS times and never under
    the noise floor, a ramp frame at most STEP_HALVINGS times (its design step is sub-floor)."""
    dev = np.abs(np.asarray(mult) - 1.0)  # per-bus designed load-shift fraction
    if k.reject_below_floor and family == 1 and np.max(dev) < k.floor:
        return None  # a within-noise no-op; the ramp is exempt so its per-scan step may stay sub-floor
    interior = g.local_region(g.load_bus[targets], k.hops)
    if interior is None:
        return None
    limit = (AQ_HALVINGS, k.floor) if family == 1 else (STEP_HALVINGS, None)
    return _solvable_step(g, Xt, targets, mult, interior, k, limit)


def _lra_frame(g, Xt, k: FrameKnobs) -> Optional[Frame]:
    """Al: a load-conserving redistribution over up to lra_k buses of the subnetwork around a
    target line, steering that line, re-solved locally [DAT26, WU26]; a line whose redistribution
    has no stealthy state at any halving above the floor is redrawn."""
    Lp = Xt[g.load_bus, NODE.p_inj] + g.load_genP
    for _ in range(LRA_DRAWS):  # a line with no feasible or no solvable redistribution is redrawn
        red = g.lra_delta(Lp, k.intensity, k.lra_k, floor=k.floor, hops=k.hops)
        a = red.buses
        if len(a) == 0:
            continue
        dev = np.abs(red.delta[a]) / (np.abs(Lp[a]) + 1e-6)  # designed redistribution fraction per bus
        if k.reject_below_floor and np.min(dev) < k.floor:
            return None  # a bus inside the noise floor
        mult = 1.0 + red.delta[a] / np.where(np.abs(Lp[a]) > 1e-9, Lp[a], 1e-9)
        frame = _solvable_step(g, Xt, a, mult, red.interior, k, (AQ_HALVINGS, k.floor))
        if frame is not None:
            return frame
    return None  # no drawn line gave a redistribution with a stealthy state at any halving


def _am_frame(g, Xt, targets, mult, k: FrameKnobs, interior=None) -> Optional[Frame]:
    """Am, the multi-snapshot attack [WU26]: one step of a load redistribution held over an episode,
    drawn once at onset by the timeline walker, which passes this frame's fraction of it as `mult`
    (one multiplier per target) and the attacker's interior; the local false state of that step."""
    if interior is None:
        interior = g.local_region(g.load_bus[targets], k.hops)
    if interior is None:
        return None
    return _solvable_step(g, Xt, targets, mult, interior, k, (STEP_HALVINGS, None))


def _benign_frame(g, Xt) -> Frame:
    """A benign scan: the stored state emitted through the meter plan, and remembered for replay."""
    scan = g.emit_from_state(Xt)
    remember_benign(g, scan.node_x)
    empty = np.zeros(0, int)
    return Frame(
        scan.node_x,
        scan.node_m,
        scan.edge_x,
        scan.edge_m,
        np.zeros(g.C, np.uint8),
        0,
        empty,
        np.zeros(0, float),
        None,
        None,
    )


def _corrupt_frame(g, Xt, family, targets, k: FrameKnobs) -> Optional[Frame]:
    """Ad, As, Ar: emit the true state, then tamper the measurements at the attacked buses and
    their incident branches without re-solving, so bad-data detection can see them [DAT26]."""
    nx, nm, ex, em = g.emit_from_state(Xt)  # unpacked: corrupt() rewrites nx and ex in place
    bnx, bex = (nx.copy(), ex.copy()) if k.with_benign else (None, None)  # before corruption: same noise
    buses = g.load_bus[targets]  # corrupt() and the label index by bus, targets index the load table
    replay = replay_frame(g.benign_buf, k.replay_tau, g.rng)
    nx, ex, weak, mags = g.corrupt(
        nx, ex, buses, CORRUPT_KIND[family], replay, floor=k.floor, cap=k.intensity
    )
    nx[nm == 0] = 0.0
    ex[em == 0] = 0.0  # corrupt() can write unmetered channels; re-assert mask == 0 -> value == 0
    if k.reject_below_floor and weak:
        return None  # the replayed change fell inside the noise floor
    y = np.zeros(g.C, np.uint8)
    y[buses] = 1
    mag_bus = buses if len(mags) else np.zeros(0, int)
    return Frame(nx, nm, ex, em, y, 0, mag_bus, np.asarray(mags, float), bnx, bex)
