"""One attacked (or benign) scan, the per-frame physics shared by the shard and stream generators.

`generate` (shards) and `generate_stream` (streams) used to carry their own copy of this: re-solve
the grid under a scaled load for the stealthy families, re-solve under a load redistribution for
Al, corrupt the emitted measurements in place for Ad/As/Ar, emit the stored state for a benign
scan. One function now does it for both, with two switches in `FrameKnobs` for the two behaviours
that differ:

- `reject_below_floor`: the shard generator rejects a stealthy scan whose designed change sits
  inside the noise floor, and a replay whose realized change leaves the plausibility band, so the
  draw loop redraws; the stream keeps every scan and lets the label say what happened.
- `with_benign`: the stream also emits the un-attacked measurement of the same scan (attack
  removed, noise kept) as its `benign` layer; the shard does not.

RNG-order invariant: every released shard and stream is reproduced bit for bit from its seed, so
the order of random draws here is fixed: the power-flow re-solve and the measurement emission
first, then (streams only) the benign emission; for the corrupt-in-place families the emission,
then the replay-lag draw, then the corruption draws. Changing that order changes every released
file. tests/test_frozen.py holds the line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from ..models.frames import (  # noqa: F401  re-exported: defined here before the models package
    Frame,
    FrameKnobs,
    Scan,
)
from ..models.grid import EDGE, NODE

if TYPE_CHECKING:
    from .core import FdiaGenerator

# Family ids (see fdia_graph.FAMILIES): 0 benign, 1 Aq, 2 Ad, 3 As, 4 Ar, 5 At (ramp), 6 Al (LRA),
# 7 Am (multi-snapshot, timelines only).
RESOLVE_FAMILIES = (1, 5)  # stealthy: the grid is re-solved under a scaled load
RAMP_FAMILY = 5
LRA_FAMILY = 6
AM_FAMILY = 7
# The draw order of the single-shot families. Fixed: it sets the RNG sequence of every shard and stream.
SINGLE_SHOT_ORDER = (1, 2, 3, 4, 6)
CORRUPT_KIND = {2: "Ad", 3: "As", 4: "Ar"}  # corrupt-in-place families and their AttackMixin.corrupt code
BENIGN_BUFFER = 300  # recent benign scans kept for the replay families (FIFO)
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
) -> Optional[Frame]:
    """Build the scan of `family` on the stored operating point `Xt` ([N, 4] = |V|, P_inj, Q_inj, theta).

    targets index the generator's load-bus table (positions, not bus numbers); mult is the load
    multiplier of the re-solve families (a scalar for the ramp, one per target for Aq). Returns
    None when the scan is rejected: a non-converging power flow, no feasible redistribution, or,
    with knobs.reject_below_floor, a designed or realized change inside the noise floor.
    """
    if family in RESOLVE_FAMILIES:
        return _resolve_frame(g, Xt, family, targets, mult, knobs)
    if family == LRA_FAMILY:
        return _lra_frame(g, Xt, knobs)
    if family == AM_FAMILY:
        return _am_frame(g, Xt, targets, mult, knobs)
    if family == 0:
        return _benign_frame(g, Xt)
    return _corrupt_frame(g, Xt, family, targets, knobs)


def _resolve_frame(g, Xt, family, targets, mult, k: FrameKnobs) -> Optional[Frame]:
    """Aq and At: scale the targeted loads and re-solve the grid with generation pinned to the true
    dispatch, so the emitted measurements are a consistent (stealthy) state [DAT26]."""
    Lp = Xt[g.load_bus, 1] + g.load_genP  # base active load = stored P at load buses + generator P there
    Lq = Xt[g.load_bus, 2].copy()
    Lp_true = Lp.copy()  # the unattacked load pins the generation dispatch in solve()
    dev = np.abs(np.asarray(mult) - 1.0)  # per-bus designed load-shift fraction
    if k.reject_below_floor and family == 1 and np.max(dev) < k.floor:
        return None  # a within-noise no-op; the ramp is exempt so its per-scan step may stay sub-floor
    Lp = Lp.copy()
    Lp[targets] *= mult
    net = g.solve(Lp, Lq, Xt=Xt, Lp_true=Lp_true)
    if net is None:
        return None  # non-convergence, expected occasionally
    scan = g.emit(net)
    buses = g.load_bus[targets]
    y = np.zeros(g.C, np.uint8)
    y[buses] = 1
    bnx, bex = _benign_of(g, Xt) if k.with_benign else (None, None)
    return Frame(
        scan.node_x,
        scan.node_m,
        scan.edge_x,
        scan.edge_m,
        y,
        1,
        buses,
        np.broadcast_to(dev, buses.shape).astype(float),
        bnx,
        bex,
    )


def _lra_frame(g, Xt, k: FrameKnobs) -> Optional[Frame]:
    """Al: a load redistribution over up to lra_k buses that conserves total load, then re-solve
    with generation pinned to the true dispatch [DAT26]."""
    Lp = Xt[g.load_bus, 1] + g.load_genP
    Lq = Xt[g.load_bus, 2].copy()
    red = g.lra_delta(Lp, k.intensity, k.lra_k, floor=k.floor)
    a = red.buses
    if len(a) == 0:
        return None  # no feasible redistribution
    dev = np.abs(red.delta[a]) / (np.abs(Lp[a]) + 1e-6)  # designed redistribution fraction per bus
    if k.reject_below_floor and np.min(dev) < k.floor:
        return None  # a bus inside the noise floor
    net = g.solve(Lp + red.delta, Lq, Xt=Xt, Lp_true=Lp)
    if net is None:
        return None
    scan = g.emit(net)
    buses = g.load_bus[a]
    y = np.zeros(g.C, np.uint8)
    y[buses] = 1
    bnx, bex = _benign_of(g, Xt) if k.with_benign else (None, None)
    return Frame(scan.node_x, scan.node_m, scan.edge_x, scan.edge_m, y, 1, buses, dev.astype(float), bnx, bex)


def _am_frame(g, Xt, targets, mult, k: FrameKnobs) -> Optional[Frame]:
    """Am, the multi-snapshot attack [WU26]: one step of a load redistribution held over an episode,
    re-solved with generation pinned like Aq, then made sparse. The timeline walker draws the
    redistribution once at onset and passes this frame's fraction of it as `mult` (one multiplier
    per target); after emission every meter whose designed change against the true state sits
    under `am_sigma` accuracy-class stds is left at its un-attacked reading, so the tampered set is
    the meters the attack moves beyond noise (the l0 objective of [WU26], with the noise floor as
    the threshold). Needs `with_benign`: the twin is what the untouched meters read.
    """
    assert k.with_benign, "Am needs the benign twin (with_benign=True)"
    Lp = Xt[g.load_bus, NODE.p_inj] + g.load_genP
    Lq = Xt[g.load_bus, NODE.q_inj].copy()
    Lp_true = Lp.copy()
    Lp = Lp.copy()
    Lp[targets] *= mult
    net = g.solve(Lp, Lq, Xt=Xt, Lp_true=Lp_true)
    if net is None:
        return None
    Xa = g.state_from_net(net)  # the attacked state, AC-consistent by construction
    scan = g.emit_from_state(Xa)
    bnx, bex = _benign_of(g, Xt)
    tamper = _beyond_noise(g, Xa, Xt, scan, k.am_sigma)
    nx, ex = scan.node_x.copy(), scan.edge_x.copy()
    nx[~tamper[0]] = bnx[~tamper[0]]
    ex[~tamper[1]] = bex[~tamper[1]]
    buses = g.load_bus[targets]
    y = np.zeros(g.C, np.uint8)
    y[buses] = 1
    dev = np.abs(np.asarray(mult, float) - 1.0)
    return Frame(nx, scan.node_m, ex, scan.edge_m, y, 1, buses, dev, bnx, bex, tamper)


def _beyond_noise(g, Xa, Xt, scan: Scan, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """The metered channels whose noiseless change from the true state `Xt` to the attacked state
    `Xa` exceeds `sigma` accuracy-class stds (relative stds for P and Q with the emitter's floor,
    absolute for |V| and the angle). Am tampers exactly these; the rest read un-attacked."""
    sd = g.SD
    thr_node = np.empty(Xt.shape, float)
    thr_node[:, NODE.v] = sd["v"]
    thr_node[:, NODE.theta] = np.degrees(sd["va"])
    thr_node[:, NODE.p_inj] = np.abs(Xt[:, NODE.p_inj]) * sd["pi"] + 1e-3
    thr_node[:, NODE.q_inj] = np.abs(Xt[:, NODE.q_inj]) * sd["qi"] + 1e-3
    flows = g.clean_flows_from_states(np.stack([Xa, Xt]))  # [2, E, 2], unmetered zeroed
    thr_edge = np.empty(flows[1].shape, float)
    thr_edge[:, EDGE.p_from] = np.abs(flows[1][:, EDGE.p_from]) * sd["pf"] + 1e-3
    thr_edge[:, EDGE.q_from] = np.abs(flows[1][:, EDGE.q_from]) * sd["qf"] + 1e-3
    node = (np.abs(Xa - Xt) >= sigma * thr_node) & (scan.node_m > 0)
    edge = (np.abs(flows[0] - flows[1]) >= sigma * thr_edge) & (scan.edge_m > 0)
    return node, edge


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


def _benign_of(g, Xt):
    """The un-attacked measurement of a scan whose attacked version was just emitted (streams)."""
    scan = g.emit_from_state(Xt)
    return scan.node_x, scan.edge_x
