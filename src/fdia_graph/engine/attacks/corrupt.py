"""The in-place families Ad, As and Ar: the emitted measurements tampered at the attacked buses and
their incident branches without re-solving, so bad-data detection can see them [DAT26]."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...models.choices import FAMILY_CODE
from ...models.frames import Band, Frame, FrameKnobs, Scan, TamperTarget
from ..base import GridBase

# the in-place families and their `corrupt` code
CORRUPT_KIND = {FAMILY_CODE[n]: n for n in ("Ad", "As", "Ar")}
BENIGN_BUFFER = 300  # recent benign scans kept for the replay family (FIFO)
REPLAY_MIN_LAG = 20  # a random replay reaches at least this many benign scans back


def replay_frame(
    buffer: list[np.ndarray], tau: Optional[int], rng: np.random.Generator
) -> Optional[np.ndarray]:
    """The benign scan an Ar attack replays [DAT26]. It is drawn for every in-place family (Ad, As,
    Ar) so the random stream stays fixed; only Ar uses it.

    A fixed lag `tau` takes exactly that many scans back (clamped to what the buffer holds); no
    lag takes a random scan at least REPLAY_MIN_LAG back once the buffer is deep enough, else the
    oldest scan, else nothing (the attack then leaves the measurements untouched).
    """
    if tau is not None and buffer:
        return buffer[-min(tau, len(buffer))]
    if len(buffer) > REPLAY_MIN_LAG:
        return buffer[int(rng.integers(0, len(buffer) - REPLAY_MIN_LAG))]
    return buffer[0] if buffer else None


class CorruptMixin(GridBase):
    """Tamper emitted measurements in place, and remember the benign scans the replay draws from."""

    def remember_benign(self, nx: np.ndarray) -> None:
        """Keep a benign scan for the replay family, FIFO of BENIGN_BUFFER scans."""
        self.benign_buf.append(nx.copy())
        if len(self.benign_buf) > BENIGN_BUFFER:
            self.benign_buf.pop(0)

    def _corrupt_frame(
        self, Xt: np.ndarray, family: int, targets: np.ndarray, k: FrameKnobs
    ) -> Optional[Frame]:
        """Ad, As, Ar: emit the true state, then tamper the measurements at the attacked buses and
        their incident branches without re-solving, so bad-data detection can see them [DAT26]."""
        scan = self.emit_from_state(Xt)
        nx, nm, ex, em = (
            scan.node_x,
            scan.node_m,
            scan.edge_x,
            scan.edge_m,
        )  # corrupt() rewrites these in place
        bnx, bex = (nx.copy(), ex.copy()) if k.with_benign else (None, None)  # before corruption: same noise
        buses = self.load_bus[targets]  # corrupt() and the label index by bus, targets index the load table
        replay = replay_frame(self.benign_buf, k.replay_tau, self.rng)
        weak, mags = self.corrupt(scan, buses, CORRUPT_KIND[family], replay, k.band)
        nx[nm == 0] = 0.0
        ex[em == 0] = 0.0  # corrupt() can write unmetered channels; re-assert mask == 0 -> value == 0
        if k.reject_below_floor and weak:
            return None  # the replayed change fell inside the noise floor
        y = np.zeros(self.C, np.uint8)
        y[buses] = 1
        mag_bus = buses if len(mags) else np.zeros(0, int)
        # the in-place families write the SCADA readings; the PMU branch currents keep their true scan
        return Frame(
            nx,
            nm,
            ex,
            em,
            y,
            0,
            mag_bus,
            np.asarray(mags, float),
            bnx,
            bex,
            i_x=scan.i_x,
            i_m=scan.i_m,
            benign_i_x=scan.i_x,
        )

    def corrupt(
        self,
        scan: Scan,
        buses: np.ndarray,
        kind: str,
        replay: Optional[np.ndarray],
        band: Band = Band(0.02, 0.20),
    ) -> tuple[bool, np.ndarray]:
        """Measurement-level attacks (the BDD-detectable contrast families): tamper the emitted
        readings of `scan` in place at the attacked `buses` and their incident branches WITHOUT
        respecting the power-flow physics, which is why bad-data detection catches them [DAT26].

        The plausibility `band` keeps each tamper above the noise floor (not a within-noise no-op)
        and below the literature cap. Returns (weak, mags): `weak` flags a scan whose realized change
        left the band (only Ar can, since it replays the grid) so the caller can reject and redraw;
        `mags` is the realized per-bus |delta| / |base| on the P/Q injection channels. The family is
        decided once; each family's draws happen bus by bus, then branch by branch, in the order
        released files were built with.
        """
        target = TamperTarget(buses, self.incident_branches(buses))
        if kind == "Ad":
            return False, np.array(self._corrupt_bias(scan, target, band), float)
        if kind == "As":
            return False, np.array(self._corrupt_scaling(scan, target, band), float)
        if kind == "Ar" and replay is not None:
            mags, weak = self._corrupt_replay(scan, target, replay, band)
            return weak, np.array(mags, float)
        return False, np.zeros(0, float)  # Ar with nothing to replay yet: untouched

    def incident_branches(self, buses: np.ndarray) -> list[int]:
        """Branch positions with one of `buses` at either end."""
        return [e for e in range(self.E) if self.ei[0, e] in buses or self.ei[1, e] in buses]

    def _band_shift(self, cur: np.ndarray, band: Band) -> np.ndarray:
        """An additive perturbation with per-channel |delta| / |cur| drawn UNIFORMLY over the band
        and a random sign. An in-band draw (rather than clipping a big Gaussian) keeps Ad spread
        across the band instead of piled at the cap."""
        base = np.abs(cur) + 1e-6
        rel = self.rng.uniform(band.floor, band.cap, cur.shape)
        sign = np.where(self.rng.random(cur.shape) < 0.5, -1.0, 1.0)
        return sign * rel * base

    def _corrupt_bias(self, scan: Scan, target: TamperTarget, band: Band) -> list[float]:
        """Ad: an in-band additive shift on P/Q and a small |V| shift at each attacked bus, then an
        in-band shift on the flows of every incident branch."""
        nx, ex = scan.node_x, scan.edge_x
        mags = []
        for b in target.buses:
            base = np.abs(nx[b, 1:3]) + 1e-6
            sh = self._band_shift(nx[b, 1:3], band)
            nx[b, 1:3] += sh
            nx[b, 0] += self.rng.normal(0, 0.02)
            mags.append(float(np.max(np.abs(sh) / base)))
        for e in target.branches:
            ex[e] += self._band_shift(ex[e], band)
        return mags

    def _corrupt_scaling(self, scan: Scan, target: TamperTarget, band: Band) -> list[float]:
        """As: a multiplicative gain inside the band on P/Q at each attacked bus and on each incident flow."""
        nx, ex = scan.node_x, scan.edge_x
        mags = []
        for b in target.buses:
            gain = self.rng.uniform(1.0 + band.floor, 1.0 + band.cap)
            nx[b, 1:3] *= gain
            mags.append(abs(gain - 1.0))
        for e in target.branches:
            ex[e] *= self.rng.uniform(1.0 + band.floor, 1.0 + band.cap)
        return mags

    def _corrupt_replay(
        self, scan: Scan, target: TamperTarget, replay: np.ndarray, band: Band
    ) -> tuple[list[float], bool]:
        """Ar: replace each attacked bus's reading with an earlier benign scan's; weak when the realized
        change leaves the plausibility band. The branch flows are left as read."""
        nx = scan.node_x
        mags, weak = [], False
        for b in target.buses:
            base = np.abs(nx[b, 1:3]) + 1e-6
            cur = nx[b, 1:3].copy()
            nx[b, :] = replay[b, :]
            m = float(np.max(np.abs(nx[b, 1:3] - cur) / base))
            mags.append(m)
            if m < band.floor or m > band.cap:
                weak = True
        return mags, weak
