"""The New York grid as the source of a timeline: the year of pool states and its steps of the one path.

`NewYorkSource` is a `generation.source.TimelineSource`: `source.generate(path)` makes its timeline by the
path every timeline takes (`TimelineSource.generate`), in the same order as an IEEE one. What is the New York
grid's own:

- **Before the splits** (`prepare`): the onsets a family may take. An overload family that waits for a loaded
  grid allows only near-limit onsets (its `onset_mask`, N36). With `survey` (N42), a waiting family's allowed
  onsets on a grid of that many frames are all designed first, and its episodes are placed uniformly among
  those with an attack, reusing their designs. What is found is kept in one record, `Survey`.
- **The splits** (`cut_splits`) are whole weeks stratified on the loaded frames (`split_weeks`, N41; after a
  survey, on the onsets where an attack exists, N46), or three blocks in time order (`split_by="time"`).
- **The design**: every placed episode is designed on its own window of true pool states by its family
  (`AttackFamily.design`), from a stream keyed by family and onset, so the result does not depend on the worker
  count. A window the family finds no stealthy attack for, or one that touches an unsolved pool frame, stays
  benign: nothing is moved.
- **A frame** (`walk`): its benign scan is the exact reading of the true state with accuracy-class noise
  (`NoiseModel.sample`, one noise stream over the file's frames in time order). An attacked frame is that same
  noisy scan plus the episode's attack vector a = h(x_false) - h(x_true), never a fresh scan of the false state,
  so `z - z_benign` is exactly the attack. The labels come from the meters the attack moves beyond their noise
  (`Attacker.tampered`): `meter_tampered [T, M]`, and `bus_attacked [T, N]` true at a bus when a meter at that
  bus is tampered, so a client's labels follow from its own meters alone. The weekly splits interleave in time,
  so the frames are written in stretches, each as soon as every week before it is planned; the values do not
  depend on where a stretch ends.
- **The file** is the New York layout (`TimelineWriter`, docs/concepts/new_york_grid.md).

`SingleEpisodeSource` is the same grid with one episode placed by hand, for studying one attack.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from fdia_graph.engine.attacks.attacker import (
    Attacker,
    generator_band,
    generator_q_capability,
    generator_ramps,
    natural_steps,
)
from fdia_graph.engine.attacks.family import AttackFamily, Episode
from fdia_graph.engine.attacks.interface import InterfaceFdia
from fdia_graph.engine.attacks.ramp import Ramp
from fdia_graph.engine.sensing.measure import Measurement
from fdia_graph.engine.sensing.meters import plan_meters
from fdia_graph.engine.sensing.noise import NoiseModel
from fdia_graph.errors import ConfigError, DataConditionError, NoAdmissibleTarget
from fdia_graph.generation.design import DesignPool
from fdia_graph.generation.nygrid.case import Case, nygrid_case
from fdia_graph.generation.nygrid.config import Settings, rng
from fdia_graph.generation.nygrid.inputs import GEOJSON, LOCATIONS
from fdia_graph.generation.nygrid.interfaces import hourly_interfaces
from fdia_graph.generation.nygrid.io import FAMILIES, TimelineWriter, read_pool
from fdia_graph.generation.nygrid.network import Network
from fdia_graph.generation.nygrid.ownership import Ownership, assign_clients
from fdia_graph.generation.nygrid.pool import SOLVED, Pool, peak_ratings
from fdia_graph.generation.plan import STRATUM, split_bounds, split_column, split_weeks, week_index
from fdia_graph.generation.source import OutPath, TimelineSource
from fdia_graph.models.choices import NEW_YORK_FAMILIES, SPLITS, SplitBy
from fdia_graph.models.config import NewYorkRelease, NewYorkSettings
from fdia_graph.models.plan import Placement, Slot, SplitSpan

# weeks per stratum of the split when the weight is the surveyed attacks (N46): they sit in about 9 weeks, one
# of them holding 40%, so a 20-week stratum can leave a split one episode; 5 draws 3/1/1 per stratum
HIT_STRATUM = 5
SURVEY_CHUNK = 240  # onsets designed and saved at a time with `survey_cache`
LOG_EVERY = 50  # designs between two progress lines of a verbose build
AM, AT = NEW_YORK_FAMILIES  # the families `NewYorkSource.production` builds
AM_RISE = 16  # Am's rise scans; its window adds as many exit scans (N37)
AT_LENGTH = 16  # the slow ramp's window (scans)
RELEASE_LENGTH = {AM: 2 * AM_RISE, AT: AT_LENGTH}  # the release's episode lengths (scans)
ATTACKED_FRAC = 0.2


@dataclass(frozen=True)
class _Task:
    """One design to try: the family, the pool frame its window starts at (the key of its random
    stream), and the true state over the window, `vm [L, N]` p.u. and `va [L, N]` degrees."""

    family: str
    onset: int
    vm: np.ndarray
    va: np.ndarray


@dataclass(frozen=True)
class _Tried:
    """What one slot's design came to: the episode or None, the seconds it took, and why there is none (a
    miss reason of `engine.attacks.family`; None with an episode). `designed` is False for a slot that was
    never designed, its window touching an unsolved pool frame."""

    episode: Episode | None = None
    seconds: float = 0.0
    miss: str | None = None
    designed: bool = True


@dataclass(frozen=True)
class _Designing:
    """What a design process designs with: the families by name, and the build settings whose seed keys
    each design's random stream."""

    families: dict[str, AttackFamily]
    build: Settings


def _designing(setup: _Designing) -> _Designing:
    """A design process starts with the families it was sent: nothing more to make."""
    return setup


def _design(state: _Designing, task: _Task) -> _Tried:
    """One episode design, from the stream keyed by its family and onset."""
    start = time.perf_counter()
    ep, miss = state.families[task.family].design_explained(
        (task.onset, task.vm, task.va), rng(state.build, f"design/{task.family}/{task.onset}")
    )
    return _Tried(ep, time.perf_counter() - start, miss)


@dataclass
class Survey:
    """What a source knows about the onsets before its splits are cut (`NewYorkSource.prepare`).

    `masks` are the onsets each waiting family may take, [T] bool by family: its own `onset_mask`, or,
    after the onset survey (N42), the surveyed onsets where it found an attack. The survey also leaves
    the `designs` it found, by (family, onset), for the design stage to reuse; the onsets it `tried` per
    family; why the ones without an attack have none, counted per family (`missed`); and the weeks per
    `stratum` the splits are then stratified in (N46)."""

    masks: dict[str, np.ndarray] = field(default_factory=dict)
    designs: dict[tuple[str, int], _Tried] = field(default_factory=dict)
    tried: dict[str, np.ndarray] = field(default_factory=dict)
    missed: dict[str, dict[str, int]] = field(default_factory=dict)
    stratum: int = STRATUM

    @classmethod
    def of(cls, families: list[str], frames: int, cands: list[Slot[str]], tried: list[_Tried]) -> Survey:
        """The survey of the onsets `cands`, each with what its design came to (`tried`), over a year
        of `frames` frames."""
        survey = cls(
            masks={f: np.zeros(frames, bool) for f in families},
            tried={f: np.zeros(frames, bool) for f in families},
            stratum=HIT_STRATUM,
        )
        for c, t in zip(cands, tried):
            survey.tried[c.family][c.onset] = True
            if t.episode is not None:
                survey.masks[c.family][c.onset] = True
                survey.designs[(c.family, c.onset)] = _Tried(t.episode, t.seconds)
        survey.missed = {
            f: dict(Counter(t.miss for c, t in zip(cands, tried) if c.family == f and t.miss))
            for f in families
        }
        return survey

    def counts(self, split: np.ndarray) -> dict[str, list[int]]:
        """Per split and family ("train/Am"): the onsets tried and those with an attack."""
        out = {}
        for f in self.tried:
            for code, name in enumerate(SPLITS):
                here = split == code
                out[f"{name}/{f}"] = [int((self.tried[f] & here).sum()), int((self.masks[f] & here).sum())]
        return out


def _ownership(case: Case, root: Path) -> Ownership:
    """One client per utility, from the territories `fetch_inputs` left under `root/data`."""
    terr = root / "data" / "territories"
    with open(terr / GEOJSON, encoding="utf-8") as f:
        return assign_clients(case, json.load(f), terr / LOCATIONS)


def _interface_family(att: Attacker, pool: Pool, release: NewYorkRelease, nygrid: Path) -> InterfaceFdia:
    """The release's Am: the OPF attacker against the NYISO interfaces (N48, N49) with the units' output
    (N42) and every realism bound (N27 to N37)."""
    case = att.meas.net.case
    if pool.unit_p is None:
        raise DataConditionError(
            "the pool must carry each unit's output (built by pipelines/newyork/run_pool.py)"
        )
    # Am's exit: each bus's natural five-minute step by hour over the whole pool (what the attacker knows),
    # generator buses bounded by their units' five-minute ramp (nuclear 0, hydro and imports inf)
    steps = natural_steps(pool.time, pool.p, pool.solve_kind == SOLVED)
    steps[:, att.gen_bus] = generator_ramps(case)[att.gen_bus]
    q_lo, q_hi = generator_q_capability(case)
    gen_output = np.zeros((len(pool.time), len(case.bus_id)))  # the units' output at each bus (N42)
    np.add.at(gen_output.T, case.unit_bus, np.nan_to_num(pool.unit_p).T)
    fam = InterfaceFdia(
        att,
        hourly_interfaces(case, pool.time, nygrid, list(release.interfaces)),
        exit_scans=AM_RISE,
        gen_ramp=generator_ramps(case),
        gen_band=generator_band(case) if release.band is None else generator_band(case, release.band),
        q_lo=q_lo,
        q_hi=q_hi,
        step_bound=steps,
        frame_hour=(pool.time.astype(np.int64) // 3600) % 24,
        gen_output=gen_output,
        loads_only=False,
        gen_v_tol=release.gen_v_tol,
    )
    if release.load_cap is not None:  # otherwise the family keeps its own cap (N28)
        fam.load_cap = release.load_cap
    return fam


class NewYorkSource(TimelineSource[str, Episode]):
    """The New York grid a timeline is generated on: the `pool` of five-minute states, the `attacker` (with
    its `Measurement`), the attack `families` by name ("Am", "At"), the `Ownership` of the buses, the
    `settings` of the timeline (`NewYorkSettings`) and the `build` settings whose seed keys every draw.
    `production` builds the one of a release.

    `source.generate(path)` writes the timeline; `source.summary()` then says what was placed and built. A
    family with an `onset_mask` (an overload family waiting for a loaded grid, N36) has its episodes start
    only where the mask allows."""

    def __init__(
        self,
        pool: Pool,
        attacker: Attacker,
        families: dict[str, AttackFamily],
        own: Ownership,
        settings: NewYorkSettings = NewYorkSettings(),
        build: Settings = Settings(),
    ) -> None:
        self.pool, self.att, self.meas = pool, attacker, attacker.meas
        self.noise = NoiseModel(self.meas.plan)
        self.families, self.own, self.settings, self.build = families, own, settings, build
        self.attacked_frac = settings.attacked_frac
        # the pool frames the file covers, [first, stop): the whole pool
        self.first, self.stop = 0, len(pool.time)
        self._writer: TimelineWriter | None = None
        self._designers: DesignPool[_Designing, _Designing] | None = None
        self._reset()

    def _reset(self) -> None:
        """The records and random streams of one run (placement, noise), fresh: one source may generate
        several files."""
        self.rec = Placement.empty(list(self.families))
        self._place_rng = rng(self.build, "place")
        self._noise_rng = rng(self.build, "noise")
        self._survey = Survey()
        self._split = np.zeros(len(self.pool.time), np.int8)  # the split code of every pool frame
        self._cursor = self.first  # the next pool frame to write
        self._episodes: list[Episode] = []  # the designed episodes, sorted by onset: the file's episode rows
        self._tried: list[tuple[Slot[str], _Tried]] = []  # every placed slot with what its design came to
        self._started, self._seconds = time.perf_counter(), 0.0

    @classmethod
    def production(
        cls,
        pool_path: str | Path,
        release: NewYorkRelease = NewYorkRelease(),
        build: Settings = Settings(),
    ) -> NewYorkSource:
        """The source of the release, with every decided bound (N8, N27 to N49) in one place: the case,
        clients, meters and the pool at `pool_path` (relative to `release.root`, whose `data/` holds
        `fetch_inputs`'s files), with the families `release` names: Am targets the NYISO interfaces (N48,
        N49) from the OPF attacker with the units' output (N42), At is the slow ramp. The Am window is 32
        scans, the At window 16, and 20% of each split's frames are placed under an episode. The attacker's
        sensitivities are the fields of `NewYorkRelease`."""
        root = Path(release.root)
        case = nygrid_case()
        own = _ownership(case, root)
        pool = read_pool(root / pool_path)
        case = peak_ratings(case, pool)  # production ratings over the whole pool (N8)
        net = Network(case)
        meas = Measurement(net, plan_meters(net, own, rng(build, "meters")))
        # N71: a voltage margin widens the attacker's limits on its own copy of the case
        att_net = Network(case.with_voltage_margin(release.v_margin)) if release.v_margin else net
        att = Attacker(
            att_net, meas, hold_external=release.hold_external
        )  # N62: True keeps the interchange true
        families: dict[str, AttackFamily] = {}
        for name in release.families:
            if name == AM:
                families[name] = _interface_family(att, pool, release, root / build.nygrid_dir / "Data")
            else:
                families[name] = Ramp(att)
        settings = NewYorkSettings(
            length={f: RELEASE_LENGTH[f] for f in families},
            attacked_frac=ATTACKED_FRAC,
            workers=release.workers,
            verbose=True,
            survey=release.survey,
            survey_cache=release.survey_cache,
        )
        return cls(pool, att, families, own, settings, build)

    # ---- the steps of `generate` ---------------------------------------------------------------------

    @property
    def frames(self) -> int:
        """The frames of the timeline: the pool's length."""
        return self.stop - self.first

    def out_path(self, out: OutPath | None) -> str:
        """`out`: a New York timeline has no default path."""
        if out is None:
            raise ConfigError("a New York timeline needs the path of its file: source.generate(path)")
        return os.fspath(out)

    def open(self, path: str) -> None:
        """Start the file with the static case, ownership and meter plan, and a fresh run: its records,
        its random streams and its design processes."""
        self._reset()
        self._designers = DesignPool(self.settings.workers, _designing, _Designing(self.families, self.build))
        self._writer = TimelineWriter(path)
        # the true case, not the attacker's: a voltage margin (N71) widens only the attacker's limits
        self._writer.write_static(self.meas.net.case, self.own, self.meas.plan)

    def prepare(self) -> None:
        """Before the splits are cut: the onsets each waiting family may take (its `onset_mask`, N36). With
        `survey` (N42, N46), every such onset whose window lies in one week, on a grid of `survey` frames,
        is designed here (hours of work on a year), and the family is then held to the onsets where an
        attack was found, so every split gets its share of the weeks an attack can happen in (the
        near-limit hours of N41 left the test split one Am of 21, since most near-limit hours hold no
        attack). What is found is kept in `Survey`."""
        masks = self._family_masks()
        self._survey = self._surveyed(masks) if self.settings.survey else Survey(masks)

    def cut_splits(self) -> list[SplitSpan]:
        """The three splits, each with its frames: whole weeks stratified on the frames some waiting family
        may start at (N41; after a survey, on the onsets with an attack, N46), or three blocks in time
        order."""
        T, frac = len(self.pool.time), self.settings.split.fractions
        if self.settings.split_by == SplitBy.TIME:
            self._split = split_column(split_bounds(T, frac), T)
        else:
            loaded = np.zeros(T, bool)
            for mask in self._survey.masks.values():
                loaded |= mask
            self._split = split_weeks(
                self.pool.time, frac, loaded, rng(self.build, "split"), self._survey.stratum
            )
        return [self._span(code) for code in range(len(SPLITS))]

    def _span(self, code: int) -> SplitSpan:
        """The split `code` with the frames it holds, in time order."""
        where = self.first + np.flatnonzero(self._split[self.first : self.stop] == code)
        start, stop = (int(where[0]), int(where[-1]) + 1) if len(where) else (0, 0)
        return SplitSpan(code, start, stop, where)

    def lengths(self) -> dict[str, int]:
        """The episode length in frames of each family the source holds."""
        return self.settings.lengths_for(list(self.families))

    def placement_rng(self) -> np.random.Generator:
        """The build's "place" stream, one generator over the three splits in turn."""
        return self._place_rng

    def onset_masks(self) -> dict[str, np.ndarray]:
        """Each waiting family's allowed onsets [T] over the year (`onset_mask`, N36), or, after a survey,
        the onsets where it found an attack (N42): `Survey.masks`."""
        return self._survey.masks

    def design(self, span: SplitSpan, slots: list[Slot[str]]) -> None:
        """Every placed episode of the split designed by its family, the survey's designs reused. A slot
        with no attack keeps no episode: its frames stay benign and nothing is moved (`summary` counts
        why)."""
        self._keep(span, slots, self._designs(slots))

    def _keep(self, span: SplitSpan, slots: list[Slot[str]], tried: list[_Tried]) -> None:
        """Record what the designs of `slots` came to, and keep the episodes found for the walk, in onset
        order (an episode's place in that order is its row in the file)."""
        self._tried += list(zip(slots, tried))
        for s, t in zip(slots, tried):
            self.rec.built[span.code, self.rec.col(s.family)] += t.episode is not None
        found = [t.episode for t in tried if t.episode is not None]
        self._episodes = sorted(self._episodes + found, key=lambda ep: ep.onset)

    def walk(self, until: int) -> None:
        """Write the frames from the last one written up to `until`, a block at a time: each the noisy
        benign scan of its pool state, plus the attack vector and the labels of the episode it lies in.
        The noise stream runs over the frames in time order, so the scans are the same wherever a stretch
        or a block ends."""
        for t0 in range(self._cursor, until, self.settings.block):
            self._write_block(t0, min(until, t0 + self.settings.block))
        self._cursor = max(self._cursor, until)

    def finish(self) -> None:
        """After the walk: the episode table, onsets counted from the file's first frame."""
        assert self._writer is not None, "open() starts the file"
        self._writer.write_episodes([replace(ep, onset=ep.onset - self.first) for ep in self._episodes])
        self._seconds = time.perf_counter() - self._started

    def close(self, finished: bool) -> None:
        """Close the file and stop the design processes."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._designers is not None:
            self._designers.close(finished)
            self._designers = None

    # ---- what was built ------------------------------------------------------------------------------

    def summary(self) -> dict[str, object]:
        """What the last timeline generated holds: frames, slots placed and episodes built per split and
        family, slots skipped for unsolved frames, why designs found no attack, and the seconds spent."""
        return {
            "frames": self.frames,
            "families": list(self.families),
            "splits": list(SPLITS),
            "placed": (self.rec.requested - self.rec.dropped).tolist(),
            "built": self.rec.built.tolist(),
            "skipped_unsolved": sum(not t.designed for _, t in self._tried),
            "missed": {f: self._missed(f) for f in self.families},
            "survey": self._survey.counts(self._split),
            "survey_missed": self._survey.missed,  # why the surveyed onsets without an attack have none
            "attacked_frames": int(sum(ep.length for ep in self._episodes)),
            "design_seconds": round(float(sum(t.seconds for _, t in self._tried)), 1),
            "design_seconds_median": {f: self._median_seconds(f) for f in self.families},
            "seconds": round(self._seconds, 1),
        }

    def _missed(self, family: str) -> dict[str, int]:
        """Why the placed slots of `family` that were designed found no attack: each reason's count."""
        return dict(Counter(t.miss for s, t in self._tried if s.family == family and t.miss is not None))

    def _median_seconds(self, family: str) -> float | None:
        """The median seconds one design of `family` took; None when none was designed."""
        secs = [t.seconds for s, t in self._tried if s.family == family and t.seconds > 0]
        return round(float(np.median(secs)), 2) if secs else None

    # ---- the onsets ----------------------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.settings.verbose:
            print(msg, flush=True)

    def _family_masks(self) -> dict[str, np.ndarray]:
        """Each waiting family's allowed onsets [T] over the year (`onset_mask`, N36); a family that may
        start anywhere has none."""
        masks = {}
        for f, fam in self.families.items():
            mask = fam.onset_mask(self.pool.vm, self.pool.va, self.lengths()[f])
            if mask is not None:
                masks[f] = mask
        return masks

    def _onsets_ok(self, blocks: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """[T] per family: the onsets whose window lies in one block (`blocks` [T], a code per frame such
        as the week), on solved frames, where the family allows."""
        T, solved = len(blocks), self.pool.solve_kind == SOLVED
        run = np.r_[0, np.cumsum(blocks[1:] != blocks[:-1])]  # [T] which run of one block a frame is in
        unsolved = np.r_[0, np.cumsum(~solved)]
        ok = {}
        for f, L in self.lengths().items():
            a = np.zeros(T, bool)
            n = T - L + 1
            if n > 0:
                a[:n] = (run[L - 1 :] == run[:n]) & (unsolved[L:] - unsolved[:n] == 0)
            ok[f] = a & masks[f] if f in masks else a
        return ok

    def _surveyed(self, masks: dict[str, np.ndarray]) -> Survey:
        """N42: every allowed onset of each waiting family whose window lies in one week, on a grid of
        `survey` frames, designed; the `Survey` of what was found."""
        weeks = week_index(self.pool.time)
        ok = self._onsets_ok(weeks, masks)
        grid = np.arange(len(weeks)) % self.settings.survey == 0
        lengths = self.lengths()
        cands = [Slot(int(o), f, lengths[f]) for f in masks for o in np.flatnonzero(ok[f] & grid)]
        self._log(f"survey: {len(cands)} onsets on a {self.settings.survey}-frame grid")
        tried: list[_Tried] = []
        chunk = SURVEY_CHUNK if self.settings.survey_cache else max(len(cands), 1)
        for i in range(0, len(cands), chunk):
            tried += self._survey_chunk(cands, i, chunk)
            done = sum(t.episode is not None for t in tried)
            self._log(f"survey {min(i + chunk, len(cands))}/{len(cands)}: {done} with an attack")
        return Survey.of(list(masks), len(weeks), cands, tried)

    def _survey_chunk(self, cands: list[Slot[str]], i: int, chunk: int) -> list[_Tried]:
        """The designs of the surveyed onsets `cands[i : i + chunk]`. With `survey_cache`, each chunk is
        saved on its own (episodes, seconds, miss reasons), so a stopped build resumes."""
        cache = self.settings.survey_cache
        part = None if cache is None else Path(cache) / f"{i:06d}_{len(cands)}.pkl"
        if part is not None and part.exists():
            saved = pickle.loads(part.read_bytes())
            misses = saved[2] if len(saved) > 2 else [None] * len(saved[0])
            return [_Tried(e, t, m) for e, t, m in zip(saved[0], saved[1], misses)]
        tried = self._designs(cands[i : i + chunk])
        if part is not None:
            part.parent.mkdir(parents=True, exist_ok=True)
            columns = ([t.episode for t in tried], [t.seconds for t in tried], [t.miss for t in tried])
            part.write_bytes(pickle.dumps(columns))
        return tried

    # ---- the designs ---------------------------------------------------------------------------------

    def _task(self, slot: Slot[str]) -> _Task | None:
        """The design of `slot` to try, on its own window of pool states; None when the window touches an
        unsolved frame."""
        w = slice(slot.onset, slot.onset + slot.length)
        if not (self.pool.solve_kind[w] == SOLVED).all():
            return None
        return _Task(slot.family, slot.onset, self.pool.vm[w], self.pool.va[w])

    def _designs(self, slots: list[Slot[str]]) -> list[_Tried]:
        """What the design of each slot came to: an episode, or none where the family finds no attack or
        the window touches an unsolved frame (then it is not designed at all). A slot the survey already
        designed reuses that design."""
        assert self._designers is not None, "open() starts the design processes"
        known = self._survey.designs
        out = [known.get((s.family, s.onset), _Tried(designed=False)) for s in slots]
        todo = self._to_design(slots)
        results = self._designers.map(_design, [task for _, task in todo])
        for k, ((i, _), tried) in enumerate(zip(todo, results), 1):
            out[i] = tried
            if k % LOG_EVERY == 0 or k == len(todo):
                self._log(f"designed {k}/{len(todo)}")
        return out

    def _to_design(self, slots: list[Slot[str]]) -> list[tuple[int, _Task]]:
        """(position in `slots`, the design to try) of every slot that still needs a design and whose
        window is solved: the slow overload family first, so the workers finish together."""
        fresh = [(i, s) for i, s in enumerate(slots) if (s.family, s.onset) not in self._survey.designs]
        tasks = [(i, self._task(s)) for i, s in fresh]
        todo = [(i, task) for i, task in tasks if task is not None]
        return sorted(todo, key=lambda it: it[1].family == AT)

    # ---- the frames ----------------------------------------------------------------------------------

    def _write_block(self, t0: int, t1: int) -> None:
        """Write the pool frames t0..t1 (file frames count from `first`): the noisy benign scans, the attack
        vector of every episode that overlaps them added on top, and the labels. An episode's row in the file
        is its place among the designed episodes in onset order; every episode that starts before a frame
        being written is known by then."""
        assert self._writer is not None, "open() starts the file"
        pool, meas, plan = self.pool, self.meas, self.meas.plan
        M, N = len(plan.kind), pool.vm.shape[1]
        vm, va = pool.vm[t0:t1], pool.va[t0:t1]
        with np.errstate(invalid="ignore"):  # unsolved frames keep their place with NaN readings
            z_benign = self.noise.sample(meas.h(vm, va), self._noise_rng)
        z = z_benign.copy()
        family = np.zeros(t1 - t0, np.int8)
        episode = np.full(t1 - t0, -1, np.int32)
        meter_tampered = np.zeros((t1 - t0, M), bool)
        for e, ep in enumerate(self._episodes):
            a, b = max(t0, ep.onset), min(t1, ep.onset + ep.length)
            if a >= b:
                continue
            w = slice(ep.onset, ep.onset + ep.length)
            tam = self.att.tampered(pool.vm[w], pool.va[w], ep.a)  # its tampered meters [L, M], the labels
            z[a - t0 : b - t0] += ep.a[a - ep.onset : b - ep.onset]
            meter_tampered[a - t0 : b - t0] = tam[a - ep.onset : b - ep.onset]
            family[a - t0 : b - t0] = FAMILIES.index(ep.family)
            episode[a - t0 : b - t0] = e
        at_bus = np.zeros((M, N), np.int32)
        at_bus[np.arange(M), plan.bus] = 1  # a meter's bus: a tampered meter marks its bus attacked
        self._writer.write_frames(
            t0 - self.first,
            {
                "time": pool.time[t0:t1],
                "split": self._split[t0:t1],
                "family": family,
                "episode": episode,
                "z": z,
                "z_benign": z_benign,
                "vm": vm,
                "va": va,
                "bus_attacked": (meter_tampered.astype(np.int32) @ at_bus) > 0,
                "meter_tampered": meter_tampered,
            },
        )
        self._log(f"wrote frames {t0}..{t1}")


class SingleEpisodeSource(NewYorkSource):
    """The New York grid with one episode alone, placed by hand, for studying one attack: `family` designed
    at pool frame `onset` on its own window, the file covering the pool frames lo..hi (default the window
    itself), every frame in the train split. It is `source` (its pool, attacker, families and noise) with
    these steps of `generate` its own: no onset mask or survey, one split, the one slot in place of a random
    placement, and a design that must exist.

    `using` designs the episode in place of the source's own family of that name (a variant, such as
    `InterfaceFdia.with_overshoot`). With `lag` (frames) the attacker knows each scan only as it was `lag`
    frames earlier, a persistence forecast: the attack is designed on the window at `onset - lag` (its
    states, hours and headroom) and its vector added to the true scans from `onset`. The one design runs in
    this process. Building it raises if a frame of either window is unsolved; `generate` raises
    NoAdmissibleTarget if the family finds no attack there, and then leaves no file."""

    def __init__(
        self,
        source: NewYorkSource,
        onset: int,
        family: str = AM,
        lo: int | None = None,
        hi: int | None = None,
        using: AttackFamily | None = None,
        lag: int = 0,
    ) -> None:
        families = source.families if using is None else {**source.families, family: using}
        settings = replace(source.settings, workers=1, survey=0)
        super().__init__(source.pool, source.att, families, source.own, settings, source.build)
        self.noise = source.noise
        self.slot = Slot(onset, family, self.lengths()[family])
        self.lag = lag
        self.first = onset if lo is None else lo
        self.stop = onset + self.slot.length if hi is None else hi
        self._check_window()

    def _check_window(self) -> None:
        """The episode lies inside the file's frames, the lag reaches no frame before the pool, and both
        windows (the episode's and the one the attacker knows) are solved."""
        onset, L, lag = self.slot.onset, self.slot.length, self.lag
        if not (self.first <= onset and onset + L <= self.stop):
            raise ConfigError(
                f"the window {onset}..{onset + L} must lie inside the frames {self.first}..{self.stop}"
            )
        if not 0 <= lag <= onset:
            raise ConfigError(f"the lag must be between 0 and the onset {onset} frames, not {lag}")
        for start in {onset, onset - lag}:
            if not (self.pool.solve_kind[start : start + L] == SOLVED).all():
                raise DataConditionError(f"the window at {start} touches an unsolved frame")

    def prepare(self) -> None:
        """Nothing: the one onset is given, so no family's mask is asked and nothing is surveyed."""

    def cut_splits(self) -> list[SplitSpan]:
        """One split: every frame of the file is in train."""
        self._split = np.zeros(len(self.pool.time), np.int8)
        return [SplitSpan(0, self.first, self.stop)]

    def place(self, span: SplitSpan) -> list[Slot[str]]:
        """The one slot, where it was asked for: no random draw."""
        return [self.slot]

    def _task(self, slot: Slot[str]) -> _Task:
        """The design of the slot, on the window the attacker knows: `lag` frames before the episode's."""
        known = slice(slot.onset - self.lag, slot.onset - self.lag + slot.length)
        return _Task(slot.family, known.start, self.pool.vm[known], self.pool.va[known])

    def design(self, span: SplitSpan, slots: list[Slot[str]]) -> None:
        """The one episode designed: it must exist (NoAdmissibleTarget when the family finds no attack). The
        attack found on the known window is set at the episode's own onset, on the true scans."""
        slot, tried = slots[0], self._designs(slots)[0]
        if tried.episode is None:
            raise NoAdmissibleTarget(
                f"{slot.family} finds no attack at onset {slot.onset - self.lag} ({tried.miss})"
            )
        self._keep(span, slots, [replace(tried, episode=replace(tried.episode, onset=slot.onset))])

    def summary(self) -> dict[str, object]:
        """The one episode, as written: its place in the file, its buses, clients and tampered meters."""
        ep, (_, tried) = self._episodes[0], self._tried[0]
        w = slice(ep.onset, ep.onset + ep.length)
        tam = self.att.tampered(self.pool.vm[w], self.pool.va[w], ep.a)
        return {
            "frames": self.frames,
            "onset": ep.onset - self.first,
            "length": ep.length,
            "family": ep.family,
            "buses": ep.buses.tolist(),
            "target_branch": ep.target_branch,
            "clients": ep.clients.tolist(),
            "meters_tampered": int(tam.any(axis=0).sum()),
            "design_seconds": round(tried.seconds, 1),
        }
