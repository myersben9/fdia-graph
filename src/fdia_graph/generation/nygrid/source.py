"""The New York grid as the source of a timeline: the year of pool states and its steps of the one path.

`NewYorkSource` is a `generation.source.TimelineSource`, so `source.generate(path)` takes the same steps in
the same order as an IEEE timeline. Its own steps: `prepare` finds the onsets a family may take (optionally
by a survey that designs them all first), `cut_splits` cuts whole weeks stratified on those onsets, each
placed episode is designed on its own window of true pool states from a stream keyed by family and onset
(so the result does not depend on the worker count), and `walk` writes each frame as a noisy benign scan
plus the episode's attack vector a = h(x_false) - h(x_true), so `z - z_benign` is exactly the attack. The
file has the one layout of every timeline (`fdia_graph.schema`); `LAYERS` says what differs from an IEEE
file. `SingleEpisodeSource` is the same grid with one episode placed by hand."""

from __future__ import annotations

import json
import os
import pickle
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from fdia_graph import schema
from fdia_graph.constants.families import AM, AT, FAMILY_CODE
from fdia_graph.constants.newyork import BUS_LOCATIONS_FILE, TERRITORIES_FILE, SolveKind
from fdia_graph.engine.attacks.attacker import (
    Attacker,
    generator_band,
    generator_q_capability,
    generator_ramps,
    natural_steps,
)
from fdia_graph.engine.attacks.family import AttackFamily
from fdia_graph.engine.attacks.interface import InterfaceFdia
from fdia_graph.engine.attacks.ramp import Ramp
from fdia_graph.errors import ConfigError, DataConditionError, NoAdmissibleTarget
from fdia_graph.generation.design import DesignPool
from fdia_graph.generation.nygrid.config import Settings, rng
from fdia_graph.generation.nygrid.interfaces import hourly_interfaces
from fdia_graph.generation.nygrid.io import read_pool
from fdia_graph.generation.nygrid.pool import Pool, peak_ratings
from fdia_graph.generation.plan import STRATUM, split_bounds, split_column, split_weeks, week_index
from fdia_graph.generation.source import OutPath, TimelineSource
from fdia_graph.generation.write import TimelineWriter
from fdia_graph.models.choices import Split, SplitBy
from fdia_graph.models.config import NewYorkRelease, NewYorkSettings
from fdia_graph.models.frames import Frame, LayerSet
from fdia_graph.models.plan import Episode, Placement, Slot, SplitSpan
from fdia_graph.schema import NEW_YORK_GRAPH_LEGEND, NEW_YORK_TAMPER_RULE, Attr, Static
from fdia_graph.sensing.measure import Measurement
from fdia_graph.sensing.meters import plan_meters
from fdia_graph.sensing.noise import NoiseModel
from fdia_graph.systems.case import Case, nygrid_case
from fdia_graph.systems.network import Network
from fdia_graph.systems.ownership import Ownership, assign_clients

# weeks per stratum of the split when the weight is the surveyed attacks: they sit in about 9 weeks, one
# of them holding 40%, so a 20-week stratum can leave a split one episode; 5 draws 3/1/1 per stratum
HIT_STRATUM = 5
SURVEY_CHUNK = 240  # onsets designed and saved at a time with `survey_cache`
LOG_EVERY = 50  # designs between two progress lines of a verbose build
AM_RISE = 16  # Am's rise scans; its window adds as many exit scans
AT_LENGTH = 16  # the slow ramp's window (scans)
RELEASE_LENGTH = {AM: 2 * AM_RISE, AT: AT_LENGTH}  # the release's episode lengths (scans)
# what a New York file carries per frame: PMU currents, the flows at both branch ends, no temporal features
LAYERS = LayerSet(both_ends=True, temporal=False)


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
    design = state.families[task.family].design_explained
    ep, miss = design((task.onset, task.vm, task.va), rng(state.build, f"design/{task.family}/{task.onset}"))
    return _Tried(ep, time.perf_counter() - start, miss)


@dataclass
class Survey:
    """What a source knows about the onsets before its splits are cut (`NewYorkSource.prepare`).

    `masks` [T] bool by family: the onsets it may take (its `onset_mask`, or after a survey the onsets where
    an attack was found). A survey also leaves its `designs` by (family, onset) for the design stage to
    reuse, the onsets it `tried`, the miss reasons counted per family (`missed`), and the weeks per
    `stratum` of the splits."""

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
            for code, name in enumerate(Split.values()):
                here = split == code
                out[f"{name}/{f}"] = [int((self.tried[f] & here).sum()), int((self.masks[f] & here).sum())]
        return out


def _ownership(case: Case, root: Path) -> Ownership:
    """One client per utility, from the territories `fetch_inputs` left under `root/data`."""
    terr = root / "data" / "territories"
    with open(terr / TERRITORIES_FILE, encoding="utf-8") as f:
        return assign_clients(case, json.load(f), terr / BUS_LOCATIONS_FILE)


def _interface_family(att: Attacker, pool: Pool, release: NewYorkRelease, nygrid: Path) -> InterfaceFdia:
    """The release's Am: the OPF attacker against the NYISO interfaces, with the units' output and every
    realism bound."""
    case = att.meas.net.case
    if pool.unit_p is None:
        raise DataConditionError(
            "the pool must carry each unit's output (built by pipelines/newyork/run_pool.py)"
        )
    # Am's exit: each bus's natural five-minute step by hour over the whole pool (what the attacker knows),
    # generator buses bounded by their units' five-minute ramp (nuclear 0, hydro and imports inf)
    steps = natural_steps(pool.time, pool.p, pool.solve_kind == SolveKind.SOLVED)
    steps[:, att.gen_bus] = generator_ramps(case)[att.gen_bus]
    q_lo, q_hi = generator_q_capability(case)
    gen_output = np.zeros((len(pool.time), len(case.bus_id)))  # the units' output at each bus
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
    if release.load_cap is not None:  # otherwise the family keeps its own cap
        fam.load_cap = release.load_cap
    return fam


class NewYorkSource(TimelineSource[str, Episode]):
    """The New York grid a timeline is generated on: the `pool` of five-minute states, the `attacker` (with
    its `Measurement`), the attack `families` by name ("Am", "At"), the `Ownership` of the buses, the
    timeline's `settings` and the `build` settings whose seed keys every draw. `production` builds the one
    of a release. `generate(path)` writes the timeline; `summary()` then says what was placed and built. A
    family with an `onset_mask` (an overload family waiting for a loaded grid) starts only where it allows."""

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
        # the knobs of the release the source was built for (`production`); none on a source built by hand
        self.release: dict[str, object] = {}
        # the pool frames the file covers, [first, stop): the whole pool
        self.first, self.stop = 0, len(pool.time)
        case, plan = self.meas.net.case, self.meas.plan
        self._size = (len(case.bus_id), len(case.f))  # buses, branches
        self._meter_m = plan.scatter(np.ones(len(plan.kind), np.uint8), *self._size)  # the meter masks
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
        cls, pool_path: str | Path, release: NewYorkRelease = NewYorkRelease(), build: Settings = Settings()
    ) -> NewYorkSource:
        """The source of a release: the case, clients, meters and the pool at `pool_path` (relative to
        `release.root`, whose `data/` holds `fetch_inputs`'s files). Am targets the NYISO interfaces from the
        OPF attacker (32 scans), At is the slow ramp (16 scans). The attacker's knobs are `NewYorkRelease`'s."""
        root = Path(release.root)
        case = nygrid_case()
        own = _ownership(case, root)
        pool = read_pool(root / pool_path)
        case = peak_ratings(case, pool)  # production ratings over the whole pool
        net = Network(case)
        meas = Measurement(net, plan_meters(net, own, rng(build, "meters")))
        # a voltage margin widens the attacker's limits on its own copy of the case
        att_net = Network(case.with_voltage_margin(release.v_margin)) if release.v_margin else net
        # True keeps the interchange true
        att = Attacker(att_net, meas, hold_external=release.hold_external)
        families: dict[str, AttackFamily] = {}
        for name in release.families:
            if name == AM:
                families[name] = _interface_family(att, pool, release, root / build.nygrid_dir / "Data")
            else:
                families[name] = Ramp(att)
        settings = NewYorkSettings(
            length={f: RELEASE_LENGTH[f] for f in families},
            workers=release.workers,
            verbose=True,
            survey=release.survey,
            survey_cache=release.survey_cache,
        )
        source = cls(pool, att, families, own, settings, build)
        source.release = release.as_record()
        return source

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
        """Start the file with its layers and the static graph, and a fresh run: its records, its random
        streams and its design processes."""
        self._reset()
        self._designers = DesignPool(self.settings.workers, _designing, _Designing(self.families, self.build))
        self._writer = TimelineWriter(path, (self.frames, *self._size), self._clean, LAYERS)
        case = self.meas.net.case
        edges = np.vstack([case.f, case.t]).astype(np.int32)  # the IEEE files' type, on every platform
        self._writer.write_graph(edges, self._graph(), NEW_YORK_GRAPH_LEGEND)

    def _graph(self) -> dict[str, np.ndarray]:
        """The static graph a file carries, by `Static` name: the case's (`Case.static_graph`: the branch
        model and bus shunts) with the line ratings (the case file's, or the pool's peak flow on an unrated
        branch), bus facts and owning client. It is the true case, not the attacker's widened copy. The units,
        HVDC links, names and places are not written: they are the case file's. Integers are int32 on every
        platform."""
        case = self.meas.net.case
        has_gen = np.zeros(len(case.bus_id), np.uint8)
        has_gen[case.unit_bus] = 1
        return {
            **case.static_graph(),
            Static.EDGE_RATING: case.rating,
            Static.BUS_TYPE: case.bus_type.astype(np.int32),
            Static.BUS_VMIN: case.vmin,
            Static.BUS_VMAX: case.vmax,
            Static.BUS_BASE_KV: case.kv,
            Static.BUS_HAS_GEN: has_gen,
            Static.BUS_BASE_PD: case.pd,
            Static.BUS_BASE_QD: case.qd,
            Static.BUS_CLIENT: self.own.bus_client.astype(np.int32),
        }

    def _clean(self, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        """The noiseless truth of the file's frames a..b-1: the solved state as [V, P, Q, theta] with the
        injections it gives, and the exact flows at the metered branch ends (zero at the others)."""
        w = slice(self.first + a, min(self.first + b, self.stop))
        vm, va, net = self.pool.vm[w], self.pool.va[w], self.meas.net
        with np.errstate(invalid="ignore"):  # an unsolved frame keeps its place with NaN
            p, q = net.injections(vm, va)
            flows = np.stack(net.flows(vm, va), axis=-1)
        node = np.stack([vm, p, q, va], axis=-1)
        return node.astype(np.float32), np.where(self._meter_m.edge, flows, 0.0).astype(np.float32)

    def record(self) -> dict[str, object]:
        """The settings of the timeline (`NewYorkSettings.as_record`) with the knobs of its `release`."""
        return {**self.settings.as_record(), "release": self.release}

    def _split_frac(self) -> Sequence[float]:
        """The share of the frames each split was asked to hold."""
        return self.settings.split.fractions

    def _recorded(self) -> dict[str, object]:
        """What a file records beside its header: the clients, the form of its PMU currents (what reads
        differently from an IEEE file) and, when episodes are placed by share, the share asked for."""
        asked = self.record().get("attacked_frac")
        return {
            **({} if asked is None else {Attr.TARGET_ATTACKED_FRAC: asked}),
            Attr.CURRENT_FEAT: schema.CURRENT_FEAT_LEGEND[schema.CURRENT_POLAR],
            Attr.CURRENT_UNITS: schema.CURRENT_UNITS_LEGEND[schema.CURRENT_POLAR],
            Attr.CURRENT_FORM: schema.CURRENT_POLAR,
            Attr.CLIENT_NAMES: ",".join(str(name) for name in self.own.names),
        }

    def prepare(self) -> None:
        """Before the splits are cut: the onsets each waiting family may take (its `onset_mask`). With `survey`,
        every such onset on a grid of `survey` frames is designed here (hours on a year) and the family is held
        to the onsets with an attack: most near-limit hours hold none, so a split could get almost no episode."""
        masks = self._family_masks()
        self._survey = self._surveyed(masks) if self.settings.survey else Survey(masks)

    def cut_splits(self) -> list[SplitSpan]:
        """The three splits, each with its frames: whole weeks stratified on the frames some waiting family
        may start at (after a survey, on the onsets with an attack), or three blocks in time order."""
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
        return [self._span(code) for code in range(len(Split))]

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
        """Each waiting family's allowed onsets [T] over the year, or after a survey the onsets where it found
        an attack: `Survey.masks`."""
        return self._survey.masks

    def design(self, span: SplitSpan, slots: list[Slot[str]]) -> None:
        """Every placed episode of the split designed by its family, the survey's designs reused. A slot
        with no attack keeps no episode: its frames stay benign (`summary` counts why)."""
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
        """Write the frames from the last one written up to `until`, a block at a time. The noise stream runs
        over the frames in time order, so the scans are the same wherever a stretch or a block ends."""
        for t0 in range(self._cursor, until, self.settings.block):
            self._write_block(t0, min(until, t0 + self.settings.block))
        self._cursor = max(self._cursor, until)

    def finish(self) -> None:
        """After the walk: the columns (each frame's split and time, each episode's target branch), the
        episode table, onsets counted from the file's first frame, and the attributes."""
        writer = self.writer
        for ep in self._episodes:
            writer.add_episode(replace(ep, onset=ep.onset - self.first))
        w = slice(self.first, self.stop)
        targets = np.array([ep.target_branch for ep in self._episodes], np.int32)
        columns = {
            schema.SPLIT: self._split[w],
            schema.TIME: self.pool.time[w].astype("datetime64[s]").astype(np.int64),
            schema.path(schema.Group.EPISODES, schema.EPISODE_TARGET_BRANCH): targets,
        }
        base_mva = float(self.meas.net.case.base_mva)
        header, knobs = writer.header(base_mva, self.build.seed), self._recorded()
        writer.finish(
            columns, header, knobs, self.record(), self.rec, self._split_frac(), NEW_YORK_TAMPER_RULE
        )
        self._seconds = time.perf_counter() - self._started

    def close(self, finished: bool) -> None:
        """Close the file and stop the design processes."""
        super().close(finished)
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
            "splits": Split.values(),
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
        """Each waiting family's allowed onsets [T] over the year (`onset_mask`); a family that may start
        anywhere has none."""
        masks = {}
        for f, fam in self.families.items():
            mask = fam.onset_mask(self.pool.vm, self.pool.va, self.lengths()[f])
            if mask is not None:
                masks[f] = mask
        return masks

    def _onsets_ok(self, blocks: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """[T] per family: the onsets whose window lies in one block (`blocks` [T], a code per frame such
        as the week), on solved frames, where the family allows."""
        T, solved = len(blocks), self.pool.solve_kind == SolveKind.SOLVED
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
        """The `Survey` of every allowed onset of each waiting family whose window lies in one week, on a grid
        of `survey` frames, each designed."""
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
        saved on its own (episodes, seconds, miss reasons), so a stopped build resumes. The `_v2` in a chunk's
        name is the pickled record's version (`models.plan.Episode`)."""
        cache = self.settings.survey_cache
        part = None if cache is None else Path(cache) / f"{i:06d}_{len(cands)}_v2.pkl"
        if part is not None and part.exists():
            return [_Tried(*row) for row in zip(*pickle.loads(part.read_bytes()))]
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
        if not (self.pool.solve_kind[w] == SolveKind.SOLVED).all():
            return None
        return _Task(slot.family, slot.onset, self.pool.vm[w], self.pool.va[w])

    def _designs(self, slots: list[Slot[str]]) -> list[_Tried]:
        """What the design of each slot came to: an episode, or none where the family finds no attack or the
        window touches an unsolved frame (then it is not designed). A surveyed slot reuses its design."""
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
        """Write the pool frames t0..t1 (file frames count from `first`): the noisy benign scans, plus the
        attack vector and labels of every overlapping episode. An episode's row in the file is its place in
        onset order. A bus is labelled attacked when a meter at it is tampered beyond its noise."""
        pool, meas, plan, masks = self.pool, self.meas, self.meas.plan, self._meter_m
        M, (N, E) = len(plan.kind), self._size
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
            family[a - t0 : b - t0] = FAMILY_CODE[ep.family]
            episode[a - t0 : b - t0] = e
        at_bus = np.zeros((M, N), np.int32)
        at_bus[np.arange(M), plan.bus] = 1  # a meter's bus: a tampered meter marks its bus attacked
        attacked = (meter_tampered.astype(np.int32) @ at_bus) > 0
        seen, benign, tampered = (plan.scatter(a, N, E) for a in (z, z_benign, meter_tampered))
        none = np.zeros(0, np.int32)  # the attack vectors are not stored: observed minus benign is the attack
        for i, (fid, sid) in enumerate(zip(family.tolist(), episode.tolist())):
            frame = Frame(
                node_x=seen.node[i],
                node_m=masks.node,
                edge_x=seen.edge[i],
                edge_m=masks.edge,
                y=attacked[i],
                stealthy=int(fid != 0),
                mag_bus=none,
                mag=none,
                benign_node_x=benign.node[i],
                benign_edge_x=benign.edge[i],
                tamper=(tampered.node[i], tampered.edge[i]),
                i_x=seen.current[i],
                i_m=masks.current,
                benign_i_x=benign.current[i],
                i_tamper=tampered.current[i],
            )
            self.writer.store(t0 - self.first + i, fid, frame, sid)
        self._log(f"wrote frames {t0}..{t1}")


class SingleEpisodeSource(NewYorkSource):
    """`source` with one episode placed by hand, for studying one attack: `family` designed at pool frame
    `onset`, the file covering pool frames lo..hi (default the window), every frame in train. `using`
    designs it in place of the source's family of that name. With `lag` (frames) the attacker knows each
    scan as it was `lag` frames earlier: the attack is designed on the window at `onset - lag` and added to
    the true scans from `onset`. Building raises on an unsolved frame; `generate` raises
    NoAdmissibleTarget, and leaves no file, when the family finds no attack."""

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
        self.noise, self.release = source.noise, source.release
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
            if not (self.pool.solve_kind[start : start + L] == SolveKind.SOLVED).all():
                raise DataConditionError(f"the window at {start} touches an unsolved frame")

    def record(self) -> dict[str, object]:
        """The source's record without what places episodes by share (the attacked share, the splits, the
        survey), and the one episode: its family, onset, lag and the pool frames `lo`..`hi` of the file."""
        placed = ("attacked_frac", "split", "split_by", "survey")
        record = {k: v for k, v in super().record().items() if k not in placed}
        frames = {"onset": self.slot.onset, "lag": self.lag, "lo": self.first, "hi": self.stop}
        return {**record, "episode": {"family": self.slot.family, **frames}}

    def _split_frac(self) -> Sequence[float]:
        """Every frame is in train."""
        return (1.0, 0.0, 0.0)

    def prepare(self) -> None:
        """Nothing: the one onset is given, so no family's mask is asked and nothing is surveyed."""

    def cut_splits(self) -> list[SplitSpan]:
        """One split: every frame of the file is in train."""
        self._split = np.zeros(len(self.pool.time), np.int8)
        return [SplitSpan(0, self.first, self.stop)]

    def place(self, span: SplitSpan) -> list[Slot[str]]:
        """The one slot, where it was asked for: no random draw."""
        self.rec.requested[span.code, self.rec.col(self.slot.family)] += 1
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
