"""The New York grid as the source of a timeline: the year of pool states and its steps of the one path.

`NewYorkSource` is a `generation.source.TimelineSource`: `source.generate(path)` runs the steps written
there once for every grid (the file, the walk, the move of an episode with no design). What is this grid's
own is here: `prepare` finds the onsets a family may take (optionally by a survey that designs them all
first), the splits are whole weeks stratified on those onsets, and each placed episode is designed on its
own window of true pool states from a stream keyed by family and onset (`PoolDesigner`; so the result
does not depend on the worker count) as an attack vector a = h(x_false) - h(x_true), so `z - z_benign` is
the attack. The file has the one layout of every timeline (`fdia_graph.schema`); `LAYERS` says what differs from an IEEE
file. `SingleEpisodeSource` is the same grid with one episode placed by hand."""

from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

import numpy as np

from fdia_graph import schema
from fdia_graph.constants.families import AM, AT
from fdia_graph.constants.newyork import BUS_LOCATIONS_FILE, TERRITORIES_FILE, SolveKind
from fdia_graph.engine.attacks.attacker import (
    Attacker,
    generator_band,
    generator_q_capability,
    generator_ramps,
    natural_steps,
)
from fdia_graph.engine.attacks.family import AttackFamily, DesignedEpisode, VectorEpisode
from fdia_graph.engine.attacks.interface import InterfaceFdia
from fdia_graph.engine.attacks.ramp import Ramp
from fdia_graph.engine.core import FdiaGenerator
from fdia_graph.errors import ConfigError, DataConditionError, NoAdmissibleTarget
from fdia_graph.generation.design import Designer, Families
from fdia_graph.generation.nygrid.config import Settings, rng, stream_key
from fdia_graph.generation.nygrid.interfaces import hourly_interfaces
from fdia_graph.generation.nygrid.io import read_pool
from fdia_graph.generation.nygrid.pool import Pool, peak_ratings
from fdia_graph.generation.plan import STRATUM, split_weeks, week_index
from fdia_graph.generation.source import OutPath, TimelineSource
from fdia_graph.models.choices import Split, SplitBy
from fdia_graph.models.config import NewYorkRelease, NewYorkSettings
from fdia_graph.models.frames import LayerSet
from fdia_graph.models.plan import DesignTask, Episode, Planned, Slot, SplitSpan, Tried
from fdia_graph.schema import NEW_YORK_GRAPH_LEGEND, NEW_YORK_TAMPER_RULE, Attr, Static
from fdia_graph.sensing.measure import Measurement
from fdia_graph.sensing.meters import plan_meters
from fdia_graph.systems.case import Case, nygrid_case
from fdia_graph.systems.network import Network
from fdia_graph.systems.ownership import Ownership, assign_clients
from fdia_graph.systems.states import PoolStates

# weeks per stratum of the split when the weight is the surveyed attacks: they sit in about 9 weeks, one
# of them holding 40%, so a 20-week stratum can leave a split one episode; 5 draws 3/1/1 per stratum
HIT_STRATUM = 5
SURVEY_CHUNK = 240  # onsets designed and saved at a time with `survey_cache`
AM_RISE = 16  # Am's rise scans; its window adds as many exit scans
AT_LENGTH = 16  # the slow ramp's window (scans)
RELEASE_LENGTH = {AM: 2 * AM_RISE, AT: AT_LENGTH}  # the release's episode lengths (scans)
# what a New York file carries per frame: PMU currents, the flows at both branch ends, no temporal features
LAYERS = LayerSet(both_ends=True, temporal=False)


class PoolDesigner(Designer[Families]):
    """The design stage on a pool of solved states: each placed episode is designed by its family on
    its own window of true states, from the stream keyed by its family and onset (`config.rng`), so a
    design depends on neither the split, the slot nor the workers, and the survey's can be reused.
    With `lag` frames the window is the one the attacker knows, that many frames earlier."""

    def __init__(self, pool: Pool, net: Network, families: Families, build: Settings, workers: int) -> None:
        super().__init__(workers, dict, families, build.seed)
        self.states, self.solved = PoolStates(pool.vm, pool.va, net), pool.solve_kind == SolveKind.SOLVED
        self.build, self.lag = build, 0

    def task(self, p: Planned[DesignedEpisode]) -> DesignTask | None:
        """The design of `p` to try, on its window of pool states; None when the window touches an
        unsolved frame."""
        family, onset = p.slot.family, p.slot.onset - self.lag
        w = slice(onset, onset + p.slot.length)
        if not self.solved[w].all():
            return None
        return DesignTask(family, onset, self.states[w], stream_key(self.build, f"design/{family}/{onset}"))


@dataclass
class Survey:
    """What a source knows about the onsets before its splits are cut (`NewYorkSource.prepare`).

    `masks` [T] bool by family: the onsets it may take (its `onset_mask`, or after a survey the onsets where
    an attack was found). A survey also leaves its `designs` by (family, onset) for the design stage to
    reuse, the onsets it `tried`, the miss reasons counted per family (`missed`), and the weeks per
    `stratum` of the splits."""

    masks: dict[str, np.ndarray] = field(default_factory=dict)
    designs: dict[tuple[str, int], Tried] = field(default_factory=dict)
    tried: dict[str, np.ndarray] = field(default_factory=dict)
    missed: dict[str, dict[str, int]] = field(default_factory=dict)
    stratum: int = STRATUM

    @classmethod
    def of(cls, families: list[str], frames: int, cands: list[Slot], tried: list[Tried]) -> Survey:
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
                survey.designs[(c.family, c.onset)] = Tried(t.episode, t.seconds)
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


class NewYorkSource(TimelineSource):
    """The New York grid a timeline is generated on: the `pool` of five-minute states, the `attacker` (with
    its `Measurement`), the attack `families` by name ("Am", "At"), the `Ownership` of the buses, the
    timeline's `settings` and the `build` settings whose seed keys every draw. `production` builds the one
    of a release. `generate(path)` writes the timeline; `summary()` then says what was placed and built. A
    family with an `onset_mask` (an overload family waiting for a loaded grid) starts only where it allows."""

    legend, tamper_rule = NEW_YORK_GRAPH_LEGEND, NEW_YORK_TAMPER_RULE
    settings: NewYorkSettings
    designer: PoolDesigner

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
        self.families, self.own, self.build = families, own, build
        # the knobs of the release the source was built for (`production`); none on a source built by hand
        self.release: dict[str, object] = {}
        self.stop = len(pool.time)  # with `first`, the pool frames the file covers: the whole pool
        # the generator every scan is emitted by: the grid, its meter plan and each frame's noise
        net = self.meas.net
        g = FdiaGenerator.from_network(net, self.meas.plan, build.seed, LAYERS)
        designer = PoolDesigner(pool, net, families, build, settings.workers)
        super().__init__(g, PoolStates(pool.vm, pool.va, net), designer, settings, list(families), build.seed)
        self.noise, self._survey = g.noise, Survey()

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

    # ---- the steps of `generate` that are this grid's own ---------------------------------------------

    def out_path(self, out: OutPath | None) -> str:
        """`out`: a New York timeline has no default path."""
        if out is None:
            raise ConfigError("a New York timeline needs the path of its file: source.generate(path)")
        return os.fspath(out)

    def graph(self) -> dict[str, np.ndarray]:
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

    def record(self) -> dict[str, object]:
        """The settings of the timeline (`NewYorkSettings.as_record`) with the knobs of its `release`."""
        return {**self.settings.as_record(), "release": self.release}

    def recorded(self) -> dict[str, object]:
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

    def columns(self) -> dict[str, np.ndarray]:
        """Each frame's split and time, and each episode's target branch."""
        targets = np.array([ep.target_branch for ep in self.episodes], np.int32)
        return {
            **super().columns(),
            schema.TIME: self.pool.time[self.first : self.stop].astype("datetime64[s]").astype(np.int64),
            schema.path(schema.Group.EPISODES, schema.EPISODE_TARGET_BRANCH): targets,
        }

    def prepare(self) -> None:
        """Before the splits are cut: the onsets each waiting family may take (its `onset_mask`). With `survey`,
        every such onset on a grid of `survey` frames is designed here (hours on a year) and the family is held
        to the onsets with an attack: most near-limit hours hold none, so a split could get almost no episode."""
        masks = self._family_masks()
        self._survey = self._surveyed(masks) if self.settings.survey else Survey(masks)
        self.designer.known = self._survey.designs  # a surveyed slot reuses its design

    def split_codes(self) -> np.ndarray:
        """[T] each frame's split: whole weeks stratified on the frames some waiting family may start at
        (after a survey, on the onsets with an attack), or three blocks in time order."""
        if self.settings.split_by == SplitBy.TIME:
            return super().split_codes()
        loaded = np.zeros(self.frames, bool)
        for mask in self._survey.masks.values():
            loaded |= mask
        return split_weeks(
            self.pool.time, self.fractions, loaded, rng(self.build, "split"), self._survey.stratum
        )

    def placement_rng(self) -> np.random.Generator:
        """The build's "place" stream, fresh for each run: one generator over the three splits in turn."""
        return rng(self.build, "place")

    def onset_masks(self) -> dict[str, np.ndarray]:
        """Each waiting family's allowed onsets [T] over the year, or after a survey the onsets where it found
        an attack: `Survey.masks`."""
        return self._survey.masks

    # ---- what was built ------------------------------------------------------------------------------

    def summary(self) -> dict[str, object]:
        """What the last timeline generated holds: frames, per split and family the slots placed, the moves
        made, the episodes built and the slots given up, and the seconds spent. `missed` and
        `unsolved_attempts` count design attempts, not slots (a slot is tried again at every onset it moves
        to): why an attempt found no attack, and the attempts not made because the window touched an
        unsolved frame. Each such attempt moved its slot or, after the last move, gave it up."""
        return {
            "frames": self.frames,
            "families": list(self.families),
            "splits": Split.values(),
            "placed": (self.rec.requested - self.rec.dropped).tolist(),
            "moved": self.rec.redraws.tolist(),
            "built": self.rec.built.tolist(),
            "given_up": (self.rec.requested - self.rec.dropped - self.rec.built).tolist(),
            "unsolved_attempts": sum(not t.designed for _, t in self.designer.log),
            "missed": {f: self._missed(f) for f in self.families},
            "survey": self._survey.counts(self.split),
            "survey_missed": self._survey.missed,  # why the surveyed onsets without an attack have none
            "attacked_frames": int(sum(ep.length for ep in self.episodes)),
            "design_seconds": round(float(sum(t.seconds for _, t in self.designer.log)), 1),
            "design_seconds_median": {f: self._median_seconds(f) for f in self.families},
            "seconds": round(self.seconds, 1),
        }

    def _missed(self, family: str) -> dict[str, int]:
        """Why the design attempts of `family` found no attack: each reason's count."""
        log = self.designer.log
        return dict(Counter(t.miss for s, t in log if s.family == family and t.miss is not None))

    def _median_seconds(self, family: str) -> float | None:
        """The median seconds one design of `family` took; None when none was designed."""
        secs = [t.seconds for s, t in self.designer.log if s.family == family and t.seconds > 0]
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
        tried: list[Tried] = []
        chunk = SURVEY_CHUNK if self.settings.survey_cache else max(len(cands), 1)
        for i in range(0, len(cands), chunk):
            tried += self._survey_chunk(cands, i, chunk)
            done = sum(t.episode is not None for t in tried)
            self._log(f"survey {min(i + chunk, len(cands))}/{len(cands)}: {done} with an attack")
        return Survey.of(list(masks), len(weeks), cands, tried)

    def _survey_chunk(self, cands: list[Slot], i: int, chunk: int) -> list[Tried]:
        """The designs of the surveyed onsets `cands[i : i + chunk]`. With `survey_cache`, each chunk is
        saved on its own (episodes, seconds, miss reasons), so a stopped build resumes. The `_v2` in a chunk's
        name is the pickled record's version (`models.plan.Episode`)."""
        cache = self.settings.survey_cache
        part = None if cache is None else Path(cache) / f"{i:06d}_{len(cands)}_v2.pkl"
        if part is not None and part.exists():
            return [self._from_row(*row) for row in zip(*pickle.loads(part.read_bytes()))]
        tried = self.designer.tried([Planned(c) for c in cands[i : i + chunk]])
        if part is not None:
            self._save(part, tried)
        return tried

    @staticmethod
    def _save(part: Path, tried: list[Tried]) -> None:
        """Save a survey chunk's designs: the episodes (`models.plan.Episode`), seconds and miss reasons."""
        part.parent.mkdir(parents=True, exist_ok=True)
        found = [None if t.episode is None else cast(VectorEpisode, t.episode).episode for t in tried]
        part.write_bytes(pickle.dumps((found, [t.seconds for t in tried], [t.miss for t in tried])))

    @staticmethod
    def _from_row(episode: Episode | None, seconds: float, miss: str | None) -> Tried:
        """One saved design of a survey chunk, as the design stage holds it."""
        return Tried(None if episode is None else VectorEpisode(episode), seconds, miss)


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
        settings = replace(source.settings, workers=1, survey=0, split_by=SplitBy.TIME)
        super().__init__(source.pool, source.att, families, source.own, settings, source.build)
        self.release = source.release
        self.slot = Slot(onset, family, self.lengths()[family])
        self.lag = self.designer.lag = lag
        self.first = onset if lo is None else lo
        self.stop = onset + self.slot.length if hi is None else hi
        w = slice(self.first, self.stop)
        self.X = PoolStates(self.pool.vm[w], self.pool.va[w], self.meas.net)
        self.fractions = (1.0, 0.0, 0.0)  # every frame is in train: one block
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

    def prepare(self) -> None:
        """Nothing: the one onset is given, so no family's mask is asked and nothing is surveyed."""

    def cut_splits(self) -> list[SplitSpan]:
        """One split to plan: train."""
        return super().cut_splits()[:1]

    def place(self, span: SplitSpan) -> list[Slot]:
        """The one slot, where it was asked for: no random draw."""
        self.rec.requested[span.code, self.rec.col(self.slot.family)] += 1
        return [self.slot]

    def design(self, span: SplitSpan, slots: list[Slot]) -> None:
        """The one episode designed: it must exist (NoAdmissibleTarget when the family finds no attack). The
        attack found on the known window is set at the episode's own onset, on the true scans; the walk gets
        it at its frame of the file."""
        slot, tried = slots[0], self.designer.tried([Planned(slots[0])])[0]
        if tried.episode is None:
            raise NoAdmissibleTarget(
                f"{slot.family} finds no attack at onset {slot.onset - self.lag} ({tried.miss})"
            )
        self.designer.log.append((slot, tried))
        self._pending = [Planned(slot.at(slot.onset - self.first), tried.episode)]

    def summary(self) -> dict[str, object]:
        """The one episode, as written: its place in the file, its buses, clients and tampered meters."""
        ep, (_, tried) = self.episodes[0], self.designer.log[0]
        w = slice(self.first + ep.onset, self.first + ep.onset + ep.length)
        tam = self.att.tampered(self.pool.vm[w], self.pool.va[w], ep.a)
        return {
            "frames": self.frames,
            "onset": ep.onset,
            "length": ep.length,
            "family": ep.family,
            "buses": ep.buses.tolist(),
            "target_branch": ep.target_branch,
            "clients": ep.clients.tolist(),
            "meters_tampered": int(tam.any(axis=0).sum()),
            "design_seconds": round(tried.seconds, 1),
        }
