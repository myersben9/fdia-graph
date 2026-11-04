"""The timeline: the year of pool states as noisy scans, with attack episodes placed at random and labelled.

Each frame's benign scan is the exact reading of the true state with accuracy-class noise
(`NoiseModel.sample`). An attacked frame is that same noisy scan plus the episode's attack vector
a = h(x_false) - h(x_true), never a fresh scan of the false state, so `z - z_benign` is exactly the
attack. The splits are whole weeks stratified on the loaded frames (`split_weeks`, N41; with a survey, on
the onsets where an attack exists, surveyed by week first, N46), and episodes
are placed per split at uniform random onsets among those their family allows (`place_episodes`; an
overload family that waits for a loaded grid allows only near-limit onsets, N36), and every one is
designed on its own window of true pool states by its family; a window the family finds no stealthy
attack for, or one that touches an unsolved pool frame, stays benign. With `survey` (N42), a waiting
family's allowed onsets on a grid of that many frames are all designed first, and its episodes are
placed uniformly among those with an attack, reusing their designs, so the episodes sit uniformly
among the onsets where an attack exists. The labels come from the meters
the attack moves beyond their noise (`Attacker.tampered`): `meter_tampered [T, M]`, and `bus_attacked
[T, N]` true at a bus when a meter at that bus is tampered, so a client's labels follow from its own
meters alone. Designs are independent of each other, so they may run in worker processes; each draws
from its own generator keyed by family and onset, so the result does not depend on the worker count.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from fdia_graph.errors import ConfigError, DataConditionError, NoAdmissibleTarget
from fdia_graph.nyfed.attacks.attacker import Attacker
from fdia_graph.nyfed.attacks.episode import AttackFamily, Episode
from fdia_graph.nyfed.config import Settings, rng
from fdia_graph.nyfed.grid.clients import Ownership
from fdia_graph.nyfed.io import FAMILIES, SPLITS, TimelineWriter
from fdia_graph.nyfed.scenario.pool import SOLVED, Pool
from fdia_graph.nyfed.sensing.measure import Measurement
from fdia_graph.nyfed.sensing.noise import NoiseModel
from fdia_graph.nyfed.timeline.schedule import (
    HIT_STRATUM,
    Slot,
    place_episodes,
    split_frames,
    split_weeks,
    week_index,
)

SURVEY_CHUNK = 240  # onsets designed and saved at a time with `survey_cache`
_WORKER: dict = {}  # a worker process's families and settings, set once by `_init`


def _init(families: dict[str, AttackFamily], settings: Settings) -> None:
    _WORKER.update(families=families, settings=settings)


def _design(
    task: tuple[int, str, int, np.ndarray, np.ndarray],
) -> tuple[int, Episode | None, float, str | None]:
    """One episode design in a worker: (slot index, the episode or None, seconds, why there is none: one of
    `MISSES`, None with an episode)."""
    i, family, onset, vm, va = task
    start = time.perf_counter()
    ep, miss = _WORKER["families"][family].design_explained(
        (onset, vm, va), rng(_WORKER["settings"], f"design/{family}/{onset}")
    )
    return i, ep, time.perf_counter() - start, miss


class TimelineBuilder:
    """Holds the pool, the `Measurement`, the `NoiseModel`, the `Attacker` and the families (by name,
    "Am", "Ap" and "At"), with the `Ownership` and the `Settings` whose seed keys every draw. Settings of the
    timeline: `length` frames per episode, one number for every family or `{family: frames}` (16
    five-minute scans as [1] for Am and At, 96 for Ap, Ben 2026-10-27), `attacked_frac` the share of
    each split's frames placed under an episode, `split_frac` the train, val and test shares of the
    year, `split_by` "week" (whole weeks stratified on the loaded frames, N41) or "time" (three
    contiguous blocks, before N41), `workers` processes for the designs (1 runs them here), `block`
    frames written at a time. A family with an `onset_mask` (an overload family waiting for a loaded
    grid, N36) has its episodes start only where the mask allows. `survey` (frames, N42; 0 off) designs
    every such family's allowed onsets on that grid first and places its episodes only among those
    with an attack, saving each chunk of designs under `survey_cache` when given so a stopped build
    resumes."""

    def __init__(
        self,
        pool: Pool,
        meas: Measurement,
        noise: NoiseModel,
        attacker: Attacker,
        families: dict[str, AttackFamily],
        own: Ownership,
        settings: Settings = Settings(),
        *,
        length: int | dict[str, int] = 16,
        attacked_frac: float = 0.2,
        split_frac: tuple[float, float, float] = (0.7, 0.15, 0.15),
        split_by: str = "week",
        workers: int = 1,
        block: int = 2016,
        verbose: bool = False,
        survey: int = 0,
        survey_cache: str | Path | None = None,
    ):
        if split_by not in ("week", "time"):
            raise ConfigError(f"split_by must be 'week' or 'time', not {split_by!r}")
        self.split_by = split_by
        self.pool, self.meas, self.noise, self.att = pool, meas, noise, attacker
        self.families, self.own, self.settings = families, own, settings
        self.length, self.attacked_frac, self.split_frac = length, attacked_frac, split_frac
        self.workers, self.block, self.verbose, self.survey = workers, block, verbose, survey
        self.survey_cache = None if survey_cache is None else Path(survey_cache)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def episode_lengths(self) -> dict[str, int]:
        """The episode length in frames of each family the builder holds."""
        return {f: self.length[f] if isinstance(self.length, dict) else self.length for f in self.families}

    def _masks(self) -> dict[str, np.ndarray]:
        """Each waiting family's allowed onsets [T] over the year (`onset_mask`, N36)."""
        masks = {}
        for f, fam in self.families.items():
            onset_mask = getattr(fam, "onset_mask", None)  # only the waiting families have one
            mask = (
                onset_mask(self.pool.vm, self.pool.va, self.episode_lengths()[f])
                if onset_mask is not None
                else None
            )
            if mask is not None:
                masks[f] = mask
        return masks

    def _split(self, masks: dict[str, np.ndarray], stratum: int | None = None) -> np.ndarray:
        """The split code of every frame: whole weeks stratified on the frames some waiting family may
        start at (N41; after a survey, the onsets with an attack, in strata of `stratum` weeks, N46), or
        three contiguous blocks."""
        T = len(self.pool.time)
        if self.split_by == "time":
            return split_frames(T, self.split_frac)
        loaded = np.zeros(T, bool)
        for mask in masks.values():
            loaded |= mask
        kw = {} if stratum is None else {"stratum": stratum}
        return split_weeks(self.pool.time, self.split_frac, loaded, rng(self.settings, "split"), **kw)

    def _slots(self, split: np.ndarray, masks: dict[str, np.ndarray]) -> list[Slot]:
        """Every split's episodes, onsets in timeline frames, placed inside one contiguous run of the
        split's frames (a window never crosses a week block) and where its family allows (one generator)."""
        g = rng(self.settings, "place")
        lengths = self.episode_lengths()
        slots = []
        for code in range(len(SPLITS)):
            where = np.flatnonzero(split == code)
            if not len(where):
                continue
            allowed = {}
            for f, L in lengths.items():
                ok = np.zeros(len(where), bool)
                if len(where) >= L:  # the window's frames are consecutive in the year
                    ok[: len(where) - L + 1] = where[L - 1 :] - where[: len(where) - L + 1] == L - 1
                if f in masks:
                    ok &= masks[f][where]
                allowed[f] = ok
            slots += [
                Slot(int(where[s.onset]), s.family, s.length)
                for s in place_episodes(len(where), lengths, self.attacked_frac, g, allowed)
            ]
        return slots

    def _onsets_ok(self, blocks: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """[T] per family: the onsets whose window lies in one block (`blocks` [T], a code per frame such
        as the week), on solved frames, where the family allows."""
        T, solved = len(blocks), self.pool.solve_kind == SOLVED
        run = np.r_[0, np.cumsum(blocks[1:] != blocks[:-1])]  # [T] which run of one block a frame is in
        unsolved = np.r_[0, np.cumsum(~solved)]
        ok = {}
        for f, L in self.episode_lengths().items():
            a = np.zeros(T, bool)
            n = T - L + 1
            if n > 0:
                a[:n] = (run[L - 1 :] == run[:n]) & (unsolved[L:] - unsolved[:n] == 0)
            ok[f] = a & masks[f] if f in masks else a
        return ok

    def _survey(self, weeks: np.ndarray, masks: dict[str, np.ndarray]):
        """N42: every allowed onset of each waiting family whose window lies in one week (`weeks` [T]), on
        a grid of `survey` frames, designed. Returns `{family: [T] bool}` the onsets with an attack,
        `{(family, onset): (Episode, seconds)}` their designs, the onsets tried per family, and why the surveyed
        onsets without an attack have none, counted per family (chunks saved before the reasons were kept add
        none)."""
        ok = self._onsets_ok(weeks, masks)
        grid = np.arange(len(weeks)) % self.survey == 0
        lengths = self.episode_lengths()
        cands = [Slot(int(o), f, lengths[f]) for f in masks for o in np.flatnonzero(ok[f] & grid)]
        self._log(f"survey: {len(cands)} onsets on a {self.survey}-frame grid")
        eps, secs, misses = [], [], []
        chunk = SURVEY_CHUNK if self.survey_cache else max(len(cands), 1)
        if self.survey_cache:
            self.survey_cache.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(cands), chunk):
            # with `survey_cache`, each chunk of designs is saved on its own, so a stopped build resumes
            part = None if self.survey_cache is None else self.survey_cache / f"{i:06d}_{len(cands)}.pkl"
            if part is not None and part.exists():
                saved = pickle.loads(part.read_bytes())
                e, t = saved[0], saved[1]
                m = saved[2] if len(saved) > 2 else [None] * len(e)
            else:
                e, t, _, m = self._designs(cands[i : i + chunk])
                if part is not None:
                    part.write_bytes(pickle.dumps((e, t, m)))
            eps, secs, misses = eps + e, secs + t, misses + m
            self._log(
                f"survey {min(i + chunk, len(cands))}/{len(cands)}: {sum(x is not None for x in eps)} with an attack"
            )
        found = {f: np.zeros(len(weeks), bool) for f in masks}
        tried = {f: np.zeros(len(weeks), bool) for f in masks}
        cache = {}
        for c, ep, sec in zip(cands, eps, secs):
            tried[c.family][c.onset] = True
            if ep is not None:
                found[c.family][c.onset] = True
                cache[(c.family, c.onset)] = (ep, sec)
        missed = {f: dict(Counter(m for c, m in zip(cands, misses) if c.family == f and m)) for f in masks}
        return found, cache, tried, missed

    def _designs(self, slots: list[Slot]) -> tuple[list[Episode | None], list[float], int, list[str | None]]:
        """The episode of each slot, None where its window touches an unsolved frame or the family finds
        no attack; the design seconds; the count of slots skipped for unsolved frames; and why each designed
        slot has no attack (one of `MISSES`, None with an attack or when skipped)."""
        pool, solved = self.pool, self.pool.solve_kind == SOLVED
        tasks, skipped = [], 0
        for i, s in enumerate(slots):
            if not solved[s.onset : s.onset + s.length].all():
                skipped += 1
                continue
            w = slice(s.onset, s.onset + s.length)
            tasks.append((i, s.family, s.onset, pool.vm[w], pool.va[w]))
        tasks.sort(key=lambda t: t[1] == "At")  # the slow overload families first, so workers finish together
        out: list[Episode | None] = [None] * len(slots)
        secs = [0.0] * len(slots)
        misses: list[str | None] = [None] * len(slots)
        if self.workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(self.workers, initializer=_init, initargs=(self.families, self.settings)) as p:
                results = p.imap_unordered(_design, tasks, chunksize=1)
                for k, (i, ep, sec, miss) in enumerate(results, 1):
                    out[i], secs[i], misses[i] = ep, sec, miss
                    if k % 50 == 0 or k == len(tasks):
                        self._log(f"designed {k}/{len(tasks)}")
        else:
            _init(self.families, self.settings)
            for k, task in enumerate(tasks, 1):
                i, ep, sec, miss = _design(task)
                out[i], secs[i], misses[i] = ep, sec, miss
                if k % 50 == 0 or k == len(tasks):
                    self._log(f"designed {k}/{len(tasks)}")
        return out, secs, skipped, misses

    def _write(
        self, path, episodes: list[Episode], split: np.ndarray, lo: int = 0, hi: int | None = None
    ) -> None:
        """Write the pool frames lo..hi with `episodes` (onsets in pool frames, sorted) applied and labelled,
        and `split [T]` per frame; the file's frames and episode onsets count from lo."""
        pool, meas, plan = self.pool, self.meas, self.meas.plan
        hi = len(pool.time) if hi is None else hi
        M, N = len(plan.kind), pool.vm.shape[1]
        # each episode's tampered meters over its window [L, M], the labels
        tampered = [
            self.att.tampered(
                pool.vm[ep.onset : ep.onset + ep.length], pool.va[ep.onset : ep.onset + ep.length], ep.a
            )
            for ep in episodes
        ]
        at_bus = np.zeros((M, N), np.int32)
        at_bus[np.arange(M), plan.bus] = 1  # a meter's bus: a tampered meter marks its bus attacked

        writer = TimelineWriter(path)
        writer.write_static(self.att.net.case, self.own, plan)
        noise_rng = rng(self.settings, "noise")
        for t0 in range(lo, hi, self.block):
            t1 = min(hi, t0 + self.block)
            vm, va = pool.vm[t0:t1], pool.va[t0:t1]
            with np.errstate(invalid="ignore"):  # unsolved frames keep their place with NaN readings
                z_benign = self.noise.sample(meas.h(vm, va), noise_rng)
            z = z_benign.copy()
            family = np.zeros(t1 - t0, np.int8)
            episode = np.full(t1 - t0, -1, np.int32)
            meter_tampered = np.zeros((t1 - t0, M), bool)
            for e, (ep, tam) in enumerate(zip(episodes, tampered)):
                a, b = max(t0, ep.onset), min(t1, ep.onset + ep.length)
                if a >= b:
                    continue
                z[a - t0 : b - t0] += ep.a[a - ep.onset : b - ep.onset]
                meter_tampered[a - t0 : b - t0] = tam[a - ep.onset : b - ep.onset]
                family[a - t0 : b - t0] = FAMILIES.index(ep.family)
                episode[a - t0 : b - t0] = e
            writer.write_frames(
                t0 - lo,
                {
                    "time": pool.time[t0:t1],
                    "split": split[t0:t1],
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
        writer.write_episodes([replace(ep, onset=ep.onset - lo) for ep in episodes])
        writer.close()

    def build_episode(
        self,
        path,
        onset: int,
        family: str = "Am",
        lo: int | None = None,
        hi: int | None = None,
        using: AttackFamily | None = None,
    ) -> dict:
        """Write one episode alone: `family` designed at pool frame `onset` on its own window, over the
        pool frames lo..hi (default the window itself), every frame in the train split, for studying one
        attack by hand. `using` designs it in place of the builder's own family of that name (a variant, such as
        `InterfaceFdia.with_overshoot`). Raises if the family finds no attack there or a frame is unsolved."""
        L = self.episode_lengths()[family]
        lo, hi = (onset if lo is None else lo), (onset + L if hi is None else hi)
        if not (lo <= onset and onset + L <= hi):
            raise ConfigError(f"the window {onset}..{onset + L} must lie inside the frames {lo}..{hi}")
        if not (self.pool.solve_kind[onset : onset + L] == SOLVED).all():
            raise DataConditionError(f"the window at {onset} touches an unsolved frame")
        w = slice(onset, onset + L)
        _init({**self.families, family: using} if using is not None else self.families, self.settings)
        _, ep, sec, miss = _design((0, family, onset, self.pool.vm[w], self.pool.va[w]))
        if ep is None:
            raise NoAdmissibleTarget(f"{family} finds no attack at onset {onset} ({miss})")
        self._write(path, [ep], np.zeros(len(self.pool.time), np.int8), lo, hi)
        tam = self.att.tampered(self.pool.vm[w], self.pool.va[w], ep.a)
        return {
            "frames": hi - lo,
            "onset": onset - lo,
            "length": L,
            "family": family,
            "buses": ep.buses.tolist(),
            "target_branch": ep.target_branch,
            "clients": ep.clients.tolist(),
            "meters_tampered": int(tam.any(axis=0).sum()),
            "design_seconds": round(sec, 1),
        }

    def build(self, path) -> dict:
        """Write the timeline to `path` and return a summary: frames, slots placed and episodes built per
        split and family, slots skipped for unsolved frames, and the seconds spent."""
        start = time.perf_counter()
        pool = self.pool
        T = len(pool.time)
        masks = self._masks()
        surveyed: dict = {}
        survey_missed: dict = {}
        if self.survey:
            # N46: survey whole weeks first, then stratify the weeks on the onsets where an attack exists,
            # so every split gets its share of the weeks an attack can happen in (the near-limit hours of
            # N41 left the test split one Am of 21, since most near-limit hours hold no attack)
            found, cache, tried, survey_missed = self._survey(week_index(pool.time), masks)
            masks = found
            split = self._split(masks, HIT_STRATUM)
            for f in found:
                for code, name in enumerate(SPLITS):
                    surveyed[f"{name}/{f}"] = [
                        int((tried[f] & (split == code)).sum()),
                        int((found[f] & (split == code)).sum()),
                    ]
        else:
            split = self._split(masks)
        slots = self._slots(split, masks)
        self._log(f"{T} frames, {len(slots)} slots placed")
        if self.survey:  # the surveyed families' designs are reused, the rest designed now
            rest = [i for i, s in enumerate(slots) if (s.family, s.onset) not in cache]
            eps, sec, skipped, miss = self._designs([slots[i] for i in rest])
            designs: list[Episode | None] = [None] * len(slots)
            secs = [0.0] * len(slots)
            misses: list[str | None] = [None] * len(slots)
            for i, s in enumerate(slots):
                if (s.family, s.onset) in cache:
                    designs[i], secs[i] = cache[(s.family, s.onset)]
            for i, ep, t_, m in zip(rest, eps, sec, miss):
                designs[i], secs[i], misses[i] = ep, t_, m
        else:
            designs, secs, skipped, misses = self._designs(slots)
        kept = sorted(((s, ep) for s, ep in zip(slots, designs) if ep is not None), key=lambda k: k[0].onset)
        episodes = [ep for _, ep in kept]
        self._write(path, episodes, split)

        fams = list(self.families)
        placed = np.zeros((len(SPLITS), len(fams)), int)
        built = np.zeros_like(placed)
        for s, ep in zip(slots, designs):
            placed[split[s.onset], fams.index(s.family)] += 1
            built[split[s.onset], fams.index(s.family)] += ep is not None
        fam_secs = {f: [secs[i] for i, s in enumerate(slots) if s.family == f and secs[i] > 0] for f in fams}
        return {
            "frames": T,
            "families": fams,
            "splits": list(SPLITS),
            "placed": placed.tolist(),
            "built": built.tolist(),
            "skipped_unsolved": skipped,
            # why the placed slots of each family that were designed found no attack (`MISSES`)
            "missed": {
                f: dict(Counter(m for s, m in zip(slots, misses) if s.family == f and m is not None))
                for f in fams
            },
            "survey": surveyed,
            "survey_missed": survey_missed,  # why the surveyed onsets without an attack have none (`MISSES`)
            "attacked_frames": int(sum(ep.length for ep in episodes)),
            "design_seconds": round(float(sum(secs)), 1),
            "design_seconds_median": {
                f: round(float(np.median(t)), 2) if t else None for f, t in fam_secs.items()
            },
            "seconds": round(time.perf_counter() - start, 1),
        }
