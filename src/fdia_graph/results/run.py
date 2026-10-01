"""How a harness records what it measured: open a `Run`, add values, and the run writes them to the
store with its provenance when it closes.

    from fdia_graph.results import Run

    with Run("se.estimators", system="ieee14", settings=hp, data_release="v0.9.0", seed=1) as run:
        for name, est in methods.items():
            run.add_tree(est.score(test), method=name, levels=("family", "metric"))
        run.add("seconds", 12.5, method="wls")

A score bundle is a nested dict, so `add_tree` takes any of them: each level of nesting fills one key
(`levels`), the innermost key is the metric, and a {"mean", "std"} leaf becomes a value with its
spread. A list fills a `step` tag, one record per element.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import inspect
import json
import os
import platform
import secrets
import subprocess
from collections.abc import Iterable, Mapping
from types import TracebackType
from typing import Optional

from ..models.results import Provenance, Record, leaves
from .store import Store


class Run:
    """One run of one experiment: collects records, then writes them with a `Provenance`."""

    def __init__(
        self,
        experiment: str,
        *,
        system: str = "",
        settings: object = None,
        data_release: str = "",
        seed: Optional[int] = None,
        note: str = "",
        store: Optional[Store] = None,
        run_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        self.system, self.store = system, store or Store()
        stamp = timestamp or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        chash = config_hash(settings)
        sha, dirty = _git_state()
        self.provenance = Provenance(
            run_id=run_id or _run_id(experiment, system, stamp, chash),
            experiment=experiment,
            timestamp=stamp,
            sdk_version=_sdk_version(),
            git_sha=sha,
            dirty=dirty,
            data_release=data_release,
            seed=seed,
            config_hash=chash,
            platform=platform.platform(terse=True),
            note=note,
        )
        self.records: list[Record] = []

    def add(
        self,
        metric: str,
        value: float,
        *,
        sd: Optional[float] = None,
        ci_lo: Optional[float] = None,
        ci_hi: Optional[float] = None,
        **keys: object,
    ) -> Record:
        """One value. `system`, `method`, `family` and `split` fill those keys (the run's system by
        default); any other keyword becomes a "name=value" tag (e.g. `k=1.2`, `step=3`)."""
        return self._add(metric, value, keys, sd, ci_lo, ci_hi)

    def _add(
        self,
        metric: str,
        value: float,
        keys: dict[str, object],
        sd: Optional[float] = None,
        ci_lo: Optional[float] = None,
        ci_hi: Optional[float] = None,
    ) -> Record:
        named = {k: str(v) for k, v in keys.items() if k in ("system", "method", "family", "split")}
        rec = Record(
            experiment=self.provenance.experiment,
            metric=metric,
            value=float(value),
            system=named.get("system", self.system),
            method=named.get("method", ""),
            family=named.get("family", ""),
            split=named.get("split", ""),
            tags=tuple(f"{k}={v}" for k, v in keys.items() if k not in named),
            sd=sd,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )
        self.records.append(rec)
        return rec

    def add_tree(self, tree: Mapping[str, object], levels: Iterable[str] = (), **keys: object) -> int:
        """Every numeric leaf of a nested mapping (a score bundle, a JSON report). The outer levels
        fill the keys named in `levels`, in order, and the key holding the leaf is the metric; a
        deeper leaf than `levels` covers joins its extra keys into the metric name with "." (rare;
        name more levels instead). A {"mean", "std"} mapping is one value with its `sd`; a list adds
        one record per element with a `step` tag; strings, booleans and None are skipped. Returns the
        number of records added."""
        names = tuple(levels)
        before = len(self.records)
        for leaf in leaves(tree):
            outer, metric = leaf.keys[: len(names)], leaf.keys[len(names) :]
            if len(outer) < len(names) or not metric:
                continue  # a leaf above the levels named (e.g. a report's "k" setting): not a measurement here
            step: dict[str, object] = {} if leaf.step is None else {"step": leaf.step}
            self._add(".".join(metric), leaf.value, {**dict(zip(names, outer)), **step, **keys}, sd=leaf.sd)
        return len(self.records) - before

    def write(self) -> int:
        """Write the records collected so far (also done on leaving the `with` block)."""
        return self.store.write(self.provenance, self.records)

    def __enter__(self) -> Run:
        return self

    def __exit__(
        self, kind: Optional[type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]
    ) -> None:
        if kind is None:  # a run that failed half way writes nothing
            self.write()


def _run_id(experiment: str, system: str, stamp: str, chash: str) -> str:
    """experiment-system-time-settings-random: unique even for runs of one experiment and settings
    that finish in the same second on several systems or processes."""
    when = stamp.replace(":", "").replace("-", "").lower()
    return "-".join(p for p in (experiment, system, when, chash[:6], secrets.token_hex(3)) if p)


def config_hash(settings: object) -> str:
    """A stable hash of a run's settings: a model's fields, a mapping, or any value's repr."""
    if dataclasses.is_dataclass(settings) and not inspect.isclass(settings):
        settings = dataclasses.asdict(settings)
    blob = json.dumps(settings, sort_keys=True, default=repr)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _sdk_version() -> str:
    from .. import __version__

    return __version__


def _git_state() -> tuple[str, bool]:
    """(HEAD's sha, whether the tree has uncommitted changes); ("", False) outside a git checkout."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "", False
    return sha, bool(status.strip())
