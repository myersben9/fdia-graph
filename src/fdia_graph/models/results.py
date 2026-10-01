"""The models of a results table: the metric registry, a `Record` (one measured value with the keys
that say what it is) and a `Provenance` (which run made it), plus the parsers that read loose input
(a nested score report, a filter value, a run's settings) by its type. Every check lives here, per
the validation rule, so a store never holds an unknown metric, a non-finite value or a malformed key;
`fdia_graph.results` builds these models and never checks its input itself."""

from __future__ import annotations

import numbers
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Annotated, Optional

from .choices import Choice
from .validation import AsTuple, AtLeast, Finite, Integer, Parses, Validated


class Better(Choice):
    """Which direction of a metric is the good one; `none` for a count or a setting."""

    HIGHER = "higher"
    LOWER = "lower"
    NONE = "none"


@dataclass(frozen=True)
class Metric:
    """One registered metric."""

    unit: str  # "" for a ratio in [0, 1]
    better: Better
    fmt: str  # the default format spec of a table cell, e.g. ".3f"
    what: str  # one line: what is measured


def _m(unit: str, better: str, fmt: str, what: str) -> Metric:
    return Metric(unit, Better(better), fmt, what)


METRICS: dict[str, Metric] = {
    # ---- state estimation
    "angle_mae_deg": _m("deg", "lower", ".3f", "mean absolute angle error against the clean state"),
    "voltage_mae_pu": _m(
        "pu", "lower", ".2e", "mean absolute voltage-magnitude error against the clean state"
    ),
    # ---- localization and detection
    "macro_f1": _m("", "higher", ".3f", "per-bus F1 averaged over the active buses"),
    "macro_dr": _m("", "higher", ".3f", "per-bus detection rate averaged over the active buses"),
    "macro_fr": _m("", "lower", ".4f", "per-bus false-positive rate on benign records"),
    "macro_auprc": _m("", "higher", ".3f", "per-bus area under the precision-recall curve, averaged"),
    "node_f1": _m("", "higher", ".3f", "micro F1 over every bus call"),
    "node_precision": _m("", "higher", ".3f", "micro precision over every bus call"),
    "node_recall": _m("", "higher", ".3f", "micro recall over every bus call"),
    "sample_f1": _m("", "higher", ".3f", "per-record F1 averaged over records (per-sample macro-F1)"),
    "strict_acc": _m("", "higher", ".3f", "strict localization accuracy: predicted set equals the true set"),
    "detection_rate": _m("", "higher", ".3f", "records with any bus flagged, among attacked records"),
    "false_alarm_rate": _m("", "lower", ".4f", "benign records with any bus flagged"),
    "bus_alarm_rate": _m("", "lower", ".4f", "mean per-bus flag rate on benign records"),
    "tau": _m("", "none", ".3f", "the decision threshold calibrated on validation"),
    # ---- trusted meters
    "detected_before": _m("", "higher", ".3f", "attacked records the residual test flags, nothing secured"),
    "detected_after": _m(
        "", "higher", ".3f", "attacked records the residual test flags, the selection secured"
    ),
    "fit_seconds": _m("s", "lower", ".1f", "wall time of a fit"),
    "meters": _m("", "none", "d", "metered channels the selection chooses from"),
    "selected_meter": _m("", "none", "d", "the meter secured at a step of a selection order"),
    "attack_cost": _m("", "none", ".0f", "the attack cost after a step of a selection order"),
    # ---- the fewest-tamper attack and [WU26]'s defense
    "devices": _m("", "none", "d", "devices the attack tampers"),
    "channels": _m("", "none", "d", "measurement channels the attack tampers"),
    "cost_increase_pct": _m("%", "higher", ".1f", "rise in the attack cost the defense forces, percent"),
    "extra_devices": _m("", "higher", "d", "devices the attack adds under the defense"),
    "trusted_pmu": _m("", "none", "d", "the PMU bus trusted at a configuration step"),
    "decision_seconds": _m("s", "lower", ".2f", "wall time of one defense decision"),
    "selection_share": _m("", "none", ".2f", "share of tests in which a PMU is trusted at a step"),
    "max_change_pu": _m("pu", "none", ".2f", "largest change the attack makes on one channel"),
    "proven_share": _m("", "higher", ".2f", "searches that proved their support the fewest"),
    "episodes": _m("", "none", "d", "attack episodes"),
    "seconds_per_episode": _m("s", "lower", ".1f", "generation wall time per episode"),
    "lower_bound": _m("", "none", "d", "the certifier's lower bound on the devices"),
    "gap": _m("", "lower", "d", "devices between the search's count and the certifier's bound"),
    "mismatch_mw": _m("MW", "lower", ".1f", "the relaxed point's distance from an AC power flow"),
    "certified": _m("", "higher", "d", "1 when the certifier proves the search's count minimal, else 0"),
    "uncertain": _m(
        "", "lower", "d", "1 when the certifier's verdict sits at the solver's tolerances, else 0"
    ),
    "angle_bounded_buses": _m("", "none", "d", "area buses whose angle move bound tightening bounds"),
    "voltage_move_pu": _m("pu", "none", ".1f", "median bound on a bus voltage move after tightening"),
    # ---- generation and benchmarks
    "seconds": _m("s", "lower", ".2f", "wall time"),
    "ms_per_record": _m("ms", "lower", ".3f", "wall time per record (or per frame, for generation)"),
    "attacked_frac": _m("", "none", ".3f", "share of frames attacked"),
    "frames": _m("", "none", "d", "frames"),
    "redraws": _m(
        "", "lower", "d", "episodes moved to a new onset or line pair because their attack was infeasible"
    ),
    "shortfall": _m("", "lower", "d", "episodes given up after the redraw cap"),
    "minutes": _m("min", "lower", ".1f", "wall time in minutes"),
    "buses": _m("", "none", "d", "buses of a system"),
    "branches": _m("", "none", "d", "branches of a system"),
    "quantile": _m(
        "", "none", ".3f", "a quantile of a quantity over the operating pool (the tag names which)"
    ),
}

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_.+-]*$")
_TAG = re.compile(r"^[a-z][a-z0-9_]*=[^;=]+$")


def _slug(value: str) -> str:
    if not _SLUG.match(value):
        raise ValueError(value)
    return value


def _metric(value: str) -> str:
    METRICS[value]  # a KeyError means unregistered
    return value


def _tags(value: tuple[object, ...]) -> tuple[str, ...]:
    tags = tuple(sorted(str(t) for t in value))
    if any(not _TAG.match(t) for t in tags) or len({t.split("=")[0] for t in tags}) != len(tags):
        raise ValueError(value)
    return tags


# the keys that, with the run, identify a record; a query filters on any of them
KEYS = ("system", "method", "family", "split", "metric", "tags")


@dataclass(frozen=True)
class Record(Validated):
    """One measured value. `tags` holds any further key as "name=value" (e.g. "k=1.2", "step=3")."""

    experiment: Annotated[str, Parses(_slug, "must be a lower-case slug (letters, digits, . _ + -)")]
    metric: Annotated[str, Parses(_metric, "must be a metric of results.METRICS")]
    value: Annotated[float, Finite()]
    system: str = ""
    method: str = ""
    family: str = ""
    split: str = ""
    tags: Annotated[tuple[str, ...], AsTuple(), Parses(_tags, 'must be distinct "name=value" tags')] = ()
    sd: Annotated[Optional[float], Finite(), AtLeast(0.0)] = None  # spread over seeds, when averaged
    ci_lo: Annotated[Optional[float], Finite()] = None
    ci_hi: Annotated[Optional[float], Finite()] = None
    run_id: str = ""

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (self.ci_lo is None) == (self.ci_hi is None), "ci_lo and ci_hi come together"
        yield self.ci_lo is None or self.ci_hi is None or self.ci_lo <= self.ci_hi, "ci_lo <= ci_hi"

    @property
    def unit(self) -> str:
        return METRICS[self.metric].unit

    def tag(self, name: str, default: str = "") -> str:
        """The value of tag `name`, or `default`."""
        return next((t.split("=", 1)[1] for t in self.tags if t.split("=", 1)[0] == name), default)

    def key(self) -> tuple[str, ...]:
        """What the record measures, without the run: two records with equal keys are the same
        measurement from different runs."""
        return (self.system, self.method, self.family, self.split, self.metric, ";".join(self.tags))


@dataclass(frozen=True)
class Provenance(Validated):
    """Which run made a set of records: enough to reproduce or to distrust it."""

    run_id: Annotated[str, Parses(_slug, "must be a lower-case slug")]
    experiment: Annotated[str, Parses(_slug, "must be a lower-case slug")]
    timestamp: str  # UTC, ISO 8601
    sdk_version: str
    git_sha: str = ""
    dirty: bool = False  # uncommitted changes in the working tree when the run started
    data_release: str = ""
    seed: Annotated[Optional[int], Integer()] = None
    config_hash: str = ""  # of the run's settings, so equal settings hash equal
    platform: str = ""
    note: str = ""  # e.g. "migrated from docs/se/results/se_ieee14.json"


# ---- loose input, read by its type
@dataclass(frozen=True)
class Leaf:
    """One numeric leaf of a nested report: its keys (one per outer level, then the metric), its
    value, its spread when the leaf was a {"mean", "std"} pair, and its index when it sat in a list."""

    keys: tuple[str, ...]
    value: float
    sd: Optional[float] = None
    step: Optional[int] = None


def leaves(tree: object, path: tuple[str, ...] = ()) -> Iterator[Leaf]:
    """Every numeric leaf of a nested mapping. A {"mean", "std"} mapping is one leaf with its spread;
    a list yields one leaf per numeric element with its index; strings, booleans and None are not
    measurements and are skipped."""
    if isinstance(tree, Mapping):
        for name, child in tree.items():
            yield from _leaves_of(child, (*path, str(name)))


def _leaves_of(child: object, here: tuple[str, ...]) -> list[Leaf]:
    """The leaves one value of a report holds, at key path `here`."""
    if isinstance(child, Mapping) and {"mean", "std"} <= set(child):
        return [Leaf(here, float(child["mean"]), sd=float(child["std"]))]
    if isinstance(child, Mapping):
        return list(leaves(child, here))
    if _is_number(child):
        return [Leaf(here, float(child))]  # type: ignore[arg-type]
    if isinstance(child, (list, tuple)):
        return [Leaf(here, float(v), step=i) for i, v in enumerate(child) if _is_number(v)]
    return []


def _is_number(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def as_values(value: object) -> tuple[object, ...]:
    """A query filter as the values it accepts: a collection is any of its members, else the value."""
    return tuple(value) if isinstance(value, (list, tuple, set, frozenset)) else (value,)


@dataclass(frozen=True)
class RunRecords(Validated):
    """What a store writes in one go: a run's provenance and its records, all of that experiment."""

    provenance: Provenance
    records: Annotated[tuple[Record, ...], AsTuple()]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            all(r.experiment == self.provenance.experiment for r in self.records),
            f"every record of a run belongs to its experiment {self.provenance.experiment!r}",
        )
