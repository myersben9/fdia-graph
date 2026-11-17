"""What the formulas and the parsers accept, one model each, checked when built (`models.validation`).

A formula keeps its array signature: it builds its input model from its arguments and computes on
the model's fields, which the model has already accepted. A parser (a family name, a system id, a
release name, a line reference) is a model whose field parses the loose input once. Nothing is
checked in a function body.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, ClassVar, Optional, Union, cast

import numpy as np

from ..errors import NoAdmissibleTarget
from .choices import FAMILIES, FAMILY_CODE, GENERATED_FAMILIES, Capability
from .validation import AsArray, AsTuple, AtLeast, Dims, Integer, NonEmpty, Parses, Required, Validated


# ---- parsers of loose input -----------------------------------------------------------------------
def _family_code(f: object) -> int:
    if isinstance(f, str):
        return FAMILY_CODE[f]
    if isinstance(f, (int, np.integer)) and not isinstance(f, bool) and int(f) in FAMILIES:
        return int(f)
    raise KeyError(f)  # a float such as 1.9, a bool, or a code outside the table


def family_codes(families: Sequence[Union[str, int]]) -> tuple[int, ...]:
    """Family names or integer codes, as codes."""
    return tuple(_family_code(f) for f in families)


def system_number(system: Union[str, int]) -> int:
    """ "ieee118", "IEEE118", "118" or 118 -> 118."""
    return int(str(system).strip().lower().replace("ieee", ""))


def release_numbers(release: str) -> tuple[int, ...]:
    """ "v0.8.0", "0.8.0" or "data-v0.8.0" -> (0, 8, 0)."""
    m = re.fullmatch(r"(?:data-)?v?(\d+)\.(\d+)\.(\d+)", str(release).strip())
    if m is None:
        raise ValueError(release)
    return tuple(int(x) for x in m.groups())


def iso_date(d: object) -> _dt.date:
    """'YYYY-MM-DD' (a longer ISO timestamp is cut to its date), a date or a datetime, as a date.
    The pattern is matched first: from Python 3.11 `fromisoformat` also reads '20240131', and an
    input must mean the same on every supported version."""
    if isinstance(d, _dt.datetime):
        return d.date()
    if isinstance(d, _dt.date):
        return d
    if not isinstance(d, str) or not re.match(r"\d{4}-\d{2}-\d{2}", d.strip()):
        raise ValueError(d)
    return _dt.date.fromisoformat(d.strip()[:10])


FAMILY_WORDS = (
    f"must name families; unknown family in it (known: {sorted(FAMILY_CODE)} or codes {sorted(FAMILIES)})"
)


@dataclass(frozen=True)
class FamilySelection(Validated):
    """A selection of attack families by name or code."""

    families: Annotated[Sequence[Union[str, int]], Parses(family_codes, FAMILY_WORDS)]

    @property
    def codes(self) -> tuple[int, ...]:
        return tuple(int(c) for c in self.families)


@dataclass(frozen=True)
class GeneratedFamilies(Validated):
    """The families a timeline generates: At and Am only. The single-snapshot families of older
    releases (Aq, Ad, As, Ar, Al) stay readable (`FamilySelection`) but are not generated."""

    families: Annotated[Sequence[Union[str, int]], Parses(family_codes, FAMILY_WORDS)]

    @property
    def codes(self) -> tuple[int, ...]:
        return tuple(int(c) for c in self.families)

    def invariants(self) -> Iterable[tuple[bool, str]]:
        made = {FAMILY_CODE[n] for n in GENERATED_FAMILIES}
        old = sorted(FAMILIES[c] for c in self.codes if c not in made)
        yield (
            not old,
            f"the generator makes {', '.join(GENERATED_FAMILIES)} only; {', '.join(old)} are single-snapshot "
            "families kept readable in old releases (fdia-graph 0.20 generates them)",
        )


@dataclass(frozen=True)
class DateSpan(Validated):
    """An inclusive span of days; each end a 'YYYY-MM-DD' string, a date or a datetime."""

    start: Annotated[Union[str, _dt.date], Parses(iso_date, "must be a date like '2024-01-31'")]
    end: Annotated[Union[str, _dt.date], Parses(iso_date, "must be a date like '2024-01-31'")]

    @property
    def first_day(self) -> _dt.date:
        return cast(_dt.date, self.start)  # a date once the model is built

    @property
    def last_day(self) -> _dt.date:
        return cast(_dt.date, self.end)

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield self.first_day <= self.last_day, f"the span ends ({self.end}) before it starts ({self.start})"


@dataclass(frozen=True)
class ProfileSource(Validated):
    """What `load_profile` was handed: a source that reads itself (it has `loads()`), such as `IsoFolder`,
    `CsvColumn` or `RawSeries`."""

    source: object

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            callable(getattr(self.source, "loads", None)),  # what `LoadSource`'s runtime check tests
            "source must be a LoadSource: IsoFolder, CsvColumn, RawSeries, or any object with loads()",
        )


@dataclass(frozen=True)
class LoadValues(Validated):
    """A load series handed in directly: a non-empty 1-d sequence of numbers, in any unit."""

    values: Annotated[Union[Sequence[float], np.ndarray], AsArray(float), Dims(1), NonEmpty()]

    @property
    def array(self) -> np.ndarray:
        return cast(np.ndarray, self.values)  # an array once the model is built


@dataclass(frozen=True)
class StateSource(Validated):
    """Where an operating-point pool comes from: an array in memory, or a path (a directory of
    X_*.npy, an .npz or an HDF5 file); neither falls back to $FDIA_GRAPH_INIT, then the download."""

    states: Optional[Union[np.ndarray, str, os.PathLike[str]]] = None

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.states is None or isinstance(self.states, (np.ndarray, str, os.PathLike)),
            f"states must be an array or a path, got {type(self.states).__name__}",
        )
        yield (
            not isinstance(self.states, np.ndarray) or np.issubdtype(self.states.dtype, np.number),
            f"an in-memory state pool must be numeric, got dtype {getattr(self.states, 'dtype', None)}",
        )

    @property
    def array(self) -> Optional[np.ndarray]:
        """The pool itself when one was passed in memory, as float64."""
        return self.states.astype(np.float64) if isinstance(self.states, np.ndarray) else None

    @property
    def path(self) -> Optional[str]:
        """The path passed, as a string (a `pathlib.Path` too), or None; an array is never read as
        one, since it has no truth value."""
        if self.states is None or isinstance(self.states, np.ndarray):
            return None
        return os.fspath(self.states)


@dataclass(frozen=True)
class DatasetName(Validated):
    """A dataset name as the registry looks it up: a built-in name in any case and with spaces
    around it, unless a local registration uses that exact spelling; any other name exactly."""

    name: Union[str, int]
    local: frozenset[str]
    builtin: frozenset[str]
    aliases: dict[Union[str, int], str] = field(default_factory=dict)  # "118" and 118 -> "ieee118"

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            isinstance(self.name, str) or (isinstance(self.name, int) and not isinstance(self.name, bool)),
            f"name must be a dataset name or a bus count, got {self.name!r}",
        )

    @property
    def key(self) -> Union[str, int]:
        n = self.aliases.get(self.name, self.name)
        if isinstance(n, str) and n not in self.local and n.strip().lower() in self.builtin:
            return n.strip().lower()  # "IEEE118" like system_id; a local name stays case-sensitive
        return n


@dataclass(frozen=True)
class AdmissibleTargets(Validated):
    """The families a timeline attacks, against the number of targets the case offers each family
    code; a family with none is `NoAdmissibleTarget`, found before any frame is walked."""

    error: ClassVar[type[ValueError]] = NoAdmissibleTarget

    families: Annotated[Sequence[Union[str, int]], Parses(family_codes, FAMILY_WORDS)]
    targets: dict[int, int]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        codes = {int(f) for f in self.families} - {0}  # codes once parsed; benign needs no target
        empty = sorted(f for f in codes if self.targets.get(f, 0) == 0)
        names = ", ".join(FAMILIES[f] for f in empty)
        yield not empty, f"no admissible target on this case for {names}; drop them from families"


@dataclass(frozen=True)
class SystemRef(Validated):
    """A ladder system by either spelling; `number` is its bus count."""

    system: Annotated[Union[str, int], Parses(system_number, "must be like 'ieee118' or 118")]

    @property
    def number(self) -> int:
        return int(self.system)


@dataclass(frozen=True)
class SupportedSystem(Validated):
    """A system the case builders support."""

    system: Annotated[Union[str, int], Parses(system_number, "must be like 'ieee118' or 118")]
    supported: frozenset[int] = frozenset()

    @property
    def number(self) -> int:
        return int(self.system)

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.number in self.supported,
            f"unknown system {self.number}; supported: {sorted(self.supported)}",
        )


FIRST_RELEASE = (0, 8, 0)  # the first data release of timelines, the only layout the package reads


@dataclass(frozen=True)
class ReleaseName(Validated):
    """A data release name; `numbers` orders releases. The first release this package reads is v0.8.0:
    the record shards and stream files before it are read by fdia-graph 0.21 and earlier."""

    release: Annotated[
        Union[str, tuple[int, ...]], Parses(release_numbers, "must be a data release name like 'v0.8.0'")
    ]

    @property
    def numbers(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.release)

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.numbers >= FIRST_RELEASE,
            "data releases before v0.8.0 are record shards: read them with fdia-graph 0.21 "
            "(pip install fdia-graph==0.21.0)",
        )


# ---- arrays ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StatePool(Validated):
    """An operating-point pool [T, N, 4]."""

    X: Annotated[np.ndarray, AsArray(np.float64)]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.X.ndim == 3 and self.X.shape[2] == 4,
            f"a state pool is [T, N, 4], got shape {self.X.shape}",
        )


@dataclass(frozen=True)
class FieldRequest(Validated):
    """Fields asked of a dataset view against the ones it offers."""

    fields: Annotated[Sequence[str], AsTuple()]
    offered: tuple[str, ...]
    known: tuple[str, ...]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        unknown = [k for k in self.fields if k not in self.offered]
        yield not unknown, f"unknown field(s) {unknown}; this shard carries {list(self.known)}"


@dataclass(frozen=True)
class CsvSpec(Validated):
    """One column of one CSV file."""

    path: Union[str, os.PathLike[str]]
    column: Annotated[Optional[str], Required()]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield isinstance(self.path, (str, os.PathLike)), f"path must be a file path, got {self.path!r}"
        yield isinstance(self.column, str), f"column must be a column name, got {self.column!r}"


def capability_names(capabilities: Sequence[str]) -> tuple[str, ...]:
    """Capability names, each one of `Capability`."""
    return tuple(Capability(c).value for c in capabilities)


@dataclass(frozen=True)
class Requirement(Validated):
    """What a consumer (`by`, in words) needs of a dataset view."""

    capabilities: Annotated[
        Sequence[str], Parses(capability_names, f"must name capabilities {Capability.values()}")
    ]
    by: str


@dataclass(frozen=True)
class TrustablePmus(Validated):
    """A trusted-PMU schedule's buses against the generator's grid and meter plan
    (`MinimizeMixin.min_tamper`'s `trust`): every trusted bus is a bus of the grid with a PMU, since a
    trusted PMU pins its own |V| and angle [WU26 eqs. 27, 29] and a bus without one has nothing to
    trust. Built from the schedule and the plan, since `TrustSchedule` alone does not know the grid."""

    buses: Annotated[Sequence[int], AsTuple()]  # the schedule's buses (`TrustSchedule.buses`)
    pmu: frozenset[int]  # the buses with a PMU in the meter plan
    n_bus: Annotated[int, Integer(), AtLeast(1)]  # buses of the grid

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            all(0 <= int(b) < self.n_bus for b in self.buses),
            f"trusted buses must be buses of the {self.n_bus}-bus grid, got {self.buses!r}",
        )
        yield (
            set(int(b) for b in self.buses) <= set(self.pmu),
            f"trusted buses must carry a PMU in the meter plan {sorted(self.pmu)}, got {self.buses!r}",
        )
