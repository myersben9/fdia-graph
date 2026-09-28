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

from .choices import FAMILIES, FAMILY_ALIAS, Capability, Iso, Reduce
from .errors import NoAdmissibleTarget
from .validation import (
    AsArray,
    AsTuple,
    AtLeast,
    Dims,
    Integer,
    IntegerDtype,
    NonEmpty,
    OneOf,
    Parses,
    Required,
    Validated,
)

# ---- parsers of loose input -----------------------------------------------------------------------
_FAMILY_NAMES = {**{v: k for k, v in FAMILIES.items()}, **FAMILY_ALIAS}


def _family_code(f: object) -> int:
    if isinstance(f, str):
        return _FAMILY_NAMES[f]
    if isinstance(f, (int, np.integer)) and not isinstance(f, bool) and int(f) in FAMILIES:
        return int(f)
    raise KeyError(f)  # a float such as 1.9, a bool, or a code outside the table


def family_codes(families: Sequence[Union[str, int]]) -> tuple[int, ...]:
    """Family names (with the legacy aliases) or integer codes, as codes."""
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
    f"must name families; unknown family in it (known: {sorted(_FAMILY_NAMES)} or codes {sorted(FAMILIES)})"
)


@dataclass(frozen=True)
class FamilySelection(Validated):
    """A selection of attack families by name, alias or code."""

    families: Annotated[Sequence[Union[str, int]], Parses(family_codes, FAMILY_WORDS)]

    @property
    def codes(self) -> tuple[int, ...]:
        return tuple(int(c) for c in self.families)


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
    """What `load_profile` was handed: a source that reads itself (it has `loads()`), or the pre-0.21
    form, an operator name with a directory, a CSV path with its column, or the load values."""

    source: object
    path: Optional[str] = None
    column: Optional[str] = None

    @property
    def text(self) -> str:
        """The operator name or CSV path, for the kinds "iso" and "csv"."""
        return cast(str, self.source)

    @property
    def series(self) -> Sequence[float]:
        """The load values, for the kind "values"."""
        return cast(Sequence[float], self.source)

    @property
    def kind(self) -> str:
        """ "source", "values", "iso", "csv", or "unsupported" (which the invariant refuses)."""
        if callable(getattr(self.source, "loads", None)):  # what `LoadSource`'s runtime check tests
            return "source"
        if isinstance(self.source, str):
            return "iso" if self.source.lower() in Iso.values() else "csv"
        if isinstance(self.source, (list, tuple, np.ndarray)):
            values = np.asarray(cast(Sequence[float], self.source))
            if values.ndim == 1 and values.size and np.issubdtype(values.dtype, np.number):
                return "values"
        return "unsupported"

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.kind != "unsupported",
            "source must be a LoadSource, an operator name, a CSV path, or a 1-d sequence of load values",
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


@dataclass(frozen=True)
class StreamSystem(Validated):
    """The system a deprecated torch_data helper streams when given neither a dataset nor a stream."""

    system: Annotated[
        Optional[Union[str, int]],
        Required("is required: pass dataset=<fg.load(..., order='time')>, stream=<dict>, or a system name"),
        Parses(system_number, "must be like 'ieee118' or 118"),
    ]

    @property
    def number(self) -> int:
        return int(cast(int, self.system))  # an int once the model is built


@dataclass(frozen=True)
class ReleaseName(Validated):
    """A data release name; `numbers` orders releases."""

    release: Annotated[
        Union[str, tuple[int, ...]], Parses(release_numbers, "must be a data release name like 'v0.8.0'")
    ]

    @property
    def numbers(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.release)


@dataclass(frozen=True)
class OutageRef(Validated):
    """A line taken out, by name or by index, against the case's lines; `index` is the line index."""

    outage: Annotated[
        Union[str, int], Required("is required: an outage is a line name or an integer line index")
    ]
    names: tuple[str, ...]
    indices: tuple[int, ...]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        if isinstance(self.outage, str):
            hits = self.names.count(self.outage)
            yield hits != 0, f"no line named {self.outage!r} in this case"
            yield hits <= 1, f"line name {self.outage!r} is ambiguous ({hits} matches); pass an index"
            return
        yield (
            Integer().holds(self.outage),
            f"an outage is a line name or an integer line index, got {self.outage!r}",
        )
        yield (
            int(self.outage) in self.indices,
            f"line index {self.outage} is not in this case (lines are {min(self.indices)}..{max(self.indices)})",
        )

    @property
    def index(self) -> int:
        if isinstance(self.outage, str):
            return self.indices[self.names.index(self.outage)]
        return int(self.outage)


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
class ShapedArray(Validated):
    """An array a caller hands back (scores, estimates) that must match the view it belongs to."""

    values: Annotated[np.ndarray, AsArray(np.float64)]
    shape: tuple[int, ...]
    what: str = "values"

    def invariants(self) -> Iterable[tuple[bool, str]]:
        dims = ", ".join(str(d) for d in self.shape)
        yield self.values.shape == self.shape, f"{self.what} must be [{dims}], got {self.values.shape}"


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


@dataclass(frozen=True)
class SubBatch(Validated):
    """Records per sub-batch of a batched computation."""

    sub: Annotated[int, Integer(), AtLeast(1)] = 50


# ---- the federated formulas ----------------------------------------------------------------------
@dataclass(frozen=True)
class EdgeList(Validated):
    """A branch list [2, E] over buses 0..N-1."""

    edge_index: Annotated[np.ndarray, AsArray()]
    N: Annotated[int, Integer(), AtLeast(0)]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        ei = self.edge_index
        yield (
            ei.ndim == 2 and ei.shape[0] == 2 and IntegerDtype().holds(ei),
            f"edge_index must be an integer [2, E] array, got shape {ei.shape}",
        )
        yield (
            not ei.size or (ei.min() >= 0 and ei.max() < self.N),
            f"edge_index names a bus outside 0..{self.N - 1}",
        )


@dataclass(frozen=True)
class ClientCount(Validated):
    """K clients over N buses."""

    N: Annotated[int, Integer()]
    K: Annotated[int, Integer()]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield 1 <= self.K <= self.N, f"K must be between 1 and the {self.N} buses, got {self.K}"


@dataclass(frozen=True)
class ClientGraph(Validated):
    """The client of every bus [N] and the bus adjacency [N, N]; with K, the clients must be exactly
    0..K-1."""

    assignment: Annotated[np.ndarray, AsArray()]
    adjacency: Annotated[np.ndarray, AsArray()]
    K: Annotated[Optional[int], Integer()] = None

    def invariants(self) -> Iterable[tuple[bool, str]]:
        n = len(self.assignment) if self.assignment.ndim == 1 else -1
        yield (
            n >= 0 and self.adjacency.shape == (n, n),
            f"need an [N] assignment and an [N, N] adjacency, got {self.assignment.shape} and {self.adjacency.shape}",
        )
        if self.K is not None:
            used = np.unique(self.assignment)  # never range(K): K may be any size
            yield (
                self.K >= 1 and len(used) == self.K and used[0] == 0 and used[-1] == self.K - 1,
                f"the assignment must use exactly the clients 0..{self.K - 1}",
            )


@dataclass(frozen=True)
class Halo(Validated):
    """Client k's halo of `depth` hops over a client graph."""

    graph: ClientGraph
    k: Annotated[int, Integer()]
    depth: Annotated[int, Integer()]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield self.depth >= 0, f"depth must be >= 0, got {self.depth}"
        yield bool((self.graph.assignment == self.k).any()), f"client {self.k} owns no bus"


@dataclass(frozen=True)
class AssignmentSpec(Validated):
    """A client-of-every-bus array over a branch list, with an optional attackable mask. The grid has
    the buses the branches name, or the assignment's length when there are no branches."""

    assignment: Annotated[np.ndarray, AsArray()]
    edge_index: Annotated[np.ndarray, AsArray()]
    attackable: Annotated[Optional[np.ndarray], AsArray(bool)] = None

    @property
    def N(self) -> int:
        return int(self.edge_index.max()) + 1 if self.edge_index.size else len(self.assignment)

    def invariants(self) -> Iterable[tuple[bool, str]]:
        a, ei = self.assignment, self.edge_index
        yield (
            ei.ndim == 2 and ei.shape[0] == 2 and IntegerDtype().holds(ei) and (not ei.size or ei.min() >= 0),
            f"edge_index must be a non-negative integer [2, E] array, got shape {ei.shape}",
        )
        yield (
            a.ndim == 1 and len(a) > 0 and len(a) >= self.N and IntegerDtype().holds(a),
            f"assignment must be one integer client per bus ({self.N} buses)",
        )
        K = int(a.max()) + 1
        yield a.min() >= 0 and len(np.unique(a)) == K, f"clients must be numbered 0..{K - 1} with none empty"
        if self.attackable is not None:
            yield (
                self.attackable.shape == a.shape,
                f"the attackable mask must be one flag per bus, shape {a.shape}",
            )


@dataclass(frozen=True)
class PartitionOnGrid(Validated):
    """A partition's assignment fit for a system of N buses."""

    assignment: Annotated[np.ndarray, AsArray()]
    K: Annotated[int, Integer()]
    N: Annotated[int, Integer()]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        a = self.assignment
        yield (
            a.ndim == 1 and IntegerDtype().holds(a),
            f"the partition's assignment must be a 1-D integer array, got {a.dtype} {a.shape}",
        )
        yield len(a) == self.N, f"the partition covers {len(a)} buses, the system has {self.N}"
        labels = np.unique(a)
        yield (
            np.array_equal(labels, np.arange(self.K)),
            f"the partition must number its clients 0..{self.K - 1}, got {labels.tolist()}",
        )


@dataclass(frozen=True)
class ClientUpdates(Validated):
    """One tensor per client and its weight (record count)."""

    tensors: Annotated[Sequence[np.ndarray], AsTuple()]
    weights: Annotated[Union[Sequence[float], np.ndarray], AsArray(np.float64)]

    @property
    def weight_array(self) -> np.ndarray:
        return cast(np.ndarray, self.weights)  # an array once the model is built

    def invariants(self) -> Iterable[tuple[bool, str]]:
        n, w = len(self.tensors), self.weight_array
        yield (
            n > 0 and n == len(w),
            f"need one weight per client tensor, got {n} tensors and {len(w)} weights",
        )
        yield (
            w.shape == (n,),
            f"need one scalar weight per client, shape ({n},), got {w.shape}",
        )
        yield (
            bool((np.isfinite(w) & (w > 0)).all()),
            "client weights must be finite and positive",
        )
        shape = np.shape(self.tensors[0])
        yield (
            all(np.shape(t) == shape for t in self.tensors),
            f"every client tensor must have the shape {shape}",
        )


@dataclass(frozen=True)
class MomentParts(Validated):
    """(count, mean [C], variance [C]) per part, to be pooled."""

    parts: Annotated[Sequence[tuple[float, np.ndarray, np.ndarray]], AsTuple()]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield len(self.parts) > 0, "pool_moments needs at least one part"
        shape = np.shape(self.parts[0][1])
        yield (
            all(
                np.shape(m) == shape and np.shape(v) == shape and np.isfinite(c) and c > 0
                for c, m, v in self.parts
            ),
            f"every part needs a finite positive count and moments of shape {shape}",
        )


@dataclass(frozen=True)
class FeatureBlock(Validated):
    """A client's feature block [n, N, C] with at least one record and one bus."""

    X: Annotated[np.ndarray, AsArray(), Dims(3)]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.X.shape[0] * self.X.shape[1] != 0,
            f"need a non-empty [n, N, C] block, got shape {self.X.shape}",
        )


@dataclass(frozen=True)
class Affinity(Validated):
    """A square adjacency, an attackable mask over its buses, and the weight of an attackable bus."""

    adjacency: Annotated[np.ndarray, AsArray()]
    attackable: Annotated[np.ndarray, AsArray(bool)]
    heavy: float = 8.0

    def invariants(self) -> Iterable[tuple[bool, str]]:
        A = self.adjacency
        yield A.ndim == 2 and A.shape[0] == A.shape[1], f"need a square [N, N] adjacency, got shape {A.shape}"
        yield self.attackable.shape == (A.shape[0],), f"need a [{A.shape[0]}] attackable mask"
        yield (
            bool(np.isfinite(self.heavy)) and 0 < self.heavy <= np.finfo(np.float32).max,
            f"heavy must be a finite positive float32 weight, got {self.heavy}",
        )


@dataclass(frozen=True)
class StateBlocks(Validated):
    """Per-client (state columns, basis) blocks over a state of dimension d."""

    blocks: Annotated[Sequence[tuple[np.ndarray, np.ndarray]], AsTuple()]
    d: Annotated[int, Integer()]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        ok = len(self.blocks) > 0 and all(_index_array(c, self.d) for c, _ in self.blocks)
        yield ok, f"need at least one block of integer state columns inside 0..{self.d - 1}"
        cols = np.concatenate([np.asarray(c) for c, _ in self.blocks])
        yield len(np.unique(cols)) == len(cols), "the blocks' state columns must be disjoint"
        yield (
            all(np.ndim(V) == 2 and np.shape(V)[0] == len(c) for c, V in self.blocks),
            "each basis needs one row per state column of its block",
        )


def _index_array(c: object, d: int) -> bool:
    """A one-dimensional integer array of state columns inside 0..d-1."""
    c = np.asarray(c)
    if c.ndim != 1 or not np.issubdtype(c.dtype, np.integer):
        return False
    return not len(c) or bool(c.min() >= 0 and c.max() < d)


# ---- the metrics -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelGrids(Validated):
    """Predicted and true per-bus labels [n, N]."""

    pred: Annotated[np.ndarray, AsArray(bool)]
    truth: Annotated[np.ndarray, AsArray(bool)]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.pred.shape == self.truth.shape and self.pred.ndim == 2,
            f"need two [n, N] boolean arrays of one shape, got {self.pred.shape} and {self.truth.shape}",
        )


@dataclass(frozen=True)
class TauSearch(Validated):
    """Per-bus counts [n_taus, N] at each candidate threshold, the buses to average and the grid."""

    counts: tuple[np.ndarray, np.ndarray, np.ndarray]
    active: Annotated[np.ndarray, AsArray(bool)]
    taus: Annotated[np.ndarray, AsArray()]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.taus.ndim == 1 and self.active.ndim == 1,
            f"taus and active must be one-dimensional, got shapes {self.taus.shape} and {self.active.shape}",
        )
        shape = (len(self.taus), len(self.active))
        yield (
            len(self.taus) > 0 and bool(self.active.any()) and all(np.shape(c) == shape for c in self.counts),
            f"need [n_taus, N] counts of shape {shape}, a non-empty tau grid and an active bus",
        )


@dataclass(frozen=True)
class RankedLabels(Validated):
    """Scores [n] and labels [n] with at least one positive."""

    score: Annotated[np.ndarray, AsArray(np.float64)]
    truth: Annotated[np.ndarray, AsArray(bool)]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.score.ndim == 1 and self.score.shape == self.truth.shape and bool(self.truth.any()),
            "average_precision needs matching 1-D scores and labels with a positive",
        )


@dataclass(frozen=True)
class Aggregation(Validated):
    """How a per-meter quantity aggregates to a bus."""

    reduce: Annotated[str, OneOf(Reduce)]


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
