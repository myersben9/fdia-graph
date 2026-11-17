"""What the analysis formulas accept, one model each, checked when built by the validation engine of
`fdia_graph.models.validation`.

A formula keeps its array signature: it builds its input model from its arguments and computes on
the model's fields, which the model has already accepted. Nothing is checked in a function body.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Annotated, Optional, Union, cast

import numpy as np

from fdia_graph.models.validation import (
    AsArray,
    AsTuple,
    AtLeast,
    Dims,
    Integer,
    OneOf,
    Rule,
    Validated,
)

from .choices import Reduce


class IntegerDtype(Rule):
    """The one rule only the analysis models declare; every other rule is the generation toolkit's."""

    says = "must hold integers"

    def holds(self, value: object) -> bool:
        return np.issubdtype(np.asarray(value).dtype, np.integer)


# ---- arrays ---------------------------------------------------------------------------------------
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
