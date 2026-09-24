"""Every data model of the package in one place (docs/plans/DATA_MODELS_PLAN.md, step 5).

Grouped by what the data is: `fields` (the field groups the data bundles share), `grid` (the static system), `frames` (what the generators pass
around per scan), `training` (what a learned localizer's fitting pieces share), `data` (what a user gets back), `scores` (result tables), `assets` (how files
are found). The package's own modules import only numpy, typing and dataclasses, so no model
depends on a producer and no import cycle is possible (importing it still runs the parent
package, loader and h5py included). Every model is also importable from the module that
produces it, which is where it used to be defined.

PUBLIC names the bundles a user receives from the public API; the data dictionary lists those.
"""

from .assets import AssetSpec, DownloadTarget, LineCandidate
from .base import Bundle
from .data import (
    ArraysBundle,
    BatchBundle,
    EpisodeTable,
    RecordBundle,
    Stream,
    Summary,
    TrueState,
)
from .federated import Partition, RoundLog
from .fields import (
    CleanFields,
    GraphFields,
    LabelFields,
    RecordIds,
    ScanFields,
    StreamLayers,
    TemporalFields,
)
from .frames import Frame, FrameKnobs, OperatingLimits, Redistribution, ResolvedPool, Scan
from .grid import (
    BRANCH,
    EDGE,
    INTACT,
    NODE,
    Admittances,
    BranchColumns,
    BranchModel,
    EdgeColumns,
    MeterBias,
    MeterPlan,
    NodeColumns,
    Outage,
)
from .scores import (
    BenignMetrics,
    ErrorPair,
    EstimatorScores,
    FamilyMetrics,
    GridScores,
    JacobianOutputs,
    LocalizerScores,
    OverallMetrics,
    PerBusMetrics,
    PerBusScores,
    TrustScores,
)
from .training import OptimConfig

PUBLIC = (
    "RecordBundle",
    "BatchBundle",
    "ArraysBundle",
    "Summary",
    "EpisodeTable",
    "Stream",
    "EstimatorScores",
    "ErrorPair",
    "LocalizerScores",
    "TrustScores",
    "OverallMetrics",
    "BenignMetrics",
    "FamilyMetrics",
    "JacobianOutputs",
    "PerBusScores",
    "PerBusMetrics",
    "GridScores",
    "LineCandidate",
)

__all__ = [
    "Bundle",
    "ScanFields",
    "LabelFields",
    "RecordIds",
    "TemporalFields",
    "CleanFields",
    "GraphFields",
    "StreamLayers",
    "BranchModel",
    "NodeColumns",
    "EdgeColumns",
    "BranchColumns",
    "NODE",
    "EDGE",
    "BRANCH",
    "Admittances",
    "MeterPlan",
    "MeterBias",
    "Outage",
    "INTACT",
    "Scan",
    "Frame",
    "FrameKnobs",
    "OptimConfig",
    "Partition",
    "RoundLog",
    "OperatingLimits",
    "Redistribution",
    "ResolvedPool",
    "RecordBundle",
    "BatchBundle",
    "ArraysBundle",
    "Summary",
    "EpisodeTable",
    "Stream",
    "TrueState",
    "ErrorPair",
    "EstimatorScores",
    "OverallMetrics",
    "BenignMetrics",
    "FamilyMetrics",
    "GridScores",
    "PerBusMetrics",
    "PerBusScores",
    "LocalizerScores",
    "TrustScores",
    "JacobianOutputs",
    "AssetSpec",
    "DownloadTarget",
    "LineCandidate",
]
