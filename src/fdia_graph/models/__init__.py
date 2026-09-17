"""Every data model of the package in one place (docs/plans/DATA_MODELS_PLAN.md, step 5).

Grouped by what the data is: `grid` (the static system), `frames` (what the generators pass
around per scan), `data` (what a user gets back), `scores` (result tables), `assets` (how files
are found). The package's own modules import only numpy, typing and dataclasses, so no model
depends on a producer and no import cycle is possible (importing it still runs the parent
package, loader and h5py included). Every model is also importable from the module that
produces it, which is where it used to be defined.

PUBLIC names the bundles a user receives from the public API; the data dictionary lists those.
"""

from .assets import AssetSpec, DownloadTarget, LineCandidate
from .base import Bundle
from .data import ArraysBundle, BatchBundle, RecordBundle, ShardArrays, Stream, Summary, TrueState
from .frames import Frame, FrameKnobs, Record, Redistribution, ResolvedPool, Scan
from .grid import INTACT, Admittances, BranchModel, MeterBias, MeterPlan, Outage
from .scores import (
    BenignMetrics,
    ErrorPair,
    EstimatorScores,
    FamilyMetrics,
    JacobianOutputs,
    LocalizerScores,
    OverallMetrics,
)

PUBLIC = (
    "RecordBundle",
    "BatchBundle",
    "ArraysBundle",
    "Summary",
    "Stream",
    "EstimatorScores",
    "ErrorPair",
    "LocalizerScores",
    "OverallMetrics",
    "BenignMetrics",
    "FamilyMetrics",
    "JacobianOutputs",
    "LineCandidate",
)

__all__ = [
    "Bundle",
    "BranchModel",
    "Admittances",
    "MeterPlan",
    "MeterBias",
    "Outage",
    "INTACT",
    "Scan",
    "Frame",
    "FrameKnobs",
    "Record",
    "Redistribution",
    "ResolvedPool",
    "RecordBundle",
    "BatchBundle",
    "ArraysBundle",
    "Summary",
    "ShardArrays",
    "Stream",
    "TrueState",
    "ErrorPair",
    "EstimatorScores",
    "OverallMetrics",
    "BenignMetrics",
    "FamilyMetrics",
    "LocalizerScores",
    "JacobianOutputs",
    "AssetSpec",
    "DownloadTarget",
    "LineCandidate",
]
