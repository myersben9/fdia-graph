"""Every data model of the package in one place (docs/DATA_MODELS_PLAN.md, step 5).

Grouped by what the data is: `grid` (the static system), `frames` (what the generators pass
around per scan), `data` (what a user gets back), `scores` (result tables), `assets` (how files
are found). Models import only numpy and typing, so this package imports without torch,
pandapower or h5py. Every model is also importable from the module that produces it, which is
where it used to be defined.

PUBLIC names the bundles a user receives from the public API; the data dictionary lists those.
"""

from .base import Bundle
from .grid import BranchModel, Admittances, MeterPlan, MeterBias, Outage, INTACT
from .frames import Scan, Frame, FrameKnobs, Record, Redistribution, ResolvedPool
from .data import RecordBundle, BatchBundle, ArraysBundle, Summary, ShardArrays, Stream, TrueState
from .scores import (
    ErrorPair,
    EstimatorScores,
    OverallMetrics,
    BenignMetrics,
    FamilyMetrics,
    LocalizerScores,
    JacobianOutputs,
)
from .assets import AssetSpec, DownloadTarget, LineCandidate

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
