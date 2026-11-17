"""Every data model of the analysis toolkit in one place: the ones only the estimators, the localizers,
the federated trainers, the trust selectors and the results store use. A model the generation
toolkit also uses is in `fdia_graph.models`, and is imported from there.

Grouped by what the data is: `config` (the settings every consumer accepts, and `inputs`, what the
formulas accept, both checked by the engine in `fdia_graph.models.validation`, with the fixed sets in
`choices`; `inputs` also holds the one rule only they declare), `scores` (the score tables), `data`
(the truth an estimator is scored against), `grid` (the estimator's pseudo-measurement records),
`federated` (a partition and a round's log), `training` (what a learned localizer's fitting pieces
share), `results` (the records of the results store). The package's own modules import only numpy,
the standard library and `fdia_graph.models`, so no model depends on a producer.

PUBLIC names the bundles a user receives from the public API; the data dictionary lists those.
"""

from .config import (
    DqnConfig,
    FederatedSettings,
    FitOptions,
    GateConfig,
    HuberConfig,
    JacobianWeightingConfig,
    LearnedConfig,
    LocalizerConfig,
    PerBusReport,
    PriorConfig,
    RemovalConfig,
    SolveConfig,
    TrainerConfig,
    TrustConfig,
)
from .data import TrueState
from .federated import Partition, RoundLog
from .grid import PseudoLinks, PseudoVoltages
from .inputs import (
    Affinity,
    Aggregation,
    AssignmentSpec,
    ClientCount,
    ClientGraph,
    ClientUpdates,
    EdgeList,
    FeatureBlock,
    Halo,
    LabelGrids,
    MomentParts,
    PartitionOnGrid,
    RankedLabels,
    ShapedArray,
    StateBlocks,
    SubBatch,
    TauSearch,
)
from .results import Leaf, Metric, Provenance, Record, RunRecords
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
)

__all__ = [
    "Leaf",
    "Metric",
    "Provenance",
    "Record",
    "RunRecords",
    "PseudoLinks",
    "PseudoVoltages",
    "OptimConfig",
    "Aggregation",
    "ShapedArray",
    "SubBatch",
    "EdgeList",
    "ClientCount",
    "ClientGraph",
    "Halo",
    "AssignmentSpec",
    "PartitionOnGrid",
    "ClientUpdates",
    "MomentParts",
    "FeatureBlock",
    "Affinity",
    "StateBlocks",
    "LabelGrids",
    "TauSearch",
    "RankedLabels",
    "SolveConfig",
    "FitOptions",
    "HuberConfig",
    "RemovalConfig",
    "PriorConfig",
    "JacobianWeightingConfig",
    "GateConfig",
    "LocalizerConfig",
    "LearnedConfig",
    "TrainerConfig",
    "PerBusReport",
    "TrustConfig",
    "FederatedSettings",
    "DqnConfig",
    "Partition",
    "RoundLog",
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
]
