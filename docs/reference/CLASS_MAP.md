# Class map

How the pieces of `fdia_graph` connect: one module diagram and six class diagrams, all drawn from
the source by `tools/class_diagrams.py` (the AST gives each class's public methods, its bases and the
classes of other modules it names), so they cannot drift from the code. CI checks that the committed
sources match; `python tools/class_diagrams.py` rewrites them and `tools/render_mermaid.py` renders.
Read with [`ROADMAP.md`](../ROADMAP.md) (what each file is for) and
[`CONCEPTS_TO_CODE.md`](CONCEPTS_TO_CODE.md) (which function implements which idea).

An open arrowhead is inheritance, a dashed arrow is a class using another (an argument type, a return
type or a collaborator it builds), and a box marked with another package's name is a participant that
lives there.

## Modules

Which module imports which. Four foundations are read by nearly every module and are left out so the
real dependencies show: `models/` (every returned record, every config model and the one validation
engine), `formulas/` (the equations), `schema.py` (the file protocol) and `errors.py` (the named
errors). `__init__.py`, the `fg.*` public API, imports everything and is left out for the
same reason. `streams.py` and `torch_data.py` are deprecated and will be removed in a future release.

![Module diagram: torch_data and streams over dataset; federated over localization, se and dataset; trust and localization over se, which reads dataset and engine (the accuracy classes), and trust over timeline for its temporal layers; generation over engine, timeline, download and registry; profiles over engine and registry; timeline over dataset, engine, generation and registry; download and engine over registry](../figures/diagrams/modules.png)

## The dataset

`FdiaGraph` is the file. Five mixins over `DatasetBase` give it its faces: the static graph physics
(`GraphMixin`), the admittance matrices (`AdmittanceMixin`), records and batches (`RecordsMixin`), the
exports (`ExportMixin`) and the time-ordered windows and episodes (`SequenceMixin`). The methods that
produce records return `models` bundles (a record, a batch, an exported split, the episode table, the
summary); the graph properties are tensors or arrays, `loader()` is a DataLoader and `windows()` a tuple.

![Class diagram of the dataset package: DatasetBase with GraphMixin, AdmittanceMixin, RecordsMixin, ExportMixin and SequenceMixin under it and FdiaGraph inheriting all five; the mixins use ArraysBundle, Summary, BatchBundle, RecordBundle, EpisodeTable, Admittances and BranchModel](../figures/diagrams/classes_dataset.png)

## The engine

`FdiaGenerator` is the AC model of one system with its meter plan. `GridBase` declares what the mixins
provide: `MeasurementMixin` emits scans, `PhysicsMixin` re-solves the whole grid under new loads and
reads a scan's load and generation, and `AttackMixin` builds every attack (next section).
`engine/records.py` emits a benign scan and hands an attacked family to the mixin; `timeline.py`
decides when and where each episode runs.

![Class diagram of the engine package: GridBase with MeasurementMixin and PhysicsMixin under it, AttackMixin composed of EpisodeDesignMixin, StealthyMixin and CorruptMixin, and FdiaGenerator inheriting MeasurementMixin, PhysicsMixin and AttackMixin; the mixins use Scan, AttackDesign, Frame, FrameKnobs and ResolvedPool, and the generator holds a BranchModel, a MeterPlan, a MeterBias, an Outage, a PandapowerNet and its PpcTables](../figures/diagrams/classes_engine.png)

## The attacks

Every attack is built in `engine/attacks/`, one module per concern, composed into `AttackMixin`:
`AreaMixin` finds the attacker's area, `FalseStateMixin` solves and checks its false state and the
attack vector, `RedistributionMixin` draws the load redistribution behind `Al` and `Am`,
`StealthyMixin` builds the stealthy frames, `EpisodeDesignMixin` draws what an episode attacks at its
onset, and `CorruptMixin` tampers `Ad`, `As` and `Ar` in place. `AttackMixin.attack_frame` is the one
entry for an attacked scan.

![Class diagram of the attack package: GridBase with AreaMixin and CorruptMixin under it, FalseStateMixin under AreaMixin, RedistributionMixin under FalseStateMixin, StealthyMixin and EpisodeDesignMixin under RedistributionMixin, and AttackMixin inheriting EpisodeDesignMixin, StealthyMixin and CorruptMixin; the mixins use AttackDesign, RampDesign, AmDesign, Frame, FrameKnobs, OperatingLimits, Redistribution, Scan, Band and TamperTarget](../figures/diagrams/classes_attacks.png)

## State estimation

`SEBase` holds the shared solve: the measurement function, the chord Jacobian, the meter weights and the
chord-Newton loop. Each estimator overrides one hook. `GatedPrior` takes a fitted localizer as its gate;
`JacobianFeatures` is the per-bus feature block the localizers can consume.

![Class diagram of the estimation package: SEBase with WLS, AdaptiveWeighting, ResidualRemoval, SubspacePrior and JacobianWeighting under it, GatedPrior under SubspacePrior using a LocalizerBase gate, JacobianFeatures beside them; the estimators use ErrorPair, EstimatorScores, TrueState and JacobianOutputs](../figures/diagrams/classes_estimation.png)

## Localization and trusted meters

`LocalizerBase` calibrates any per-bus score to one false-alarm budget; the threshold arms score a stored
feature, `ResidualLocalizer` runs an estimator, the learned arms train a per-bus network.
`TrustSelector` chooses the meters to secure on the WLS Jacobian, by row reduction (`TrustedMeters`) or
by a deep Q-network (`TrustedMetersDQN`), and scores the residual test with them secured.

![Class diagram of localization and trust: LocalizerBase with SwingThreshold, DeltaThreshold, ResidualLocalizer and LearnedLocalizer under it, BusCNN and BusMLP under LearnedLocalizer; ResidualLocalizer uses an SEBase estimator, the learned arms use JacobianFeatures; TrustSelector with TrustedMeters and TrustedMetersDQN under it, using WLS; the scores are LocalizerScores, PerBusScores, GridScores and TrustScores](../figures/diagrams/classes_localization_trust.png)

## Federated training

`FederatedLocalizer` is a `LearnedLocalizer` whose fit runs FedAvg over a `Partition` of the buses:
each client trains its own copy with the shared `LocalTrainer` on its own buses, and `FedBusMLP` and
`FedBusCNN` take their encoder from `BusMLP` and `BusCNN`. `RoundLog` records every round.
`RegionalPrior` is a `SubspacePrior` whose basis is fitted per client of a `Partition`.

![Class diagram of the federated package: FederatedLocalizer under LearnedLocalizer, FedBusMLP under FederatedLocalizer and BusMLP, FedBusCNN under FederatedLocalizer and BusCNN; the federated localizer uses LocalTrainer, Partition, RoundLog and JacobianFeatures; RegionalPrior under SubspacePrior uses Partition](../figures/diagrams/classes_federated.png)

## The records

Everything a call returns is a `Bundle`, a dict with attribute access and a fixed key order. The record
forms share seven field-group mixins, one per group of keys, so a record, a batch, an exported split and
a stream carry the same names. An exported split adds an eighth, export-only group,
`PreviousFrameFields` (the previous frame's readings on a timeline).

![Class diagram of the record bundles: Bundle, the seven shared field-group mixins as one node, and RecordBundle, BatchBundle, ArraysBundle and Stream inheriting both; ArraysBundle also takes an eighth, export-only group, PreviousFrameFields; EpisodeTable, Summary and TrueState under Bundle](../figures/diagrams/classes_models.png)

## The scores

The result of every `score()` is a `Bundle` too: `ErrorPair` and `EstimatorScores` for the estimators,
`OverallMetrics`, `BenignMetrics`, `FamilyMetrics` and `LocalizerScores` for the localizers, `TrustScores`
for the trusted meters, `JacobianOutputs` for the feature block.

![Class diagram of the score bundles: ErrorPair, EstimatorScores, OverallMetrics, BenignMetrics, FamilyMetrics, LocalizerScores, PerBusMetrics, PerBusScores, GridScores, TrustScores and JacobianOutputs all under Bundle](../figures/diagrams/classes_scores.png)

The named tuples of `models/frames.py` (`Scan`, `Frame`, `OperatingLimits`, `FrameKnobs`,
`Redistribution`, `ResolvedPool`), `models/grid.py` (the column indices, `BranchModel`, `Admittances`,
`MeterPlan`, `MeterBias`, `Outage`) and `models/assets.py` (`AssetSpec`, `LineCandidate`) are plain
value types with no hierarchy, so they appear only where a class above uses them.
