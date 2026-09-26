# Plan: validation in one place

Goal: every check on an input is declared once, on the data model that carries the input, and
enforced by one engine. No function body validates its own arguments. A reader finds what a value
may be next to the field that holds it, the error messages come from one place, and adding a
parameter means adding a field with its rule rather than another `if ...: raise` at the top of a
function.

Implemented in one change (section 4 lists what landed and what is left). It keeps the
compatibility promise of the readability series: public signatures, accepted values, generated files
and numbers stay the same; only the wording of the error messages changes, and it becomes uniform.

## 1. Where validation sits today

`raise ValueError` / `raise TypeError` on `main` (0.20.0), 119 sites in 24 modules, by kind:

| kind | sites | examples | problem |
|---|---:|---|---|
| argument and configuration values | about 50 | `c must be > 0` (five estimator constructors), `fa_target must be in (0, 1)` (localizers, trust), the fixed-set strings (`units`, `split`, `kcl`, `calibrate`, ...), `_check_knobs`, `_check_settings`, `check_window_args`, `_check_frac` | each constructor re-implements the same range checks with its own wording; the loader checks `split`/`units`/`order` twice (`fg.load` and `FdiaGraph.__init__`) |
| dataset capabilities | about 13 | "needs a timeline", "needs the benign layer", "has no clean layer", `require_physical`, `_check_timeline`, federated `_check_units` | the same property of a dataset is tested in several modules with different messages |
| array contracts of formulas and models | about 38 | `need two [n, N] boolean arrays of one shape` (`formulas.metrics`), eighteen checks in `formulas.federated`, nine in `federated.partition` | a model such as `Partition` is checked by a separate `check_partition` instead of by itself |
| parsing loose input | about 9 | `family_ids`, `registry.system_id`, `release_tuple`, `engine.core._line_id`, `GatedPrior(gate="oracle" or a localizer)` | the type of the argument decides what it means, inside the consumer |
| runtime data conditions | about 13 | "fit needs benign records", "tune_threshold needs attacked records in val", "no room for a 60-frame episode", an outage that islands the grid | these depend on the data read, so they cannot move to construction; they only need named error types |

Reproduce the inventory with `git grep -n "raise ValueError\|raise TypeError" -- src`.

## 2. Design

### 2a. The engine: `fdia_graph/models/validation.py`

A rule is a small declared object; a field carries its rules in its annotation; one base class runs
them.

```python
@dataclass(frozen=True)
class HuberConfig(Validated):
    c: Annotated[float, Positive()] = 1.5
    tol: float = 1e-4
    reweight: Annotated[Optional[str], OneOf(Reweight)] = None   # stored as the canonical string
```

| piece | does |
|---|---|
| `Rule` (`Positive`, `AtLeast(n)`, `InRange(lo, hi, ...)`, `Integer`, `Finite`, `OneOf(<Choice>)`) | one predicate and the phrase it prints ("must be > 0"); `OneOf` also converts the value to the choice's canonical string |
| `Validated.__post_init__` | the one place validation runs: applies each field's rules in order (a None value of an `Optional` field skips them), then the model's `invariants()` |
| `invariants()` | cross-field rules declared as (condition, phrase) pairs: `train_frac + val_frac < 1`, `W <= T`, `partition.K == K` |
| `ConfigError(ValueError)` | the one error: "`HuberConfig.c` must be > 0, got -1" (model, field, rule, value); a `ValueError` subclass, so every caller catching `ValueError` keeps working |

Standard library only (`dataclasses`, `typing.Annotated`, `enum`), so Python 3.9 and no new
dependency. The fixed sets are `Choice` enums, all in `models/choices.py`; a field is typed `str` and
declares its set with `OneOf`, so a caller passes a plain string and a consumer reads one.

### 2b. Configuration models

Every consumer's settings become one model in `models/` (the repo rule for data models). Public
signatures do not change: the keyword arguments build the model, and a class also accepts the
model itself.

| model | fields | replaces |
|---|---|---|
| `LoadOptions` | split, families, units, order, format, include_gaps, heldout, preload, seed | the three `check_*` calls done twice, `split_or_none`, `family_ids(families)` pre-check |
| `ExportRequest` | format, fields, flatten_features | the `format` check and "a pandas frame carries every field" |
| `WindowSpec` | W, stride, label, layer, per_bus | `check_window_args`, the `layer` check |
| estimator configs, one per class | `npass`, `iters`, `c`, `threshold`, `cond_mult`, `rank_frac`, `reweight`, `huber_c`, `gate_factor`, `calibrate` | ten checks in `se/methods.py`, two in `se/base.py` |
| `LocalizerConfig`, `LearnedConfig`, `TrustConfig` | `fa_target`, `layers`, `hidden`, `features`, `clip`, `k` | the constructor checks in `localization/` and `trust/` |
| `FederatedSettings` | K, rounds, local_epochs, halo, grad_clip, kcl, partition | `_check_settings`, the `epochs` and `kcl` checks |
| `TimelineKnobs` | attacked_frac, ramp and Am shapes, hops, lengths, am_direction | `_check_knobs` |
| `GeneratorOptions`, `SplitFractions` | max_load_mw, frames; train/val fractions | the checks in `engine/core.py`, `generation.py`, `torch_data.py` |

### 2c. Dataset capabilities: one table

A `Capability` enum (`TIMELINE`, `BENIGN_LAYER`, `CLEAN_LAYER`, `PHYSICAL_UNITS`, `TIME_ORDER`,
`CONSECUTIVE`) and one table on the dataset mapping each to its predicate and its
message. A consumer declares what it needs, `requires = (Capability.TIMELINE,
Capability.BENIGN_LAYER)`, and the base class checks it once at `fit`/`score` entry through
`ds.require(...)`. Replaces `require_physical`, `_check_timeline`, federated `_check_units`, and
the `has_clean` / `has_benign` checks in `se`, `trust` and `se.jacobian`.

### 2d. Models that check themselves

The formulas take raw arrays by design, so each keeps a one-line declared contract at the top,
written with the engine's helpers (`expect(condition, says)`, `present(value, says)` for a value
that must not be None, `expect_ndim`, `expect_same_shape`); the eighteen hand-written checks in
`formulas.federated` and the nine in `federated.partition` are declarations now. `Partition`
validating itself in `__post_init__` (so `check_partition` disappears) is left for a follow-up.

### 2e. Parsing loose input once

`GatedPrior(gate="oracle")` builds an `OracleGate` with the localizer's interface, so the gate is
always an object with `localize(ds)` and nothing checks its type afterwards. The profile sources
(`IsoFolder`, `CsvColumn`, `RawSeries`, the per-operator feeds) follow the same rule in the next pull
request. The parsers of loose input (`family_ids`, `registry.system_id`, `release_tuple`,
`engine.core._line_id`) are each the one place their input is parsed and declare it with
`present`/`expect`; turning them into value objects (`Family`, `SystemId`, `Release`, `LineRef`) is
left for a follow-up.

### 2f. Runtime data conditions

They stay where the data is read, raised as named errors from `fdia_graph/errors.py`
(`NoBenignRecords`, `NotFitted`, `NoAdmissibleTarget`, `GridIslanded`), subclassing the built-in
type they raise today, so no caller's `except` changes.

## 3. Keeping it that way

`tools/readability.py` gains one check: `raise ValueError` / `raise TypeError` only in
`validation.py`, `errors.py` and `models/`. CONTRIBUTING gets the rule: a new argument is a field
with a rule on a config model, never a check in a function body.

## 4. Status

| part | state |
|---|---|
| engine, `ConfigError`, `MissingCapability` (`models/validation.py`) | done |
| every fixed set as a `Choice` (`models/choices.py`) | done |
| 20 config models (`models/config.py`), every constructor and entry point building its model | done |
| dataset capabilities (`dataset.base.CAPABILITIES`, `ds.require`) | done |
| formula and partition contracts as `expect` / `present` declarations | done |
| named errors for data conditions (`errors.py`) | done |
| readability rule: no `raise ValueError` / `TypeError` outside `models/` | done, with a test that the package has none |
| profile sources and operator feeds | next pull request |
| `Partition` self-check; `Family`, `SystemId`, `Release`, `LineRef` value objects | follow-up |

Removed helpers (`check_split`, `check_units`, `check_order`) keep a deprecated alias for one minor
version; `require_physical` stays as the named requirement of the estimators and delegates to the
capability table.
