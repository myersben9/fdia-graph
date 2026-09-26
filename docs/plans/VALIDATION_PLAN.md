# Plan: validation in one place

Goal: every check on an input is declared once, on the data model that carries the input, and
enforced by one engine. No function body validates its own arguments. A reader finds what a value
may be next to the field that holds it, the error messages come from one place, and adding a
parameter means adding a field with its rule rather than another `if ...: raise` at the top of a
function.

This is the plan only. It keeps the compatibility promise of the readability series: public
signatures, accepted values, generated files and numbers stay the same; only the wording of the
error messages changes, and it becomes uniform.

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

### 2a. The engine: `fdia_graph/validation.py`

A rule is a small declared object; a field carries its rules in its annotation; one base class runs
them.

```python
@dataclass(frozen=True)
class HuberConfig(Validated):
    c: Annotated[float, Positive()] = 1.5
    npass: Annotated[int, AtLeast(1)] = 40
    iters: Annotated[int, AtLeast(1)] = 8
    reweight: Optional[Reweight] = None          # a Choice: converted from its string by the engine
```

| piece | does |
|---|---|
| `Rule` (`Positive`, `AtLeast(n)`, `InRange(lo, hi, closed=...)`, `Integer`, `Finite`) | one predicate and the phrase it prints ("must be > 0") |
| `Validated.__post_init__` | the one place validation runs: converts every `Choice`-typed field from its string, applies each field's rules, then the model's `invariants()` |
| `invariants()` | cross-field rules declared as (condition, phrase) pairs: `train_frac + val_frac < 1`, `W <= T`, `partition.K == K` |
| `ConfigError(ValueError)` | the one error: "`HuberConfig.c` must be > 0, got -1" (model, field, rule, value); a `ValueError` subclass, so every caller catching `ValueError` keeps working |

Standard library only (`dataclasses`, `typing.Annotated`, `enum`), so Python 3.9 and no new
dependency. The `Choice` enums (open PR #147) stay: they become the field types, and converting
them moves from the call sites into the engine.

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
`CONSECUTIVE`, `SPLIT`) and one table on the dataset mapping each to its predicate and its
message. A consumer declares what it needs, `requires = (Capability.TIMELINE,
Capability.BENIGN_LAYER)`, and the base class checks it once at `fit`/`score` entry through
`ds.require(...)`. Replaces `require_physical`, `_check_timeline`, federated `_check_units`, and
the `has_clean` / `has_benign` checks in `se`, `trust` and `se.jacobian`.

### 2d. Models that check themselves

`Partition` validates its assignment in `__post_init__` (today `check_partition` and
`partition_from_assignment` do it from outside). The formulas take raw arrays by design, so they
keep a one-line declared contract at the top, written with the engine's shared helpers
(`expect_shape(x, "[n, N]")`, `expect_same_shape(a, b)`), which build the message; the eighteen
hand-written checks in `formulas.federated` become declarations.

### 2e. Parsing loose input once

Value objects that parse themselves at the public boundary: `Family` (a name, an alias or a
code), `SystemId` ("ieee118" or 118), `Release` ("v0.8.3"), `LineRef` (a name or an index). A
config field typed as one of them is parsed by the engine like a `Choice`. `GatedPrior(gate=...)`
takes an `OracleGate` object with the localizer's interface instead of the string "oracle". The
profile sources (`IsoFolder`, `CsvColumn`, `RawSeries`, the per-operator feeds) follow the same
rule and are ready on a local branch.

### 2f. Runtime data conditions

They stay where the data is read, raised as named errors from `fdia_graph/errors.py`
(`NoBenignRecords`, `NotFitted`, `NoAdmissibleTarget`, `GridIslanded`), subclassing the built-in
type they raise today, so no caller's `except` changes.

## 3. Keeping it that way

`tools/readability.py` gains one check: `raise ValueError` / `raise TypeError` only in
`validation.py`, `errors.py` and `models/`. CONTRIBUTING gets the rule: a new argument is a field
with a rule on a config model, never a check in a function body.

## 4. Steps

One pull request each, each behaviour-neutral (same accepted values, strict frozen suite
unchanged), each with a CHANGELOG entry for the message wording.

| step | scope |
|---|---|
| 1 | the engine (`validation.py`, `errors.py`), the `Choice` enums (PR #147, reworked onto the engine), `LoadOptions`, `ExportRequest`, `WindowSpec` |
| 2 | dataset capabilities |
| 3 | estimator, localizer, trust and federated configs |
| 4 | `TimelineKnobs`, `GeneratorOptions`, `SplitFractions`, the profile sources |
| 5 | self-checking models, formula contracts, the parsing value objects, named runtime errors |
| 6 | the readability check that keeps new validation in one place |

Removed helpers (`check_split`, `check_units`, `check_order`, `check_partition`,
`require_physical`) keep a deprecated alias for one minor version.
