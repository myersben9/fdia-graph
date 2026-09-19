# Plan: field groups for the repeated model fields

Status: implemented 2026-09-18 with option B and the proposed decisions (labels and record ids split,
`TypeError` for a missing required field). Ben: "our models in data.py are pretty repetitive, can we
consolidate models that reuse fields?"

## 1. The repetition, measured

Ten models hold 33 distinct fields; 18 of those fields appear in two or more models, and the four
measurement arrays appear in eight.

| field | models that carry it |
|---|---|
| `node_x`, `node_m`, `edge_x`, `edge_m` | RecordBundle, BatchBundle, ArraysBundle, ShardArrays, Stream, Scan, Frame, Record (8) |
| `y` | the same minus Scan (7) |
| `family`, `stealthy`, `timestep`, `temporal_delta`, `swing` | RecordBundle, BatchBundle, ArraysBundle, ShardArrays, Stream or Frame, Record (6) |
| `seq_id` | RecordBundle, BatchBundle, ArraysBundle, ShardArrays, Record (5) |
| `edge_index`, `clean`, `edge_clean` | RecordBundle, BatchBundle, ArraysBundle, Stream (4) |
| `edge_attr`, `edge_clean_full` | three each |

Each occurrence repeats the field's type and its one-line meaning, so a change to what `swing`
means has to be made in six places, and the data dictionary renders the same sentence six times.

What is *not* repetition: the leading axis. A record's `node_x` is `[N, 4]`, a batch's `[B, N, 4]`,
a split's `[n, N, 4]`, a stream's `[T, N, 4]`. That difference is the documentation a reader wants,
and it belongs to the bundle, not to the field.

## 2. Two ways to consolidate, and the one proposed

| option | removes | costs | verdict |
|---|---|---|---|
| A. build bundles from a field table with `make_dataclass` | all repetition | pyright and IDEs no longer see the fields, which is why the models exist | no |
| B. field groups as dataclass mixins, each bundle inherits the groups it needs | the repeated types and meanings | dataclass inheritance fixes field order and forces defaults last; required-ness and precise per-bundle types need a small mechanism | **yes, with the two mechanisms below** |
| C. keep the classes, move only the meanings into one table the doc generator reads | the repeated sentences | the repeated fields stay | fallback if B proves fiddly |

## 3. Design (option B)

### 3.1 The groups (`models/fields.py`)

Frozen dataclass mixins with no behaviour, every field with a default of `None`, typed `Array` (a numpy
array or a torch tensor) or `Scalars` (an int on a record, a vector on a batch or a split), the
meaning written once per field:

| group | fields |
|---|---|
| `ScanFields` | `node_x`, `node_m`, `edge_x`, `edge_m` |
| `LabelFields` | `y`, `family` |
| `RecordIds` | `stealthy`, `seq_id`, `timestep` |
| `TemporalFields` | `temporal_delta`, `swing` |
| `CleanFields` | `clean`, `edge_clean`, `edge_clean_full` |
| `GraphFields` | `edge_index`, `edge_attr` |
| `StreamLayers` | `benign`, `edge_benign` |

Each field's comment names the trailing shape only (`[..., N, 4]`); the bundle's docstring names
the leading axis (none, `B`, `n`, `T`).

### 3.2 Two mechanisms in `Bundle`

1. **`_order`**: the dict-key order of a bundle, declared explicitly. Today `_tail` moves one
   required field to the end; `_order` generalizes it to the whole key order, so the dict view a
   user sees stays exactly what it is now while the dataclass field order follows the inheritance.
   `_tail` stays as a shorthand.
2. **`_required`**: the fields a bundle must have; `__post_init__` raises `TypeError` naming any
   that is None. Groups can then give every field a `None` default (what inheritance needs) while
   `RecordBundle` still refuses to exist without `node_x`, as it does today.

### 3.3 The bundles after

| bundle | groups | own fields | `_required` |
|---|---|---|---|
| `RecordBundle` | Scan, Label, Ids, Temporal, Clean, Graph | none | scan, labels, ids, `edge_index` |
| `BatchBundle` | Scan, Label, Ids, Temporal, Clean, Graph | none | scan, `y` |
| `ArraysBundle` | Scan, Label, Ids, Temporal, Clean, Graph | `edge_reactance` (and `edge_attr` arrives with Graph, never filled) | none (fields are requested) |
| `ShardArrays` | none, kept explicit: every field is required and numpy, and the writer's type checks rely on both | all twelve | all |
| `Stream` | Scan, Label, Ids (only `timestep` filled), Temporal, Clean (`edge_clean_full` never filled), Graph, StreamLayers | `episodes`, `system`, `attacked_frac` | the arrays and `episodes` |
| `Scan`, `Frame`, `Record` (NamedTuples) | unchanged | | |

`Stream` shows the one wrinkle: it carries `y`, `family`, `timestep` but not `stealthy` or
`seq_id`. Either `LabelFields` splits into `LabelFields` (`y`, `family`) and `RecordIds`
(`stealthy`, `seq_id`, `timestep`) with `Stream` taking `timestep` as an own field, or `Stream`
takes the whole group and lists the two as absent. The split is cleaner; decision 2 below.

The engine's NamedTuples stay: they are positional by design (`Scan` unpacks into `Frame`), have
no defaults, and are internal.

### 3.4 What a reader sees

`data.py` shrinks from six near-identical field lists to six short classes that say which groups
they take and what their leading axis is. `tools/models_doc.py` renders each bundle as "fields of
ScanFields, LabelFields, ... plus `edge_reactance`", and one table per group, so the dictionary
stops repeating itself too.

## 4. What does not change

- Every dict key, in the same order (`_order` is set to today's order and a test asserts it).
- Every attribute name and every `to_dict()`.
- Construction by keyword, pickling, `ordered()`, the default collate.
- Files, scores, the frozen references.

## 5. Sequence (one PR)

1. `Bundle._order` and `_required`, with tests (order kept, a missing required field raises).
2. `models/fields.py` with the six groups.
3. Rebuild the five bundles on the groups with `_order` = today's key order; `dataclasses.fields`
   names and the dict view are asserted equal to the current ones in a test before the switch.
4. `tools/models_doc.py` renders groups; regenerate the dictionary section.
5. Strict frozen suite, pyright, readability gate, changelog.

## 6. Decisions for Ben

1. Option B (proposed) or the fallback C.
2. Split `LabelFields` into labels and record ids (proposed), or let `Stream` carry two absent fields.
3. `_required` failures raise `TypeError` (proposed, matching a missing dataclass argument today) or
   `ValueError`.
