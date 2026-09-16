# Plan: structured data models for what functions return and objects hold

Goal (Ben, 2026-09-16): where a function returns a bundle of things, or an object holds a bundle
of related arrays, the bundle is a named, typed model rather than a tuple indexed by position or
a dict indexed by string. A reader then sees the fields in one place, pyright checks them, and a
caller that receives the model can hand it on to the next function unchanged.

This is the plan only. It follows the readability series (`READABILITY_PLAN.md`) and keeps its
compatibility promise: nothing a user of the package can see changes.

## 1. What already exists

The readability series introduced eight models where the old code passed tuples or closures:

| model | kind | holds | used by |
|---|---|---|---|
| `engine.records.Frame` | NamedTuple | one emitted scan: measurements, masks, label, stealthy flag, designed magnitudes, benign twin | both generators |
| `engine.records.FrameKnobs` | NamedTuple | the attack settings of a run | both generators |
| `generation.Record` | NamedTuple | one finished shard record: the scan, ids, the two temporal features | the writer |
| `formulas.network.BranchModel` | NamedTuple | the per-branch pi model (r, x, b, g, tap, shift, status) | `branch_admittances`, the loader |
| `generation._FrameContext` | dataclass | what every record of a run shares (generator, pool, scale, knobs, magnitude log) | the record functions |
| `streams._StreamBuffers` | class | the per-frame layers of one stream and the previous emitted frame | the timeline walk |
| `streams._StreamPlan` | dataclass | which families rotate, the ramp shape, the attacked fraction | the timeline walk |
| `dataset._RecordFilter` | NamedTuple | which records a view keeps | `_record_mask` |

So the pattern is in place; what is left is to apply it to the rest of the package, with one
extra rule for the public dicts.

## 2. Inventory: what is still a bare tuple or dict

Surveyed 2026-09-16 on the step 6 branch (every function whose return annotation is a Dict or
Tuple, plus the generator's attribute groups).

### 2a. Internal, no compatibility constraint (change freely)

| today | shape | proposed model | notes |
|---|---|---|---|
| `MeasurementMixin.emit_from_state`, `emit` | 4-tuple `(nx, nm, ex, em)` | `Scan(node_x, node_m, edge_x, edge_m)` | called at seven sites, always unpacked positionally; `Frame` already holds the same four fields, so `Frame` can be built from a `Scan` |
| `SEBase._truth_of` | dict `{"x", "thsl"}` | `TrueState(x, slack_angle)` | eleven call sites read `tr["x"]`, `tr["thsl"]` |
| `AttackMixin.lra_delta`, `_lra_for_line` | tuples `(delta, buses)`, `(delta, buses, flow_change)` | `Redistribution(delta, buses, line_flow_change)` | the plausibility band check reads the pieces |
| `PhysicsMixin.resolve_states` | tuple `(X, kept)` | `ResolvedPool(states, converged)` | |
| `FdiaGraph._admittances` | dict `{"ybus", "yf", "yt"}` | `Admittances(ybus, yf, yt)` | the three properties `ybus_np`, `yf_np`, `yt_np` read it; `branch_admittances` returns the same three, so the kernel returns the model too |
| `generation._stack_records` | dict of arrays | `ShardArrays` (dataclass with one field per dataset) | the writer reads it by key |
| `registry` specs (`resolve`, `_asset_spec`, `download.ensure_local(spec)`) | dict `{"kind", "name", "file", "release", "repo", "sha256", "path"}` | `AssetSpec` (frozen dataclass) | four producers, one consumer; `ensure_local` branches on `kind` |
| `download._asset_url` | tuple `(url, headers)` | `DownloadTarget(url, headers)` | |
| `FdiaGenerator` attribute groups | `edge_r … bus_shunt_b` (12 arrays), `M` dict + `flow_meter`, six `bias_*` vectors, six `outage_*` fields | `g.branch: BranchModel` (the model already exists), `g.meters: MeterPlan(vbus, pmu, inj, flow)`, `g.bias: MeterBias(pi, qi, v, va, pf, qf)`, `g.contingency: Outage(line, pos, name, from_bus, to_bus, base_flow_mw)` (`INTACT` when no line is opened) | the old attribute names stay as read-only properties for one minor version (the writer, the docs scripts and the temporal harness read them) |
| `localization.LocalizerBase._pull` | dict of arrays | `LocalizerInputs` | internal |

### 2b. Public API returning dicts (compatibility-bound)

| today | who depends on it |
|---|---|
| `FdiaGraph.__getitem__` dict record, `collate` batch dict | every training loop on the package, `torch.utils.data` default collate, our examples |
| `FdiaGraph.to_numpy`, `to_torch`, `to_tf` | our SE and localization modules, the temporal harness, users |
| `generate_stream` / `load_stream` stream dict | `windows`, `pyg_stream`, `torch_windows`, the deck scripts, users |
| `SEBase.score`, `LocalizerBase.score` nested dicts | `docs/*/make_report.py`, the frozen tests, users' tables |
| `JacobianFeatures.transform` dict `{"bus", "global", "dx_hat", "r_perp"}` | the learned localizers, `JacobianWeighting`, the deck scripts |
| `FdiaGraph.summary`, `registry.list_datasets` | users |
| `line_outage_candidates` lists of dicts | users, `docs` |

These cannot become plain dataclasses without breaking `out["node_x"]`, `**out`, `for k in out`,
JSON dumping, and the default collate. The rule for them is section 3.

## 3. Design rules

1. **Value bundles that get unpacked are NamedTuples** (`Scan`, `TrueState`, `Redistribution`,
   `Admittances`, `DownloadTarget`): immutable, cheap, `a, b = f()` still works, fields have names
   and types.
2. **Larger models with behaviour are frozen dataclasses** (`AssetSpec`, `MeterPlan`, `MeterBias`,
   `Outage`, `ShardArrays`): named fields, a `__post_init__` for validation where useful, methods
   where they belong (`AssetSpec.cache_name()`).
3. **Public dict returns become dual-access models.** `fdia_graph.models.Bundle` is a frozen
   dataclass that is also a real `dict` subclass: `__post_init__` fills the dict with the non-None
   fields, the dict side refuses writes, and pickling goes through the fields.

   ```python
   @dataclass(frozen=True, eq=False)
   class TrueState(Bundle):
       x: np.ndarray      # the 2N-1 state per record
       thsl: np.ndarray   # the slack angle reference per record
   ```

   A bundle keeps every existing use working (`isinstance(out, dict)`, `out["node_x"]`, `**out`,
   `for k in out`, `json.dump` of a score table, PyTorch's default collate, which batches a dict
   key by key) and adds `out.node_x`, pyright-checked field names, and one docstring naming every
   field. Optional layers are `Optional[...]` fields; absent ones are not in the dict, so
   `"swing" in out` keeps its meaning. A dict key that is not a valid attribute name (`"global"`)
   is declared with a trailing underscore and mapped through `_keys`; a required field the old
   dict listed last (`geo`) is named in `_tail` so the key order does not change. A `dict`
   subclass rather than a `Mapping` because users test `isinstance(x, dict)` and `json.dump`
   score tables directly.
4. **Serialization is explicit.** Every model has `to_dict()`; a bundle is already the dict
   `savez` and `json.dump` see, so files do not change.
5. **Models live next to their producers** (superseded by step 5, section 6: they live in the
   `models/` package and are re-exported from their producers), not in one giant module: `engine.records.Scan`,
   `se.base.TrueState`, `dataset.RecordBundle`, and so on. `fdia_graph.models` holds only the
   `Bundle` base (re-exporting the producers' models from it would import the whole package);
   the data dictionary lists them (step 4).
6. **A public bundle keeps the shape of the dict it replaces**, whatever its size: the stream
   dict has eighteen keys and `Stream` has eighteen fields, because splitting it into a pair would
   change what `s["node_x"]` and `**s` mean for every caller. The readability measures apply to
   functions, not to the field count of a record; a long field list with one comment per field is
   the documentation the old dict never had.

## 4. Sequencing

1. `fdia_graph.models.Bundle` with its tests (dict behaviour, JSON, default collate, `**`).
2. The internal models of 2a, one PR: `Scan`, `TrueState`, `Redistribution`, `ResolvedPool`,
   `Admittances`, `ShardArrays`, `AssetSpec`, `DownloadTarget`. Pure renames of plumbing; the
   strict frozen tests are the proof.
3. The generator's attribute groups (`g.branch`, `g.meters`, `g.bias`, `g.outage`) with the old
   attribute names as properties for one minor version; the writer and the docs scripts move to
   the models in the same PR.
4. The public dual-access models, one PR: `EstimatorScores` and `LocalizerScores` with their
   row models, `JacobianOutputs`, `Stream`, `LineCandidate`, `Summary`, then `RecordBundle`,
   `BatchBundle` and `ArraysBundle` (the loader). One PR rather than four because the base class
   change (`Mapping` to `dict` subclass) touches them all and each is a few lines at its producer.
5. Docs: `docs/reference/DATA_DICTIONARY.md` gains a "models" section listing every public
   model and its fields, generated from the dataclasses so it cannot drift.

Verification is the same net as the readability series: the strict frozen tests, pyright, and
the readability gate; plus, for every public bundle, a test that the dict view equals the
previous dict exactly.

## 4a. Status

| step | pull request | state |
|---|---|---|
| 1 `Bundle` and the internal models | #69 | merged |
| 2 the generator's attribute groups | #69 | merged |
| 3 public dual-access bundles | #70 | merged |
| 4 the data dictionary's models section | #71 | merged |
| 5 one `models/` package | #72 | open |

Decisions taken 2026-09-16: dual-access bundles for the public dicts; the DataLoader test came
first (`tests/test_models.py`); the base class is named `Bundle`; the generator's old attribute
names are properties that warn for one minor version.

## 5. Decisions for Ben

1. Dual-access models for the public dicts (proposed), or a hard switch to dataclasses in a
   major version. Dual access costs one small base class and keeps every user's code working.
2. Whether `__getitem__` records become a `RecordBundle` now or only after PyG and the default
   collate are exercised in a test on a real DataLoader (proposed: the test first, in step 1).
3. The name `Bundle` for the base class, or `Model`, or `Struct`.
4. Whether the generator's old attribute names (`g.edge_r`, `g.M`, `g.bias_pi`, ...) are removed
   after one minor version or kept for good as properties (they are private in spirit but
   undocumented code reads them).

## 6. Step 5: one `models/` package (planned 2026-09-16, Ben's request)

Steps 1 to 4 left thirty model classes in thirteen files, each next to its producer (rule 5).
That avoided one giant module and import cycles, but a reader has no single place to look, which
is what Ben asked to fix. Step 5 moves every model into a `fdia_graph/models/` package grouped
by what the data is, and keeps every old import path working.

### 6.1 Layout

`models.py` becomes a package. Each file holds one kind of data, in the order a reader meets it:

| file | holds | moved from |
|---|---|---|
| `models/base.py` | `Bundle` (unchanged) | `models.py` |
| `models/grid.py` | the static description of a system: `BranchModel`, `Admittances`, `MeterPlan`, `MeterBias`, `Outage`, `INTACT` | `formulas/network.py`, `engine/base.py` |
| `models/frames.py` | what the generators pass around per scan: `Scan`, `Frame`, `FrameKnobs`, `Record`, `Redistribution`, `ResolvedPool` | `engine/records.py`, `generation.py`, `engine/attacks.py`, `engine/physics.py` |
| `models/data.py` | what users get back: `RecordBundle`, `BatchBundle`, `ArraysBundle`, `Summary`, `ShardArrays`, `Stream`, `TrueState` | `dataset.py`, `generation.py`, `streams.py`, `se/base.py` |
| `models/scores.py` | the result tables: `ErrorPair`, `EstimatorScores`, `OverallMetrics`, `BenignMetrics`, `FamilyMetrics`, `LocalizerScores`, `JacobianOutputs` | `se/base.py`, `localization/base.py`, `se/jacobian.py` |
| `models/assets.py` | how files are found: `AssetSpec`, `DownloadTarget`, `LineCandidate` | `registry.py`, `download.py`, `engine/core.py` |
| `models/__init__.py` | re-exports all of the above with an `__all__`, so `fdia_graph.models` is the one namespace to browse | |

Every class above is pure data: none has a method of its own or reads a module constant, so
each moves as written (with its docstring and field comments). Constants that describe families
(`RESOLVE_FAMILIES`, `CORRUPT_KIND`, `SINGLE_SHOT_ORDER`, `RAMP_FAMILY`, `LRA_FAMILY`) stay in
`engine/records.py`, since they belong to the attack logic, not to the record shape.

Four private workflow objects do **not** move: `generation._FrameContext`, `streams._StreamBuffers`,
`streams._StreamPlan`, `dataset._RecordFilter`. They are the state of one algorithm, read only
by the loop next to them, so they stay beside it.

### 6.2 Rules

1. **Models import only numpy and typing** (`models/data.py` may import `Bundle` from `base`,
   `models/scores.py` may import `ErrorPair` and the metric rows from itself). Producers import
   from `models`; nothing under `models/` imports a producer. Import cycles are therefore
   impossible. (The parent package still imports the loader, h5py included, on any import.)
2. **Every old import path keeps working.** Each producer module re-exports the names it used to
   define (`from .models.data import Stream` at the top of `streams.py`, and so on), so
   `from fdia_graph.streams import Stream`, `from fdia_graph.se.base import TrueState`,
   `from fdia_graph.engine.records import Frame` are unchanged. The re-exports of public names
   stay for good; those of private names for one minor version, per the compatibility promise.
3. **A test keeps it from scattering again.** `tests/test_models_package.py` walks every module of
   the package and fails if a `Bundle` subclass or a `NamedTuple` with a public name is defined
   outside `models/`. It also checks that `fdia_graph.models.__all__` names every one of them, and
   that no module under `models/` imports anything but numpy, typing and dataclasses.
4. **The data dictionary follows the package.** `fdia_graph.models.PUBLIC` names the bundles a
   user receives (`__all__` also holds the internal models and `Bundle`); `tools/models_doc.py`
   iterates `PUBLIC` instead of its hand-kept list.
5. **No behaviour change.** The strict frozen tests, pyright and the readability gate are the
   proof, as in every step before.

### 6.3 Sequence (one PR)

1. Create the package: move `models.py` to `models/base.py`, add the five domain files with the
   classes moved verbatim, and `__init__.py` with `__all__` in the table's order.
2. In each former home, replace the class with a re-export line and switch the module's own uses
   to the import. Producers that built models keep building them the same way.
3. Add `tests/test_models_package.py`; point `tools/models_doc.py` at `__all__` and regenerate
   the data dictionary section (it should render byte-identical apart from the module column).
4. Run the strict frozen suite, pyright, the readability gate and the deprecation-as-error run.
5. `CHANGELOG.md`: "models live in `fdia_graph.models`; old import paths unchanged".
   `docs/reference/FORMULAS.md` and `docs/ROADMAP.md` rows that name `engine/records.py` or
   `formulas/network.py` for a model get the new path.

### 6.4 Decisions (taken 2026-09-16, Ben delegated the call)

1. File names by kind of data: `grid`, `frames`, `data`, `scores`, `assets`. Grouping by producer
   would only reproduce today's scatter inside one folder.
2. The internal NamedTuples move too. One place to look is the point; a reader of `Frame` should
   not have to know it is internal to find it.
3. `TrueState` lives in `data.py`: it is per-record truth a user can ask for, not a score.
4. Re-exports of public names stay for good. They cost one line each and keep every published
   example and every user's import working; there is nothing to gain from removing them.
