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
3. **Public dict returns become dual-access models.** A small base class in a new module
   `fdia_graph.models`:

   ```python
   @dataclass(frozen=True)
   class Bundle(Mapping[str, Any]):
       """A typed record that still behaves as the dict it used to be."""
       def __getitem__(self, k): return getattr(self, k)
       def __iter__(self): return iter(f.name for f in fields(self) if getattr(self, f.name) is not None)
       def __len__(self): return sum(1 for _ in self)
       def to_dict(self): return {k: self[k] for k in self}
   ```

   A `RecordBundle`, `BatchBundle`, `StreamBundle`, `ScoreTable`, `JacobianOutputs` built on it
   keep every existing use working (`out["node_x"]`, `**out`, `json.dump` of a table of scalars, the
   default collate treats a `Mapping` as a dict) and add `out.node_x`, pyright-checked field
   names, and one place that documents what each field is. Optional layers are `Optional[...]`
   fields; absent ones do not appear in iteration, so `"swing" in out` keeps its meaning.
4. **Serialization is explicit.** Every model has `to_dict()`; the frozen references, the
   result JSON files and `savez` go through it, so files do not change.
5. **Models live next to their producers**, not in one giant module: `engine.records.Scan`,
   `se.base.TrueState`, `dataset.RecordBundle`, and so on. `fdia_graph.models` holds only the
   `Bundle` base and re-exports the public ones for discoverability.
6. **The readability measures apply**: a model with more than about twelve fields is two models
   (the stream dict has eighteen keys: `StreamBundle` holds the per-frame layers, `StreamGraph`
   the static graph and masks, and the stream is the pair).

## 4. Sequencing

1. `fdia_graph.models.Bundle` with its tests (dict behaviour, JSON, default collate, `**`).
2. The internal models of 2a, one PR: `Scan`, `TrueState`, `Redistribution`, `ResolvedPool`,
   `Admittances`, `ShardArrays`, `AssetSpec`, `DownloadTarget`. Pure renames of plumbing; the
   strict frozen tests are the proof.
3. The generator's attribute groups (`g.branch`, `g.meters`, `g.bias`, `g.outage`) with the old
   attribute names as properties for one minor version; the writer and the docs scripts move to
   the models in the same PR.
4. The public dual-access models, one PR each in order of blast radius: `ScoreTable` (smallest,
   used by the report scripts), `JacobianOutputs`, `StreamBundle` + `StreamGraph`, then
   `RecordBundle` and `BatchBundle` (the loader).
5. Docs: `docs/reference/DATA_DICTIONARY.md` gains a "models" section listing every public
   model and its fields, generated from the dataclasses so it cannot drift.

Verification is the same net as the readability series: the strict frozen tests, pyright, and
the readability gate; plus, for every public bundle, a test that the dict view equals the
previous dict exactly.

## 4a. Status

| step | pull request | state |
|---|---|---|
| 1 `Bundle` and the internal models | #69 | open |
| 2 the generator's attribute groups | #69 | open |
| 3 public dual-access bundles | next | |
| 4 the data dictionary's models section | after 3 | |

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
