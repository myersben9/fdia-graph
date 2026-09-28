# Plans (archive)

The design documents behind the 0.16 and 0.17 refactors, kept because they record why the code
is shaped the way it is: which measures were chosen and which rejected, what the compatibility
promise covers, and the decisions Ben took at each step. They are history, not instructions;
`CONTRIBUTING.md` says how to change the package today.

| plan | what it decided |
|---|---|
| `READABILITY_PLAN.md` | the six readability measures and their limits, the frozen-reference safety net, one attack frame for shards and streams, the `formulas/` kernel rule, the loader as concerns |
| `RESTRUCTURE_PLAN.md` | the target layout of the formula kernel and its catalogue, the closed-form Jacobian decision |
| `DATA_MODELS_PLAN.md` | typed models for everything a function returns, the dual-access `Bundle`, the `models/` package |
| `FIELD_GROUPS_PLAN.md` | field groups for the fields the data models repeat (implemented) |
| `VALIDATION_PLAN.md` | **active**: every input check declared on a data model and enforced by one engine; dataset capabilities in one table; the six-step series |
| `REFERENCE_STUDY_PLAN.md` | **proposed**: twelve changes taken from power-grid-model and OGB (standalone evaluators, per-record estimation status, provenance and seeds, release metadata and dataset cards, a timeline file checker, collect-all validation errors), ranked, with the open decisions and the sequence |
| `WU_MSFDIA_PLAN.md` | **accepted**: generate the multi-snapshot families `At` and `Am` by solving the [WU26] optimization (fewest tampered devices, eq. 12) under its constraints and line-overload goal, deprecate the single-snapshot families for generation, and move all attack generation into the engine's attack mixin; decisions D1 to D9 |
| `ONE_DATASET_PLAN.md` | **active**: one generator, one HDF5 file, one loader with a seeded order; the seventh family `Am`; data release v0.8.0 |
