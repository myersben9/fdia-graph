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

