# fdia_graph.nyfed

The New York grid package, moved into the SDK from its own repository on 2026-11-02. Until it is brought under the SDK's rules module by module (`DEFERRED` in `tools/readability.py`), these rules hold here.

`docs/nyfed/ARCHITECTURE.md` is the contract for this package. `docs/nyfed/PLAN.md` says what each block builds and links to the source it comes from.

## Stay inside the architecture

- Only create the files, classes, records and functions listed in `docs/nyfed/ARCHITECTURE.md`, with the names and signatures given there.
- If a change needs something that is not listed (a new file, class, public method, record field or dependency), stop and ask. Once it is approved, update `docs/nyfed/ARCHITECTURE.md` in the same change.
- Imports point one way: `grid` -> `scenario`, `sensing` -> `attacks` -> `timeline` -> `io` -> `dataset`. `dataset.py` imports only `io.py` and `grid/network.py`.
- pandapower and Pyomo are imported only in `scenario/solver.py`. Optimization problems are Pyomo models. h5py only in `io.py`. Random generators are made only by `config.rng()` and passed in.
- Records are frozen dataclasses of numpy arrays. A class owns state, cached work or a shared interface. Stateless math is a plain function.
- Compose classes. `AttackFamily` is the only base class. No mixins.
- Measurements are the flat vector `z` with the `MeterPlan` table. A client's data is `z[:, meters.owner == c]`. A `ClientView` never exposes another client's meters.

## Keep it simple

- Build only the block you were asked for. Do not start later blocks or add code for them.
- No extra tooling, CI steps, config systems, registries, plugin layers or helper frameworks.
- Do not add header comments that say where code was copied from.
- Do not add constraints, bounds or rules that are not in the plan or in a cited equation. If one seems needed, ask.
- When porting from fdia-graph, take the logic at the linked lines and rewrite it into this layout. Do not bring over its mixins, bundles or legacy paths.

## Work

- Tests go in the matching `tests/nyfed/test_nyfed_<package>.py`. They read the inputs (`data/`, `out/`) under `NYFED_ROOT`. Run `pytest tests/nyfed` before saying a block is done, and report the numbers the block produced.
- Never commit or push unless asked. Never add AI attribution to commits.
