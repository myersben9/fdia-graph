# Repo roadmap

The package is two halves, and the folder structure shows it: the top level is the **SDK** (load and
serve data), `engine/` is the **theory** (power-flow physics, meter models, attack math). Loading runs
on the base install; the generation side — `engine/` plus its drivers `generation.py`/`profiles.py` —
needs the `[generate]` extra. Users only touch `src/fdia_graph/__init__.py`'s `fg.*` functions.

```
src/fdia_graph/          (the package root)
├── __init__.py                      fg.* public API
├── dataset.py registry.py download.py     SDK: load path
├── streams.py torch_data.py               SDK: stream path
├── generation.py profiles.py              SDK: generation drivers
├── se/                              state estimation (WLS, robust, subspace prior)
└── engine/                          theory: FdiaGenerator
    ├── core.py                        assembly + targeting
    ├── measurement.py                 meters + noise  h(x)
    ├── physics.py                     AC solves, Ybus
    ├── attacks.py                     the six families
    └── base.py                        shared typed contract
```

How the three paths connect (generate produces the shard the load path serves):

![Block diagram: the three fg.* paths and the files behind them](fig_roadmap.svg)

## SDK (package root `src/fdia_graph/`) — load and serve data

| File | What it is |
|---|---|
| `__init__.py` | Public API. Thin wrappers with lazy imports so `load()` users never need torch/pandapower. |
| `dataset.py` | `FdiaGraph`: Dataset over one `.h5` shard. Splits, family filters, units, dict/PyG loaders. |
| `registry.py` | Dataset version control. `(name, release)` → download spec; `register_local` for generated sets. |
| `download.py` | Fetch a shard to `~/.cache/fdia_graph`, sha256-verified, atomic rename. |
| `streams.py` | Continuous attacked time series (`generate_stream`/`load_stream`) + windowing for LSTMs. |
| `torch_data.py` | `pyg_stream`/`torch_windows`: streams as ready PyG graphs / per-bus sequence tensors. |
| `generation.py` | `generate()`: drives the engine over a state pool, writes + registers the shard. |
| `profiles.py` | ISO load profiles (NYISO/CAISO/ERCOT) → normalized scaling → AC operating-state pools. |
| `se/` | sklearn-style state estimation: `SEBase` chord-Newton core + WLS/robust/prior classes. |

## `engine/` — the physics, meters, and attack math

| File | What it is |
|---|---|
| `core.py` | `FdiaGenerator` assembly: grid setup, meter plan, RNG, attack targeting. |
| `measurement.py` | Mixin: meter placement + accuracy-class noise (per-meter bias + per-scan jitter); `h(x)`. |
| `physics.py` | Mixin: AC solves, Ybus, emit exact measurements from a stored state. |
| `attacks.py` | Mixin: the six attack families (`Aq Ad As Ar At Al`). |
| `base.py` | Typed attribute contract the three mixins share (no runtime behavior). |

## Reading order for new students

1. `README.md` — install + quickstart.
2. `docs/DATA_DICTIONARY.md` — what every array and shape means (`node_x`, `edge_attr`, ...).
3. This page — which file does what.
4. `docs/CONCEPTS_TO_CODE.md` — paper formulas → the code that implements them.
5. `docs/EXAMPLES.md` — full training examples to copy from.
