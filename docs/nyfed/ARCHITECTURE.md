# Architecture

In fdia-graph since 2026-11-02 as `fdia_graph.nyfed`: the package paths below are under `src/fdia_graph/nyfed/`, the tests are `tests/nyfed/test_nyfed_<package>.py` (inputs under `NYFED_ROOT`), the tools `tools/nyfed/`, and the results files sit in `results/` as they were.

## Principles

A setting is an argument of the function or constructor that uses it. Code never changes a library object's
settings, or a shared record's arrays, after it is built: a variant is a copy (`with_overshoot`, `with_voltage_margin`).

Each package's `__init__.py` says what its stage does and lists its public names (`__all__`), imported on first
use so that reading a dataset never loads pandapower or Pyomo. A name outside `__all__` is internal and may change.

1. A class owns state, cached work, or a shared interface. A plain function is stateless math.
2. Classes are composed, not stacked. One small interface (`AttackFamily`) is the only base class.
3. Data records are frozen dataclasses of numpy arrays.
4. One job per file. A file stays short enough to read in one sitting.
5. Dependencies point one way, shown in the diagram below.
6. pandapower lives in `scenario/solver.py`, h5py in `io.py`, random number creation in `config.py`. Pyomo models are
   built in `scenario/solver.py` and in the attack OPFs (`attacks/opf_attack.py`, `attacks/interface_attack.py`), and
   every one of them gets its Ipopt from `solvers.py`, which checks that it runs.
7. Measurements are one flat vector `z` per frame with a meter table beside it. A client's data is `z[:, meters.owner == c]`.

## Files

```
src/fdia_graph/nyfed/
  __init__.py          the package map (its docstring) and the main entry points, loaded on first use
  _lazy.py             attach: the public names of a package, imported on first use
  config.py            Settings, rng()
  build.py             fetch_inputs, production_builder (the release build in one place)
  io.py                TimelineWriter, open_timeline, read_pool, write_pool, SPLITS, FAMILIES
  solvers.py           ipopt (the checked Ipopt every OPF runs on; SolverUnavailable when it cannot run)
  tables.py            read_rows (the input CSV tables, read whole and closed)
  experiments.py       study_days, variant, load_calib, add_study_arguments, run_parallel (what every study tool repeats)
  dataset.py           Dataset, ClientView
  grid/
    __init__.py        what the stage does, and its public names
    case.py            Case
    network.py         Network
    nygrid.py          read_npcc, modify_mpc, nygrid_units
    fixes.py           split_zone_j, restore_765kv, externals_at_base, hvdc_links
    ratings.py         line_ratings
    interfaces.py      Interface, hourly_interfaces, NYISO_INTERFACES (N48)
    clients.py         Ownership, assign_clients
    ny_bus_locations.csv  each NY bus located on the territories (N12)
  scenario/
    __init__.py        what the stage does, and its public names
    inputs.py          ZoneLoad, NygridHour, read_nygrid, fetch_nyiso
    loads.py           LoadSeries, LoadModel (the bus profiles, with their within-hour terms)
    load_sources.py    the building-stock, EIA-861, census and IAC inputs the profiles come from, downloaded once
    load_fit.py        fluctuation_fit, switching_fit (how the within-hour constants were fitted, N31)
    load_metrics.py    the measures LoadModel.validate compares with the real load
    dispatch.py        Dispatch, hourly_dispatch
    solver.py          AcSolver
    pool.py            Pool, PoolBuilder, peak_ratings
  sensing/
    __init__.py        what the stage does, and its public names
    meters.py          MeterPlan, plan_meters
    measure.py         Measurement
    noise.py           NoiseModel
  attacks/
    __init__.py        what the stage does, and its public names
    episode.py         Episode, AttackFamily (design, design_explained), MISSES
    attacker.py        Attacker, natural_steps, generator_ramps, generator_band, generator_q_capability, wrap_angles
    msfdia.py          MsFdia (Am, and Ap with a step bound)
    opf_attack.py      OpfFdia (Am, Ap with the false state from a local AC OPF, N35)
    interface_attack.py InterfaceFdia (Am against a NYISO interface, N48, N49; overshoot and max_overshoot, N58)
    ramp.py            Ramp
  timeline/
    __init__.py        what the stage does, and its public names
    schedule.py        Slot, place_episodes, split_weeks, split_frames, week_index
    builder.py         TimelineBuilder
  detect/
    __init__.py        what the stage does, and its public names
    estimate.py        WlsFit, wls, exact_state, reference_bus (the residual test's AC weighted least squares)
    rules.py           AlarmRule (protocol), PersistenceRule (k of the last n past a fitted threshold, N79)
    fleets.py          FLEET_STREAM, fleet_bias, with_fleet (other realizations of the meters' constant biases, N81)
  operator/
    __init__.py        what the stage does, and its public names
    sced.py            DcSced, Step, shift_factors, outage_factors, shortage_usd (N59, N60)
    day.py             OperatorDay, operator_day, INTERFACES (an attacked day as the operator meets it)
    ties.py            tie_meters, neighbour_reads, tie_gap, interchange_alarm, seen_interfaces (the interchange check, N62)
    truegrid.py        TrueGrid (state, run_state, change: AC power flows on the true grid)
    policy.py          OperatorPolicy (the operator's choices, validated), OperatorMode, AlarmAnswer, Entry, MODES, AFTER, ENTRY, entry_share
    run.py             Twin, Response, respond, Operator (protocol), NyisoOperator, REG_CAP (the operator scan by scan)
    summary.py         summarize (the attack's consequence, attacked less benign)
    response.py        every name above, re-exported for the imports written before the split
tools/nyfed/
  operator_cost.py     the operator on single-episode days: results/nygrid.operator_cost.csv and the per-scan trace
  tie_check.py         the interchange check's false alarms over 2019, calibrated on January to March (N67)
  damage_study.py      both forms of the overload attack (at the limit, at its largest overshoot) on near-limit onsets drawn at random, each through the operator: damage.csv (N58 to N60)
  build_timeline.py    the release timeline from a pool (--fetch downloads the inputs)
  build_single_episodes.py  one Am per file over its whole day, for studying one attack by hand
  matlab/nygrid/       MATLAB reference for NYgrid's AC feasibility (README inside)
  nygrid_ac_summary.py summaries of those runs
  run_pool.py          builds the pool in parallel chunks (a week each, from its first hour's AC point), then repairs each seam by solving the next chunk's first frames again from the chunk before's last state until the thermal dispatch meets the stored one within 0.5 MW, so the year is one continuous dispatch; per-unit ramps from attacks/attacker.py's constants, NYgrid's reactive limits (N38)
results/
  nygrid.ac_year.csv, nygrid.ac_ablation.csv   per-hour MATLAB results, with .json summaries
tests/nyfed/
  test_nyfed_{grid,scenario,sensing,attacks,timeline,dataset,operator}.py
  test_nyfed_{solvers,choices,misses,tables,configs,detect,policy,public,experiments}.py   checks that need no data (they run in CI)
  test_nyfed_golden.py, golden/   snapshots of the operator, the detection tools and the interface attack
  data/matlab_ref/     operating points from dump_hours.m that the Python port must match
```

`tools/`, `results/` and `tests/data/` follow the fdia_graph_sdk layout, so the move into the SDK
copies them as they are.

## How the pieces talk

```
                     Settings
                        |
          Case ---> Network          Ownership
            |          |                 |
            |     +----+-----------+     |
            v     v                v     v
          AcSolver           Measurement (MeterPlan, NoiseModel)
            |                      |
       PoolBuilder ---> Pool    Attacker
                         |         |  uses MsFdia, Ramp
                         |         v
                         +--> TimelineBuilder ---> TimelineWriter ---> timeline.h5
                                                                          |
                                                              Dataset ---> ClientView
```

- `grid` imports only `config`.
- `scenario` and `sensing` import `grid`.
- `attacks` imports `grid` and `sensing`. Its local re-solves (the subnetwork Newton of [1], load and
  flow goals, one least-squares solve since N18 holds the edge's non-free injections) live in
  `attacks/attacker.py` as the module function `_local_flow_solve`, pure numpy so `attacks` never imports pandapower; they are not `Network` methods
  (A1 decision, see PLAN.md). A family's `design` takes `window`, the tuple `(onset, vm [L, N],
  va [L, N])` of the frames it covers. The attacker's subnetwork may span utilities, and stealth is
  global: the attack vector is written on every meter it moves beyond noise, whoever owns it, so the
  operator's one residual test over all meters reads one AC state; `Episode.clients` records the
  owners of those meters, one or several.
- `timeline` imports everything above it and `io`.
- `dataset` imports only `io` and `grid/network.py`, so loading data never needs pandapower.

## Records

Frozen dataclasses of arrays, passed between classes.

| record | file | fields |
|---|---|---|
| `Case` | `grid/case.py` | `base_mva`, buses (`bus_id`, `zone`, `area`, `kv`, `vmin`, `vmax`, `lat`, `lon`, `is_ny`), branches (`f`, `t`, `r`, `x`, `b`, `tap`, `shift`, `rating`), units (`bus`, `fuel`, `pmin`, `pmax`, `qmin`, `qmax`, `ramp` MW/min, NaN where unknown), HVDC links (`f`, `t`, `pmin`, `pmax`), base state |
| `Ownership` | `grid/clients.py` | `names [K]` (the 7 utilities), `bus_client [N]` (-1 for external buses), `tie_lines [L]` |
| `LoadSeries` | `scenario/loads.py` | `time [T]`, `p [T, N]`, `q [T, N]` |
| `NygridHour` | `scenario/inputs.py` | one hour of NYgrid inputs: `time`, `fuel_mw [7]`, `iface_flow`, `iface_pos`, `iface_neg [17]`, `nuclear_mw [6]`, `stl_cf`, `thermal_mw [227]`, NYgrid's bus split `ny_pd`, `ny_qd [N]`, netted renewables `renew_mw [N]`, the thermal units' linear cost `thermal_cost [227]` ($/MWh, weekly fuel price times heat rate, allocateGen.m 'lm'), the proxy-bus LBMPs `ext_lbmp [4]` (NPX, O H, PJM, H Q) |
| `Dispatch` | `scenario/dispatch.py` | `hour [H]`, `unit_p [H, G]`, `unit_on [H, G]`, `hvdc_p [H, D]`, the bus loads they balance `pd`, `qd [H, N]`, and NYgrid's generation cost `unit_cost [H, G]` ($/MWh: thermal from `thermal_cost`, nuclear and hydro at NYgrid's fixed draws, external units at their area's proxy LBMP) |
| `Pool` | `scenario/pool.py` | `time [T]`, `vm [T, N]`, `va [T, N]`, `p [T, N]`, `q [T, N]`, `solve_kind [T]` (`SOLVED`, `FAILED`, `NO_HOUR`, `NO_LOAD`; unsolved frames keep their place with NaN state, branch flows are `Network.flows(vm, va)`), `unit_p [T, G]` each unit's real-time dispatch basepoint (N32; None for a pool written before N32) |
| `MeterPlan` | `sensing/meters.py` | one row per meter: `kind` (v, va, p, q, pf, qf, i, ia), `bus`, `branch`, `end`, `owner`, `sigma_rel`, `sigma_abs`, `bias` |
| `Episode` | `attacks/episode.py` | `family`, `clients` (owners of the tampered meters), `onset`, `length`, `buses`, `target_branch`, `a [length, M]` |
| `Slot` | `timeline/schedule.py` | one placed episode: `onset` (frame), `family` ("Am", "Ap" or "At"), `length` |

## Classes

| class | file | holds | methods |
|---|---|---|---|
| `Network` | `grid/network.py` | a `Case`, its `Ybus`, `Yf`, `Yt` built once, the bus adjacency | `injections(vm, va)`, `flows(vm, va)`, `currents(vm, va)`, `neighbors(buses, hops)` |
| `LoadModel` | `scenario/loads.py` | built as `LoadModel(case, own, nygrid_data, locations, cache_dir)`: each NY load bus's utility sector mix, its five nearest county prototype pools at 15 minutes (`building_stock_ny_15min.npz`), the IAC plant curves | `profiles(zone_load, rng) -> LoadSeries` (15-minute shares, then each bus's AR(1) fluctuation and industrial switching pulses, rescaled to the zone, N31), `validate(held_out, series)` (the four Sec. IV metrics of [2] per bus against the held-out year's zone band, plus the five-minute checks `step_distribution`, `step_zone_corr`, `step_zone_share`) |
| `AcSolver` | `scenario/solver.py` | the `Network`, the Pyomo AC-OPF model (two objectives: deviation, and generation cost plus a quadratic penalty holding anchored units to a schedule), one pandapower net built from the `Case`, the last solved state for warm starts | `project(pd, qd, unit_p, unit_on, hvdc_p)` (hourly min-deviation AC-OPF as a Pyomo model solved by Ipopt), `dispatch(pd, qd, cost, p_lo, p_hi, q_lo, q_hi, unit_on, hvdc_p, init=None, anchor=None, anchor_weight=1000)` (one five-minute least-cost AC-OPF within per-unit P and Q bounds, started from `init` = the previous (unit_p, unit_q), units with a finite `anchor` (MW) held near it at `anchor_weight` $/MW^2h, N32, N39), `power_flow(pd, qd, unit_p, unit_on, unit_vm, hvdc_p, start=None)` (pandapower, at a frame's basepoints; `start` = (vm, va) starts Newton at the dispatch's own solution) |
| `PoolBuilder` | `scenario/pool.py` | built as `PoolBuilder(solver, loads, dispatch, renew, ramp, q_cap=None)`: an `AcSolver`, the 5-minute `LoadSeries`, the `Dispatch`, NYgrid's hourly netted renewables `renew [H, N]`, each unit's ramp `ramp [G]` (MW per frame, inf unbounded), its reactive cap `q_cap [G]` (None: the case's limits, as the grid uses, N38); after a build `final` = (unit_p, unit_q, vm, va) of its last solved frame and per frame `slack_mw`, `cost`, `ramp_bound`, `anchor_dev` | `build(start=None) -> Pool` (projects every hour the frames touch, then a least-cost AC-OPF per frame in order from `start` or from the first hour's AC point: thermal units within their limits and their ramp of the previous frame's basepoint, nuclear at the hourly AC points' output interpolated, hydro and external units anchored to it (N39), every unit in service for reactive power as in NYgrid's case; then a power flow at the basepoints), `report()` (solve counts, voltages, reference bus residual, cost, anchor deviation, ramp-held share, reactive fallbacks) |
| `Measurement` | `sensing/measure.py` | a `Network`, a `MeterPlan` | `h(vm, va) -> z`, `jacobian(vm, va)`, `for_client(c)` |
| `NoiseModel` | `sensing/noise.py` | the accuracy classes, the per-meter biases | `sigmas(z)`, `sample(z, rng)` |
| `AttackFamily` | `attacks/episode.py` | interface only | `design(window, rng) -> Episode` |
| `Attacker` | `attacks/attacker.py` | a `Network`, a `Measurement` | `area(center, hops)` (the local subnetwork of [1] in PLAN.md, which may span utilities, or `None`), `false_state(vm, va, area, load_change)` (At's load goal: the pretended loads prescribed, the other free injections of the area and its edge absorbing within the same bounds and active set, every other bus held; module `_local_flow_solve` with no target line), `false_flow_state(vm, va, support, lines, target_mva, load_cap, step=None, start=None)` (Am's flow goal, module `_local_flow_solve`: the support first grows over its non-free New York edge buses (`absorb`, N25); the free injections of the support and its edge move, every other New York bus there, zero-injection ones included, holds its injection, a bus outside New York on the edge absorbs unless it has no unit and no injection (`_held_bus`, N44, bus 140), which holds; loads capped at 0.5 of their true P and Q (D16, N24); an active set pins a free injection at a bound and releases it when the solve without the pin leaves it inside; Ap's `step = (centre, bound_p, bound_q)` adds a box around a centre; `start = (vm, va)` warm-starts the state and the active set from the previous scan's false state), `attack_vector(vm_t, va_t, vm_f, va_f)` (global, in full), `tampered(vm, va, a)` (the meters moved beyond noise, angle changes wrapped: eq. 12 count, labels, owners), `load_buses(vm, va, buses=None)`, `free_injection_buses(vm, va, buses)`, `boundary(interior)`, `sigma(z)`, `bus_load(vm, va)` |
| `MsFdia` | `attacks/msfdia.py` | an `Attacker`, the ratings; for Ap (ours) `step_bound [24, N]` (`natural_steps` at load buses, `generator_ramps` at generator buses, on the reported output there; a column of inf leaves that bus unbounded per step), `frame_hour [T]` the clock hour of the frames onsets index, `kappa` (1.0), `reactive_step` (False: Ap bounds only the active step, N22; True: the P+Q variant), `family` ("Am" by default, "Ap"), `target_order` ("loading" by default, the most loaded eligible line at the window's last scan first, N26; "random"), `bound_rise`, `exit_scans` (N30), `wait_share` (N36, None or `NEAR_LIMIT` 0.9) | `design(window, rng)`; with `step_bound=None` it is Am. Both solve a support's scans in order, each warm-started from the previous scan's false state. `loaded_frames(vm, va, share) -> [T]` whether some candidate line carries `share` of its rating; `onset_mask(vm, va, length) -> [T] or None` the onsets whose last rise scan is loaded (N36) |
| `OpfFdia` | `attacks/opf_attack.py` | `MsFdia`'s settings plus `gen_ramp [N]` (`generator_ramps`), `gen_band [N]` (`generator_band`, 3% of the upper limit, N33/N37), `q_lo`, `q_hi [N]` (`generator_q_capability`, widened to the unit's true Q, N34/N38), `q_ramp [N]` (unbounded by default, N37), `windows_solved` (3), `gen_output [T, N]` (the units' output at each bus, from the pool's `unit_p`; the generator bounds act on it and the load at the bus moves within the load cap, N42), `loads_only` (False; True pretends no generator bus, N43) | `design(window, rng)`: MsFdia's target choice and support search, each candidate support tested by a one-scan AC OPF at the full overload with every static bound (`_static`), then solved scan by scan by Ipopt with the ramp, step and band bounds, each scan kept within reach of the full-overload solution and of zero at the last scan; the support whose window tampers the fewest meters is kept and an attack still tampering at the last scan does not exist |
| `InterfaceFdia` | `attacks/interface_attack.py` | `OpfFdia`'s settings plus `interfaces` (a list of `Interface`, built over the same frames as the pool) and `near` (0.9); `windows_solved` 6 | `onset_mask(vm, va, length)` (last rise scan in a real 2019 hour at `near` of some interface's limit, N48), `design(window, rng)`: the goal is the model's interface flow plus that hour's real headroom (N49); supports grow from one interface branch's ends at a time; each window is solved with the meter objective (the change beyond noise on every P and Q flow meter touching the interior and every interior voltage meter, added to the injection terms) and with `OpfFdia`'s objective where that fails; the fewest tampered meters over all anchors is kept |
| `Ramp` | `attacks/ramp.py` | an `Attacker`, the ramp settings | `design(window, rng)` |
| `TimelineBuilder` | `timeline/builder.py` | built as `TimelineBuilder(pool, meas, noise, attacker, families, own, settings, *, length=16 or {family: frames} (Am, At 16, Ap 96, N23), attacked_frac=0.2, split_frac=(0.7, 0.15, 0.15), split_by="week" (N41) or "time", workers=1, block=2016, verbose=False, survey=0, survey_cache=None)`: the pool, the `Measurement`, the `NoiseModel`, the `Attacker`, the families by name (`{"Am": MsFdia, "Ap": MsFdia(step_bound=...), "At": Ramp}`), the `Ownership` and `Settings` (whose seed keys every draw: `place`, `noise`, and `design/{family}/{onset}` per episode, so the result does not depend on `workers`) | `build(path) -> dict` (writes the file; returns slots placed and episodes built per split and family, slots skipped for unsolved frames, the median design seconds per family, seconds). The splits come from `split_weeks` on the frames some waiting family may start at (or `split_frames` with `split_by="time"`); each split's episodes are placed by `place_episodes` inside one contiguous run of its frames and, for a family with an `onset_mask`, only where it allows; each is designed on its own window of true pool states (in `workers` spawn processes when above 1); a window touching an unsolved frame, or one the family returns None for, stays benign. With `survey` (frames, N42) every allowed onset of a family with an `onset_mask` whose window lies in one week, on that grid, is designed first, the weeks are then split stratified on the onsets with an attack in strata of `HIT_STRATUM` weeks (N46), and the family's episodes are placed only among those with an attack, reusing the designs, each chunk of `SURVEY_CHUNK` designs saved under `survey_cache` when given so a stopped build resumes; the summary adds `survey` (tried and found per split and family). `build_episode(path, onset, family="Am", lo=None, hi=None) -> dict` writes one episode alone, designed at pool frame `onset`, over the pool frames lo..hi (default its window), every frame in the train split, the file's frames and onset counted from lo (single-episode files for the analytical study, Ben 2026-11-02); both write through `_write`. Frames: `z_benign = NoiseModel.sample(h(x_true))`, `z = z_benign + a` on episode frames; `meter_tampered` is `Attacker.tampered`, `bus_attacked` is true where a tampered meter sits at the bus |
| `DcSced` | `operator/sced.py` | built as `DcSced(case, unit_p [T, G], cost [T, G], interfaces)`: the pool basepoint, the units' costs, the interfaces it enforces | `step(t, seen [I], limit [I], load_change, prev) -> Step`: the scan's dispatch, least movement then least cost (N53), free units only (not nuclear, hydro or imports), within limits and five-minute ramps from `prev`, the unrelieved interface flow and unmatched load as slacks |
| `Step` | `operator/sced.py` | `dev [G]` (MW from the basepoint), `shed [N]` (MW of load relief), `unrelieved [I]`, `unmatched`, `relief_usd` (the dispatch's cost less that of the same dispatch with no interface limit) | |
| `OperatorDay` | `operator/response.py` | one timeline file's frames with the pool basepoint, the interpolated unit costs, the real zonal LBMP and the interfaces | |
| `Twin`, `Response` | `operator/response.py` | the operator's run on the attacked scans and on the benign twin: dispatch, regulation, unrelieved and unmatched MW, the true interface and N-1 changes, fuel and settled $ per scan | |
| `TimelineWriter` | `io.py` | an open HDF5 file, built as `TimelineWriter(path)` | `write_static(case, own, plan)`, `write_frames(t0, block)` (`block` a dict of the per-frame arrays of frames `t0..`, datasets grown as written), `write_episodes(episodes)`, `close()` |
| `Dataset` | `dataset.py` | an open timeline file (`open_timeline`) with its `case`, `own`, `plan` | `clients`, `client(name, split=None) -> ClientView`, `operator(split=None) -> ClientView` (every meter and bus, the operator's full scan); `split` is "train", "val", "test" or None for all frames |
| `ClientView` | `dataset.py` | one client's meter, bus and frame selection, read when made, no file handle: `buses [N_c]` (bus positions it owns), `frames` (the indices of the selected frames, exact for the interleaved week split) | `z [T, M_c]`, `labels [T, N_c]` (`bus_attacked` of its buses), `meters` (its `MeterPlan` rows), `node_features()` (`[T, N_c, 4]` metered \|V\|, angle, P, Q per owned bus, NaN where not metered), `temporal()` (`[T, M_c]` change from the previous selected frame, zero at the first and after a gap in the selection, angle readings wrapped to (-180, 180]; `z` stays raw) |

## Functions

Stateless steps, each called once in the build.

| function | file | in | out |
|---|---|---|---|
| `read_npcc(path)` | `grid/nygrid.py` | `npcc.mat` | `Case` |
| `modify_mpc(case)` | `grid/nygrid.py` | `Case` | `Case` with every `modifyMPC.m` change |
| `nygrid_units(case, data_dir)` | `grid/nygrid.py` | `Case`, NYgrid `Data/` | `Case` with the fixed unit list of NYgrid's hourly case (thermal, nuclear, hydro, external, HQ) |
| `split_zone_j`, `restore_765kv`, `hvdc_links` | `grid/fixes.py` | `Case` | `Case` |
| `externals_at_base(case, base)` | `grid/fixes.py` | an hourly `Case`, the base `Case` | `Case` with external load at base and external units covering the balance |
| `line_ratings(case)` | `grid/ratings.py` | `Case` | `Case` with the ratings NYgrid gives (modifyMPC.m ties, fixed single-branch interface limits), 0 elsewhere; the rest are filled by `peak_ratings` |
| `hourly_interfaces(case, time, data, names=None, year=2019)` | `grid/interfaces.py` | `Case`, the pool's frame times, NYgrid's data folder | a list of `Interface` (branches and signs from `updateOpCond.m` L420-430, real hourly limit and flow per frame from `interflowHourly_2019.csv`); `Interface.model_flow(pf)` the signed sum |
| `peak_ratings(case, pool, factor=1.0)` | `scenario/pool.py` | `Case`, `Pool` | `Case` with each unrated branch at `factor` times its peak apparent flow over the solved frames (N8; 1.0 by N47) |
| `write_pool(path, pool)`, `read_pool(path)` | `io.py` | `Pool` | `pool.h5`: `time` (int64 seconds, NYISO local clock), `vm`, `va`, `p`, `q`, `solve_kind`, `unit_p` (when present) |
| `plan_meters(net, own, rng, vbus_frac=0.6)` | `sensing/meters.py` | `Network`, `Ownership`, a generator | `MeterPlan`: P, Q at every NY bus, P, Q flow at every branch end on a NY bus, \|V\| at a random 60% of NY buses, PMUs (voltage and branch-current phasors) at the smallest set that sees every NY bus (N7), each meter owned by its bus's client, accuracy class and drawn bias per meter |
| `assign_clients(case, territories, locations)` | `grid/clients.py` | `Case`, territory polygons, path to `data/territories/ny_bus_locations.csv` (substation location or zone-chosen utility per NY bus) | `Ownership`, one client per utility |
| `read_nygrid(data_dir, year)` | `scenario/inputs.py` | NYgrid `Data/` | `NygridHour` list |
| `fetch_nyiso(year)` | `scenario/inputs.py` | NYISO pal archive | `ZoneLoad` |
| `hourly_dispatch(case, hours, hold_externals=True)` | `scenario/dispatch.py` | | `Dispatch`; `hold_externals=False` keeps NYgrid's own external scaling (the AC ablation) |
| `split_weeks(time, frac, weight, rng, stratum=STRATUM)` | `timeline/schedule.py` | frame times, the shares, a weight per frame (the loaded onsets) | split column `[T]`: whole weeks ranked by weight, cut into strata of 20, each stratum drawn into train, val, test in the shares (N41) |
| `split_frames(T, frac)` | `timeline/schedule.py` | | split column `[T]`: train, val, test as three contiguous blocks of the year in time order (before N41) |
| `natural_steps(time, p, solved, pct=95.0)` | `attacks/attacker.py` | the pool's frame times, net injections `p [T, N]` (MW), which frames are solved | `[24, N]` each bus's `pct` percentile of \|p(t) - p(t-1)\| over consecutive solved frames, by the clock hour of the later frame: the natural five-minute step Ap's attacker knows (ours) |
| `place_episodes(T, lengths, frac, rng, allowed=None)` | `timeline/schedule.py` | one split's frame count, `{family: length}`, the attacked share, `{family: [T] allowed onsets}` | `Slot` list: the whole episodes closest to `frac T` frames, families equal in attacked frames (a family capped at what its allowed onsets hold), each at a uniform random onset among the allowed ones where it fits without overlap (adjacency and gaps come only from this) |
| `ipopt(**options)` | `solvers.py` | Ipopt's options | a Pyomo `SolverFactory` checked to start; `SolverUnavailable` with the fix when Pyomo or Ipopt is missing or Ipopt cannot start |
| `read_rows(path, encoding)` | `tables.py` | a CSV table of the inputs | its rows as dicts, the file closed |
| `open_timeline(path)` | `io.py` | `timeline.h5` | the read-only file with its `Case`, `Ownership`, `MeterPlan` (a file written before `Case.ramp` reads it as NaN) |
| `generator_ramps(case, minutes=5.0)` | `attacks/attacker.py` | `Case` | `[N]` each bus's units' ramp over `minutes` (MW) summed at the bus: NYgrid's ramp, nuclear 0, hydro 10% of nameplate per minute, imports 300 MW per 15 minutes per unit (N34), inf at a bus with no unit |
| `generator_band(case)` | `attacks/attacker.py` | `Case` | `[N]` 3% of each bus's units' upper limit (MW), the basepoint band (N33) |
| `generator_q_capability(case)` | `attacks/attacker.py` | `Case` | `([N], [N])` reactive capability plus or minus 0.312 Pmax per bus (MVAr, N34) |
| `fetch_inputs(root)` | `build.py` | a folder | `root/data`: NYgrid's `Data/` at commit 47698b6, NYISO's pal archives for the year, the NYS DPS territories and the packaged `grid/ny_bus_locations.csv`, each kept when already there (the load model fetches its own sources on the first pool build) |
| `production_builder(pool_path, families=("Am", "At"), *, interfaces=("SPR/DUN-SOUTH",), workers=12, settings=Settings(), root=".", survey=12, survey_cache=None, hold_external=False, v_margin=0.0, gen_v_tol=None)` | `build.py` | the pool and `fetch_inputs`'s folder | the release `TimelineBuilder`: peak ratings (N8), Am from the OPF attacker against the interfaces (N48, N49) or the ratings, Ap bounded by natural steps, At, every realism bound (N27 to N37, N42), 32-scan Am and 16-scan At windows, 20% attacked. The attacker's sensitivities are arguments: `hold_external` (N62), `v_margin` widens the voltage limits on the attacker's own copy of the case (N71), `gen_v_tol` holds generator voltages near their schedule (N75) |
| `wrap_angles(d, kind)` | `attacks/attacker.py` | reading differences `[..., M]`, meter kinds | the differences with angle readings (va, ia) wrapped to (-180, 180] |
| `InterfaceFdia.max_overshoot(window, rng, high=800.0, tol=25.0)` | `attacks/interface_attack.py` | a window | `(Episode or None, MW)`: the damaging variant at the largest overshoot past the limit the bounds allow, by bisection (N58), on a copy of the family |
| `InterfaceFdia.with_overshoot(mw)` | `attacks/interface_attack.py` | MW past the limit | a copy of the family with that goal; the original unchanged |
| `Case.with_voltage_margin(margin)` | `grid/case.py` | p.u. | a copy with every bus's voltage limits widened (N71); the original unchanged |
| `TimelineBuilder.build_episode(path, onset, family, lo, hi, using=None)` | `timeline/builder.py` | one onset | one episode alone over frames lo..hi; `using` designs it with a family variant (such as `with_overshoot`) in place of the builder's own |
| `shortage_usd(mw)` | `operator/sced.py` | MW over an interface limit for one scan | $ at the 2019 transmission shortage curve (N60) |
| `shift_factors(case)`, `outage_factors(case, ptdf, outages)` | `operator/sced.py` | `Case` | `[L, N]` DC shift factors (reference bus out), `[L, K]` line outage distribution factors |
| `operator_day(timeline, pool, data_dir, interfaces=("SPR/DUN-SOUTH",))` | `operator/response.py` | a timeline file, its pool, NYgrid's `Data/` | `OperatorDay` |
| `tie_meters(case, plan)`, `neighbour_reads(plan, rows, far, rng)` | `operator/response.py` | the case and plan, the far-end flows | the ties with a New York-end P-flow meter, and the neighbours' readings of them through meters of the same class (N67) |
| `tie_gap(day, z)` | `operator/response.py` | `OperatorDay`, scans `[T, M]` | `[T]` the net interchange read at the New York ends of the ties less what the neighbours meter at their ends (the true flow there), the interchange check of N62 |
| `seen_interfaces(plan, interfaces, z)` | `operator/response.py` | the meter plan, the interfaces, scans `[T, M]` | `[T, I]` each interface's flow as the operator reads it from its branches' P-flow meters |
| `respond(day, delay=0, over=0.0, mode="trusting", calib=None)`, `summarize(response, voll=None)` | `operator/response.py` | `OperatorDay` | `Response` (O1, O2, O5, N50 to N65: every interface of `INTERFACES` enforced, quick-start units, regulation capped at `REG_CAP` with the rest an ACE excursion, the interchange alarm, and with `defended` the operator that stops acting on the tampered scans from the alarm), and its day totals, attacked less benign, with regulation and shed by utility |

## The timeline file

| group | dataset | shape |
|---|---|---|
| `case/` | the `Case` arrays (strings UTF-8 bytes), `bus_client`, `client_names`, `tie_lines` | static |
| `meters/` | the `MeterPlan` table, attribute `kind_codes` | `[M]` |
| `frames/` | `time` (int64 s), `split` (int8, attribute `split_codes`), `family` (int8, attribute `family_codes` benign, Am, At, Ap), `episode` (int32 row, -1 benign) | `[T]` |
| `frames/` | `z` (observed), `z_benign`, float32 | `[T, M]` |
| `frames/` | `vm`, `va` (true state) | `[T, N]` |
| `labels/` | `bus_attacked` | `[T, N]` |
| `labels/` | `meter_tampered` | `[T, M]` |
| `episodes/` | one row per `Episode`: `family`, `onset`, `length`, `target_branch`; ragged `buses`, `clients` as flat arrays with `[S + 1]` offsets `buses_ptr`, `clients_ptr` | `[S]` |

`z - z_benign` is the attack. Temporal features are computed at load time from each client's own `z`.
Unsolved pool frames keep their place with NaN state and readings and no labels.

## Blocks to files

| block | files |
|---|---|
| F1, F2 | `pyproject.toml`, `config.py`, `io.py` |
| G1 | `grid/case.py`, `grid/nygrid.py`, `grid/network.py` |
| G2 | `grid/fixes.py` |
| G3 | `grid/nygrid.py` (unit list) |
| G4 | `grid/ratings.py` |
| G5 | `grid/clients.py`, `data/territories/ny_bus_locations.csv` |
| P1 | `scenario/inputs.py` |
| P2 | `scenario/loads.py` |
| P3 | `scenario/dispatch.py` |
| P4, P5 | `scenario/solver.py`, `scenario/pool.py` |
| M1 | `sensing/meters.py` |
| M2 | `sensing/measure.py`, `sensing/noise.py` |
| A1 | `attacks/episode.py`, `attacks/attacker.py` |
| A2 | `attacks/msfdia.py`, `attacks/opf_attack.py`, `attacks/interface_attack.py` and `grid/interfaces.py` (N48, N49) |
| A3 | `attacks/ramp.py` |
| A4 | `attacks/msfdia.py` (Ap option), `natural_steps`, `generator_ramps`, `wrap_angles` in `attacks/attacker.py`, `Case.ramp` from `nygrid_units` |
| D1 | `timeline/schedule.py`, `timeline/builder.py`, `TimelineWriter` in `io.py` |
| D2 | `dataset.py`, `open_timeline` in `io.py` |
| D3 | `build.py` (`fetch_inputs`, `production_builder`), `tools/nyfed/build_timeline.py`, `tools/nyfed/build_single_episodes.py` |
| O1, O2, O5 | `operator/sced.py`, `operator/response.py`, `tools/nyfed/operator_cost.py` |
