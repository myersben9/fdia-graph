# NY Federated Grid SDK: Build Outline

In fdia-graph since 2026-11-02 as `fdia_graph.nyfed`: the package paths below are under `src/fdia_graph/nyfed/`, the tests are `tests/nyfed/test_nyfed_<package>.py` (inputs under `NYFED_ROOT`), the tools `tools/nyfed/`, and the results files sit in `results/` as they were.

A small Python package that generates and loads a New York FDIA dataset. Each utility is a federated client. The grid is the modified NPCC-140 of NYgrid [3], run in AC.

Sources: fdia-graph at [`9565688`](https://github.com/myersben9/fdia-graph/tree/9565688), NYgrid at [`47698b6`](https://github.com/AndersonEnergyLab-Cornell/NYgrid/tree/47698b6).

## Package layout

```
nyfed/
  grid/       case, AC fixes, units, ratings, clients
  scenario/   inputs, load profiles, dispatch, AC pool
  sensing/    meter plan, noise
  attacks/    attacker area, MS-FDIA, ramp
  timeline/   episode placement, write
  dataset.py  loader, client views
tests/
```

Files, types, functions and how they connect are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Blocks

### F. Foundation

| block | what | source |
|---|---|---|
| F1 | package, tests, CI | [`pyproject.toml`](https://github.com/myersben9/fdia-graph/blob/9565688/pyproject.toml), [`smoke.yml`](https://github.com/myersben9/fdia-graph/blob/9565688/.github/workflows/smoke.yml) |
| F2 | file layout and download | [`schema.py`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/schema.py), [`download.py#L85`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/download.py#L85) |

### G. Grid

| block | what | source |
|---|---|---|
| G1 | read `npcc.mat` and apply `modifyMPC.m` | [`Data/npcc.mat`](https://github.com/AndersonEnergyLab-Cornell/NYgrid/blob/47698b6/Data/npcc.mat), [`modifyMPC.m`](https://github.com/AndersonEnergyLab-Cornell/NYgrid/blob/47698b6/modifyMPC.m) |
| G2 | AC fixes: split the zone J generators, restore the 765 kV line, keep external areas at base load, model HVDC links as DC lines | [`modifyMPC.m#L75`](https://github.com/AndersonEnergyLab-Cornell/NYgrid/blob/47698b6/modifyMPC.m#L75), [`updateOpCond.m#L313`](https://github.com/AndersonEnergyLab-Cornell/NYgrid/blob/47698b6/updateOpCond.m#L313), [`updateOpCond.m#L405`](https://github.com/AndersonEnergyLab-Cornell/NYgrid/blob/47698b6/updateOpCond.m#L405) |
| G3 | `Case` object with a fixed unit list and admittances | [`network.py#L59`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/formulas/network.py#L59), [`updateOpCond.m#L179`](https://github.com/AndersonEnergyLab-Cornell/NYgrid/blob/47698b6/updateOpCond.m#L179) |
| G4 | line ratings: NYgrid's 10 tie ratings and the 2 single-branch interface limits with a fixed value (F-G, H-I); multi-branch and hourly interface limits give no per-branch number, so those branches stay unrated | [`modifyMPC.m#L52`](https://github.com/AndersonEnergyLab-Cornell/NYgrid/blob/47698b6/modifyMPC.m#L52), [`updateOpCond.m#L418`](https://github.com/AndersonEnergyLab-Cornell/NYgrid/blob/47698b6/updateOpCond.m#L418) |
| G5 | client map: each NY substation located by name and placed on utility territories, so every bus, branch end and meter has one owner | NYS DPS utility service territories (data.ny.gov awza-4vgu), OpenStreetMap `power=substation` (Overpass, 2026-10-26), EIA-860 plant coordinates (August 2021 extract), [`bus_ny_type_zone_new.csv`](https://github.com/AndersonEnergyLab-Cornell/NYgrid/blob/47698b6/Data/bus_ny_type_zone_new.csv); located buses in `data/territories/ny_bus_locations.csv` |

### P. Operating points

| block | what | source |
|---|---|---|
| P1 | 2019 historical inputs and NYISO 5-minute zonal load | [`readOpCond.m`](https://github.com/AndersonEnergyLab-Cornell/NYgrid/blob/47698b6/Utility/readOpCond.m), [`profiles.py#L290`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/profiles.py#L290), [NYISO pal](https://mis.nyiso.com/public/P-58Blist.htm) |
| P2 | a unique load profile per bus, summing to the real zone load | method of [2]; NREL ResStock and ComStock AMY2018 release 1 by county ([OEDI](https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/)), EIA-861 2019 `Sales_Ult_Cust_2019.xlsx`, IAC database ([ITAC](https://itac.university/storage/ITAC_Database.zip)), ORNL/TM-2013/407 App. B curves, 2019 Census county gazetteer |
| P3 | hourly unit dispatch | [`updateOpCond.m#L60`](https://github.com/AndersonEnergyLab-Cornell/NYgrid/blob/47698b6/updateOpCond.m#L60), [`allocateGen.m`](https://github.com/AndersonEnergyLab-Cornell/NYgrid/blob/47698b6/Utility/allocateGen.m) |
| P4 | hourly min-deviation AC-OPF, checked against the MATLAB results for 2019 | [`tools/nyfed/matlab/nygrid`](../../tools/nyfed/matlab/nygrid/README.md), `results/nygrid.ac_year.csv` |
| P5 | 5-minute operating points from a ramp-constrained real-time AC dispatch (N32), written as the pool | [`profiles.py#L162`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/profiles.py#L162) |

Status N31 (`scenario/loads.py`, `out/loads_2019_v2.npz`): the ResStock/ComStock prototypes are cached at their native 15 minutes
(`data/loads/building_stock_ny_15min.npz`, all 956 county files, 107 MB; the 55 counties the load buses' pools use hold 861 of them,
so the cache was not restricted; its hourly means equal the old hourly cache to 6e-8). Li's loop keeps hourly peaks and swaps whole
hours (four rows); the IAC/ORNL plant curves are held flat within the hour; each bus's 15-minute share is interpolated to the frames
from the middle of each quarter hour. Fluctuation, fitted on NYISO 2018 with the regulation component of Kirby and Hirst
(ORNL/TM-2001/13: load less the mean of L(t-3) .. L(t+2)): robust residual spread 0.2554 size^-0.516 (R^2 0.61, ten zones; zone H,
at 2.1% where the fit gives 0.58%, is left out; with H the fit is 0.566 size^-0.621), median residual lag-1 autocorrelation -0.020,
which through the filter is AR(1) phi 0.379 (white noise gives -0.233), so the AR(1)'s own spread is 0.3255 size^-0.516 (3.0% at
100 MW, 0.92% at 1 GW). Switching, fitted on zone D 2019: 2,799 steps above 5% of the 551 MW mean, max 314 MW, merged into 2,765
events (34 two-frame runs); 1,382 pulses a year over the model's 125.4 MW of zone D industrial load (its one NYSEG bus at NYSEG's
17.9% industrial mix) is 0.00126 pulses per MW-hour; size over that industrial load 0.2196 plus a generalized Pareto excess (shape
0.366, scale 0.0582) truncated at 2.50; duration lognormal, median 101 min (log sd 1.63; empirical 10/50/90% 5/135/685 min); 51.7%
of pulses start with a drop. Effect (2019 year, against the 2018 zones; medians over the 31 load buses, before -> after):
within-zone step correlation 0.984 -> 0.104 (range -0.565 .. 0.25), zone share of a bus's step variance 0.986 -> 0.29, step
distribution inside the zones' band 0.958 -> 0.917; Li's Sec. IV hourly metrics inside the band: load factor 0.917 -> 0.750,
distribution 0.964 -> 0.929, autocorrelation 0.917 -> 0.812, spectrum 0.784 -> 0.611. Ablation: 15-minute prototypes alone leave the
step correlation at 0.944; adding the AR(1) alone gives 0.122 (share 0.40) with the hourly metrics unchanged (0.917, 0.964, 0.875,
0.777); the switching pulses cost the hourly metrics, most on the NGRID bus of zone F (load factor in band 1.00 -> 0.42), because zone
D's events are divided by the model's 125 MW of industrial load there, much less than zone D's real industrial base. Calibrated on
zone D's whole 551 MW instead (rate 0.000287, size 0.05 plus GPD scale 0.0133, max 0.57), the hourly metrics stay at the AR(1)-only
values and the switching adds almost nothing at five minutes. Open for Ben: which industrial base to calibrate on, or no switching.

Status N32 (`scenario/solver.py` `AcSolver.dispatch`, `scenario/pool.py`, `tools/nyfed/run_pool.py`, `out/pool_2019_v2.h5` on
`out/loads_2019_v2.npz`): each frame is a least-cost AC OPF in the same Pyomo model as the projection (a second, linear cost
objective), solved by Ipopt from the previous frame's state; a pandapower power flow started at the OPF's own state then checks it.
Every unit stays in service for reactive power, as in NYgrid's case (GEN_STATUS 1): switching units with a zero hourly setpoint fully
off made 535 evening frames of March and August infeasible at the 0.9 p.u. floor, and a power flow started from the last one settled on
a low-voltage solution in 48 November frames (V 0.03); both are fixed. Year, 53 week chunks on 27 workers, 76 min (about 1.0 s per frame
per worker under load, 0.6 s alone, half of it the OPF under the 0.312 Pmax caps that fails): 104,940 of 105,120 frames solved
(99.83%; pool_2019 99.99%), 169 failed (late-August afternoons and one January frame), 11 without load. NY voltages 0.900 to 1.100.
Reference bus output beyond the dispatch: pool_2019 median 45 MW, 95th percentile up to 199 MW per week, max 866 MW; now median 2e-12 MW,
max 1.7e-5 MW. Reactive caps of N34: feasible in 28 frames, 99.97% fall back to NYgrid's limits, since the reduced case has no shunt
compensation and its reference bus (Indian Point, 74) alone carries 1.4 to 4.1 GVAr against 645 MVAr of capability. Ramp limits hold
a mean 4.2% of the free units per frame. Against NYgrid's hourly AC point at the same prices (January week), the dispatch costs
19.6% less (10th to 90th percentile 25.9 to 13.9% less): hydro sits at its 4,270 MW nameplate (NYgrid's hydro costs are 1.4 to 9.7
$/MWh; year mean 4,263 MW), the external units supply 1.9 GW more and the HQ import 390 MW less, gas 810 MW less. A chunk starts
from the hourly AC point, so a unit's step across the 52 chunk seams reaches 1,427 MW beyond its ramp. Generator buses: median
five-minute step 2.6 -> 4.5 MW, 99th percentile 32 -> 100 MW. Ben's load-bus measurement on the pool (11 NY buses without a unit, 6
same-zone pairs): step correlation 0.977 -> 0.164, zone share of step variance 0.991 -> 0.353. Open for Ben: the reactive caps, hydro
and external energy limits (an economic dispatch without them departs from NYgrid's historical mix), interface limits (N16), the seams.
That build is kept as `out/pool_2019_v2_econ.h5`; N38, N39 and N40 decided its open points.

Status N39 (implemented, `out/pool_2019_v2.h5`, built by `tools/nyfed/run_pool.py` in 37 min on 27 workers, 0.33 s per frame per worker):
only thermal units are re-dispatched by the five-minute least-cost OPF, within their ramps. Nuclear, hydro and every external unit follow
NYgrid's hourly output interpolated to the frame, taken at the hourly AC points (`project`, which moves them at most about 9 MW per
unit from NYgrid's raw setpoints, to cover losses). Nuclear holds it exactly. Hydro and external units hold it under a quadratic penalty
of 1000 $/MW^2h (`AcSolver.dispatch(anchor=...)`, the min-deviation rule of acproj.m), since held exactly the external areas cannot cover
their own losses and the loop flows of NY's re-dispatch, and the morning pickup is infeasible (bus 125 behind the 81-125 tie short by
about 25 MW). They move a median 0.5 MW in total from the AC points, 970 MW at the 99th percentile, 2,352 MW at most; against NYgrid's
raw schedule the external units move a median 129 MW in total (over 200 MW in 11% of frames, at most 2.3 GW, the NE and PJM units at
buses 21, 24, 36 and 130), hydro a median 39 MW. Weekly means against NYgrid's hourly output: hydro +0.7 to +2.3% (year 3,386 against
3,341 MW), external units +0.4 to +1.9% (21,184 against 21,012 MW); the per-week table is `out/pool_2019_v2_weekly_vs_nygrid.csv`.
NYgrid's unit reactive limits (N38). 104,909 of 105,120 frames solved (99.80%), 200 failed (single frames spread over the year,
and late-August afternoons), 11 without load. NY voltages 0.900 to 1.100. Reference bus output beyond the dispatch: median 4e-12 MW,
95th percentile 3e-9 MW, max 4.8e-5 MW (pool_2019: median 45 MW, max 866 MW). Cost against NYgrid's hourly AC point at the same prices:
17.7% less over the year (median hour 18.5% less, 10th to 90th percentile 28.2% to 11.4% less; the economic build was 19.6% less on
its January week), so most of the gap is the thermal re-dispatch itself; the hourly AC points also run units NYgrid has at zero (966 MW
of them in the first hour checked), which the dispatch keeps off. Ramp limits hold a mean 3.4% of the free thermal units per frame.
Seams: the 52 week seams are repaired so the year is one continuous dispatch. Chunk c's first frames are solved again from chunk c-1's
last state until every thermal unit is within 0.5 MW of the stored dispatch; 51 merged, after a median 33 frames (at most 171) and
within 0.47 MW; the week of August 20 did not merge in 24 h and was solved again whole from August 19's end, after which August 27's
seam merged in 171 frames. The largest thermal ramp excess left in the year is 0.43 MW, at two merge points (inside the 0.5 MW merge
tolerance), otherwise at most 0.01 MW. Generator buses: median five-minute step 4.5 MW, 99th percentile 63 MW (pool_2019 2.6 and 32,
the economic build 4.5 and 100). Load-only buses: same-zone step correlation 0.164, zone share of step variance 0.353.

### M. Measurements

| block | what | source |
|---|---|---|
| M1 | meter plan from ownership, PMUs with branch currents | [`core.py#L191`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/engine/core.py#L191), [1] eqs. 19 and 20 |
| M2 | accuracy-class noise, one scan per state | [`noise.py#L12`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/formulas/noise.py#L12), [`measurement.py#L75`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/engine/measurement.py#L75) |

### A. Attacks

| block | what | source |
|---|---|---|
| A1 | attacker area, the local subnetwork of [1] | [`area.py#L38`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/engine/attacks/area.py#L38), [`false_state.py#L86`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/engine/attacks/false_state.py#L86) |
| A2 | MS-FDIA (`Am`) | [`minimize.py#L131`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/engine/attacks/minimize.py#L131), [`overload.py#L110`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/engine/attacks/overload.py#L110), [1] eqs. 12 to 25 |
| A4 | pattern-aware overload (`Ap`, ours): Am plus a per-bus, per-scan step bound from each bus's natural five-minute step by hour | [1] eqs. 12 to 25, NYISO zone step statistics (paper Sec. III-C) |
| A3 | slow ramp (`At`) | [`stealthy.py#L26`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/engine/attacks/stealthy.py#L26), [`episodes.py#L43`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/engine/attacks/episodes.py#L43) |

Status: A1, A2, A3 built (`attacks/{episode,attacker,msfdia,ramp}.py`, `tests/nyfed/test_nyfed_attacks.py`,
5 tests). The attacker knows one local subnetwork that may span utilities and tampers whatever meters
its support needs; stealth is global (one WLS over all 511 meters leaves the attacked residual at the
benign level, verified in the test). `Episode.clients` records the owners of the tampered meters, one
or several. `Am` ports the fewest-tamper search of [1] eq. 12 (`msfdia.MsFdia`): candidate supports
grown from the target line's ends, each solved by the flow goal (`Attacker.false_flow_state`,
`_local_flow_solve`) that frees the support's attackable loads and generators within limits (22)-(23),
the edge included, under a 0.5 load-plausibility cap (D16, ours, Yuan TSG 2011); the support that
tampers the fewest meters over the window is kept. The goal rides on each branch's true drift and
reaches S_max at the last snapshot (D9, eq. 25); no per-snapshot bound (D11); two target lines by
default, one as a fallback (D17). On the 2019 pool Am is feasible in every window tried, drives the
target to its rating, and tampers a median of about 70 of 511 meters (versus the whole-area 215 of an
unminimized solve). `At` keeps its sub-natural ramp. For Ben to confirm: (a) the local solves are
`attacks/attacker.py` functions, not a `Network.local_solve` (kept out of `grid`, which imports only
`config`); (b) the fewest-tamper objective counts distinct tampered meters, not [1]'s device grouping,
which nyfed's flat meter plan does not carry; (c) generator limits bound the reported net injection
within the units' widened capability, since the pool stores no generator/load split per bus;
(d) `Episode.target_branch` is a single int, so a two-line episode records only its first line.

Status A4: built as an option of `MsFdia` (`step_bound`, `frame_hour`, `kappa`, `family="Ap"`), with
`natural_steps`, `generator_ramps` and `wrap_angles` in `attacks/attacker.py`, `Case.ramp` from
`nygrid_units`, and `Attacker.false_flow_state(..., step=, start=)`. Ap (ours) keeps every scan-to-scan
change of a pretended load within its natural five-minute step at that hour (95th percentile of
\|dP\| over the 2019 pool), from zero at the scan before the window, its reactive change in proportion
to the load's true Q/P, and every generator bus's reported output within its units' ramp (N20). With
the coordinator's fixes N18 to N21 applied to both families, on the 80 windows of the paper's attack
table (kappa 1): Am feasible in 55 (69%, 60 before the fixes), Ap in 3 (4%); median tampered meters 63
(Am, 48 before) and 48 (Ap); median design time 124 ms per scan (Am, 232 before; the warm start
converges faster) and 95 ms (Ap, which fails fast). Every Am attacked bus now lies in zones A to C
(NGRID, NYSEG, RGE): the attacks in zones E to G relied on zero-injection buses absorbing the change.
Ap's three all target branch 216 in zone A (NGRID, NYSEG). Am's largest load step against the natural
step is a median 4.0 times (11.6 before), at most 25. Bounding only the loads (generators (22)-(23) only)
leaves Ap 13 of 80. Sign changes of a pretended load remain in about 45% of Am episodes (edge loads
crossing zero as the solution re-routes, or swinging from one side of the 0.5 cap to the other) and in
4 of 29 Ap episodes; saw-tooth increments fell from 56% to 37% of Am episodes. For Ben to confirm:
(a) Ap as an `MsFdia` option, not a new class; (b) approved (coordinator): `Case.ramp`, `generator_ramps`,
`wrap_angles`, `false_flow_state(start=)`, `natural_steps` and the `step` argument;
(c) `frame_hour [T]` passed to `MsFdia` (the window carries no clock); (d) `FAMILIES` gains "Ap" as code 3;
(e) decided (coordinator): no sign-hold rule; a pretended load may change sign inside a window. Window length
(`out/attack_length_study.json`, 40 windows per length drawn as the paper's attack table): Am feasible
65/88/80/62% at L = 16/32/48/96 scans, Ap 2/8/2/12%, never above 30%; at L = 96 dropping the loads'
reactive bound lifts Ap to 58%, kappa 4 to 32%, unbounded generators leave it at 12%. Ap's window length
is 96 (N23). The P-only Ap of N22 (`Ap-P` in the study file, the earlier rows relabelled `Ap-P+Q`) is
feasible in 35/68/78/58% of the windows at L = 16/32/48/96, median tampered meters 50.5/53/59/52.

### D. Dataset

| block | what | source |
|---|---|---|
| D1 | timeline: split, episode placement, write | [`plan.py#L146`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/generation/plan.py#L146), [`emit.py#L303`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/generation/emit.py#L303), [`write.py#L177`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/generation/write.py#L177) |
| D2 | loader with `ds.client(name)`, which returns only that client's data | [`dataset/__init__.py#L91`](https://github.com/myersben9/fdia-graph/blob/9565688/src/fdia_graph/dataset/__init__.py#L91) |
| D3 | first data release | [`upload_assets.py`](https://github.com/myersben9/fdia-graph/blob/9565688/tools/upload_assets.py) |

Status D3: the build is in the package (`build.py`, `fetch_inputs` and `production_builder`, run by `tools/nyfed/build_timeline.py`; the single-episode files by `tools/nyfed/build_single_episodes.py`). The data release waits on the three release decisions.

Status: D1, D2 built (`timeline/{schedule,builder}.py`, `TimelineWriter` and `open_timeline` in `io.py`,
`dataset.py`, `tests/nyfed/test_nyfed_timeline.py` 4 tests, `tests/nyfed/test_nyfed_dataset.py` 3 tests). The year is split into
three contiguous blocks in time order, train 70%, val 15%, test 15% (from Jan 1, Sep 13 12:00 and
Nov 7 06:00), and each split places 16-frame episodes (80 minutes, the window of [1]) at uniform
random onsets over 20% of its frames, Am and At equal in count; adjacency and gaps come only from that
placement. Each episode is designed on its own window of the 2019 pool with `peak_ratings(case, pool)`
over the whole pool (N8), each from its own generator keyed by family and onset; a window the family
returns None for, or one touching an unsolved frame, stays benign (no redraw, so the attacked share is
below the placed 20%). `out/timeline_2019.h5` (499 MB, 137 s with 24 spawn workers): 105,120 frames;
1,314 slots placed, 1,139 episodes built, Am 345/67/71 and At 460/98/98 (train/val/test), Am built in
73% of its windows (spread evenly over the months) and At in 100%; 17.3% of frames lie in an episode and
16.1% carry a label (a ramp's first and last frames add nothing); 19 pairs of episodes touch, median gap
51 frames. Per episode a median of 58 meters are tampered over 3 clients, every episode spans more than
one client. Frames with a label per client: NGRID 14.8%, NYSEG 11.3%, RGE 6.3%, ConEd 5.4%, CHGE 3.3%,
ORU 3.0%, LIPA 0.6% (one bus, five meters). For Ben to confirm: (a) `open_timeline` in `io.py` (h5py only
there), `SPLITS`/`FAMILIES` code constants in `io.py`, `case/client_names` and `case/tie_lines` stored,
`clients` stored ragged like `buses`; (b) `client(name, split=None)` and `operator(split=None)`; (c) the
`ClientView` contents (`buses`, `frames`, `node_features` and `temporal` as defined in ARCHITECTURE);
(d) `TimelineBuilder`'s constructor and settings and `build` returning a summary; (e) `z`, `z_benign`
stored as float32; (f) the 70/15/15 chronological split leaves the summer peak out of val and test.

Three families (N9 with Ap): `out/timeline_2019_v2.h5` (499 MB, 155 s with 24 workers), the same split
and placement rule, the slots shared equally, Am 307/66/66, Ap 307/66/65, At 306/65/65 placed
(train/val/test). Built: Am 233/49/47 (75%), Ap 28/1/4 (8%), At 306/65/65 (100%); one slot skipped for an
unsolved frame. 12.3/11.7/11.8% of train/val/test frames lie in an episode, 11.5/10.8/10.9% carry a label.
Median tampered meters per episode: Am 56, Ap 64, At 59; median owners 3, 2, 3. Ap's attacked buses lie
in zones A, B, C only (NGRID, NYSEG, one RGE bus). Every Am episode of the new file equals the old file's
at the same onset (Am is placed first, so its onsets are the first draws of the same stream). The old
`out/timeline_2019.h5` is kept. With the coordinator's fixes N18 to N21:
`out/timeline_2019_v3.h5` (116 s with 24 workers), the same placement. Built: Am 226/56/50 (75%), Ap
22/2/5 (7%), At 306/65/65 (100%). 12.0/12.5/12.2% of train/val/test frames lie in an episode,
11.2/11.7/11.4% carry a label. Median tampered meters: Am 66, Ap 62, At 59; median owners 2, 2, 3. Am's
attacked buses lie in zones A, B, C (two in G), Ap's in zone A. No Am or Ap episode moves a
zero-injection bus (78 Am episodes did in v2); At still does in 358 of 436 in v3, built before N18
reached At. v1 and v2 are kept.

Per-family lengths (N22, N23): `out/timeline_2019_v4.h5` (200 s with 24 workers), Am and At 16 scans,
Ap 96 (P-only), the attacked share counted in frames. Placed Am 307/66/66, Ap 51/11/11, At 307/65/65;
built Am 212/50/51 (71%), Ap 39/7/6 (71%), At 305/65/65 (99.7%). 16.3/15.9/15.4% of train/val/test frames
lie in an episode, 15.5/15.1/14.6% carry a label. Median tampered meters: Am 60, Ap 64, At 72; median
owners 2, 2, 3. No episode of any family moves a zero-injection bus. Every family's attacked buses now
lie in zones A, B, C (Am also 10 bus entries in G): At, which reached seven zones before N18, no longer
finds an area to the east, so ConEd, CHGE and LIPA carry no attacked bus. Open for Ben.

With N24 and N25: `out/timeline_2019_v5.h5` (230 s with 24 workers), the same placement as v4. Built Am
125/33/25 (41%), Ap 15/5/4 (33%), At 307/65/65 (100%). 11.4/13.0/11.6% of train/val/test frames lie in
an episode, 10.5/12.2/10.7% carry a label. Median tampered meters: Am 40, Ap 40.5, At 47. Attacked
buses reach zones A, B, C, G, I, J, K (Am), A, B, G, I, J, K (Ap) and A, B, C, E, F, G, I (At), and every
client has positive labels in every split (share of its bus labels, train: NGRID 2.0%, NYSEG 1.4%,
RGE 3.3%, ORU 3.1%, ConEd 0.7%, CHGE 0.5%, LIPA 0.3%). On the 80 paper windows: Am 36% (88% without the
N24 reactive cap), Ap-P at 96 scans 39%, At 100%. The length study was rerun: Am 22/30/40/35%, Ap-P
25/32/30/32%, Ap-P+Q 22/28/30/32% at L = 16/32/48/96.

With N24's floor: `out/timeline_2019_v6.h5` (230 s) is identical to v5 in every reading and label, and the
80-window comparison and the length study repeat v5's numbers exactly. The floor never binds: every New
York load bus of the pool carries |Q|/P = 0.203 (the load model's fixed power factor), above the 0.2
floor. Ap has 15/5/4 episodes in train/val/test; 10 test episodes at its test build rate (4 of 11) would
need about half of the test frames attacked. Decided (coordinator, delegated by Ben): N24 stays the effective 0.5 |Q| cap,
the floor stays in the code and never binds; v6 is the training timeline.

Pattern-aware evaluation set (coordinator decision): `out/timeline_2019_v6_aptest.h5` (130 s), v6's
15,768 test frames (from frame 89,352) with the same times, true states and benign scans, and Ap episodes
only (96 scans, P-only) placed on them at a 75% attacked share by `place_episodes` (generator
"aptest/place", no overlap) and designed with the build's keys `design/Ap/{onset}`; v6's Am and At test
episodes are left out, every other frame is benign. 123 slots placed, 49 Ap episodes built (40%), 29.8% of
frames labelled; median tampered meters 40, median owners 2; attacked buses in zones A, B, G, I, J, K
(NGRID 55, NYSEG 43, ORU 40, ConEd 20, RGE 2, LIPA 1 bus entries). Positive bus-label share per client:
ORU 12.2%, NGRID 3.8%, NYSEG 3.3%, ConEd 2.6%, RGE 1.2%, LIPA 0.4%, CHGE 0 (no Ap episode reaches its two
buses). It is written by `TimelineWriter`, so `Dataset` and `ClientView` read it unchanged; every frame
carries the test split code and the root attribute `eval_only = 1` marks it (the loader does not read
it). Built by a script outside the package (`make_aptest.py`, in the session scratchpad); open for Ben:
whether it becomes `scripts/` or a `TimelineBuilder` option, since the builder's noise is drawn over the
frames it is given and cannot reproduce v6's benign scans on the test block alone.

With N26: `out/timeline_2019_v7.h5` (193 s with 24 workers), v6's placement with Am and Ap trying the most
loaded target line first. Built Am 166/45/61 (v6 125/33/25), Ap 30/6/6 (15/5/4), At 307/65/65.
`out/timeline_2019_v7_aptest.h5` (102 s), built as v6's: 123 Ap slots, 102 episodes (v6 49). v7 is the
training timeline. The five tests on the March window `AP_ONSET` pin `target_order="random"`, the order
the window was chosen under, and a new test checks N26's order.

### O. Operator response

The grid operator under attack: what it does on what it sees, and what that costs on the true grid. Sources
and the full reasoning in the nyfed repository (docs/OPERATOR_COST_LITERATURE.md there) (18 references, verified
by DOI). Decisions N50 to N57.

| block | what | source |
|---|---|---|
| O1 | real-time dispatch on the falsified view: a DC dispatch on shift factors around the pool basepoint, unit limits and five-minute ramps from its own previous dispatch, the interface limit, least movement then least cost | corrective dispatch [4] |
| O2 | the N-1 consequence: the worst single outage of a New York 345 kV branch, the interface flow after it on the true grid | [5] (an N-1 dispatch makes physical consequences of FDIA rare) |
| O5 | O1 acting 1 or 3 scans late | none (a gap in the literature) |

Status: built (`operator/sced.py`, `operator/response.py`, `tools/nyfed/operator_cost.py`,
`tests/nyfed/test_nyfed_operator.py`), run on the four single-episode days (`results/nygrid.operator_cost.csv`,
`results/nygrid.operator_trace.csv`).

## Order

```
F1, F2 -> G1 -> G2 -> G3 -> G4 -> G5 -> M1 -> M2 -> A1 -> A2, A3, A4 -> D1 -> D2 -> D3
                       \-> P1 -> P3 -> P4 -> P5 ----------^
                       \-> P2 ------------------^
```

## Open decisions

| ID | decision | recommendation |
|---|---|---|
| N1 | package name | `nyfed` |
| N2 | year | 2019 |
| N3 | frame cadence | 5 minutes |
| N4 | external areas | decided: fixed boundaries, not clients (`bus_client` -1) |
| N5 | NYPA and municipal buses | decided: the utility whose territory surrounds the bus |
| N6 | light-load overvoltage hours (3.6%) | shunt reactors, otherwise drop the hours. Ipopt's projection (P4) solved all 23 of these hours it was tried on, so they may be MIPS failures rather than infeasible; the pool marks frames next to an unsolved hour `NO_HOUR` |
| N7 | PMU placement | minimum set for observability (built: 12 of the 46 NY buses, each NY bus has a PMU or a PMU neighbour) decided: no extra fixed PMUs; [1] states no density rule, so "trusted" is [1]'s defense chosen at attack time, not a fixed density. Meter coverage decided: flows at every branch end (each client meters its own end of a tie), \|V\| at 60% of buses plus every PMU bus; `plan_meters` approved |
| N8 | line ratings | decided: NYgrid's 12 ratings (G4), every other branch at 1.25 times its peak apparent flow over the 2019 pool (`peak_ratings`) |
| N9 | attack families | `Am`, `Ap` and `At` |
| N10 | clients too small to train | decided: keep all 7 utilities as clients, none merged |
| N11 | one file or one file per client | one file plus a per-client export |
| N12 | bus locations (NPCC coordinates are jittered) | decided: match each NY bus's NPCC substation name to OpenStreetMap substations or EIA-860 plants and own it by the territory holding that point; a name that is not found takes its utility from its NYISO zone and neighbours, recorded in the `note` column of `ny_bus_locations.csv` |
| N13 | HVDC links in the pool | decided: the pool runs on the AC case without them, the case the MATLAB recipe validated |
| N14 | branch flows | decided: computed on demand by `Network.flows`, not stored in the pool |
| N15 | load change inside an hour | superseded by N32 (was: carried by the reference bus 74) |
| N17 | PMU cadence | decided (Ben 2026-10-27): next step after the first full results. PMUs report every 30 s to 1 min between the 5-minute SCADA scans (sub-5-minute operating points from interpolated loads plus short-term variation, re-solved in AC). The hybrid estimator's PMU pseudo-measurements ([1] eqs. 1-3) are used now, in the paper's recovery |
| N16 | tie ratings in the projection | decided: not enforced, as in acproj.m; three ties to neighbouring areas (29-37, 66-134, 81-125) exceed their public rating in about 60% of 2019 frames, stated as a limitation of the external model, and attacks target branches inside New York |
| N18 | zero-injection buses under an attack | decided (coordinator): a New York bus with no load and no generation, or any New York bus that is not a free injection (`free_injection_buses`: NY loads above the 1 MW floor and generators), keeps its true injection at every scan, on the support's edge as well; the operator meters P and Q at every NY bus, so a 1.9 GW reading at a bus that never exceeds 2 MW is not stealthy (Am, Ap and, by the coordinator's later decision, At: its load goal now prescribes the pretended loads and lets the other free injections of the area and its edge absorb, solved least-squares like the flow goal; At stays feasible in 80 of 80 windows, median tampered meters 74.5 against 59, 7.5 ms per scan against 2.5) |
| N19 | the flow solve's active set | decided (coordinator): a pinned injection is released when the solve without its pin leaves it inside its bounds (a proper active set), and each scan is warm-started from the previous scan's false state carried onto this scan's true state, the active set with it (Am and Ap) |
| N20 | Ap's generator bound | decided (coordinator): a generator bus's per-scan change of reported output stays within its units' five-minute ramp, NYgrid's `maxRampAgc` (MW/min, MATPOWER RAMP_AGC, the column `updateOpCond.m` L235 copies; `maxRamp10` is exactly ten times it) times 5, summed at the bus (`Case.ramp`, `generator_ramps`); nuclear units at zero change (they do not follow load); hydro, imports and units without ramp data keep (22)-(23) only; where the pool's own dispatch moves faster than the ramp (a commitment change) the bound widens to the true step, so the attack never adds to it but need not hide it; reactive output (22)-(23) only; loads keep the reactive bound in proportion to Q/P |
| N21 | angle readings across +-180 degrees | decided (coordinator): every difference of angle readings (kinds va and ia), scan to scan (`ClientView.temporal`) and attack to truth (`Attacker.tampered`), is wrapped to (-180, 180]; stored `z` stays raw |
| N22 | Ap's load bound | decided (Ben, 2026-10-27): Ap bounds only the active-power step of each pretended load, by its natural five-minute P step at that hour, kappa 1; reactive load is limited only by the 0.5 load cap. Operators forecast and check real power, and the fixed power factor of the model's loads is a simplification. The P+Q variant (`MsFdia(reactive_step=True)`) stays available, as the paper reports both |
| N23 | Ap's window length | decided (Ben, 2026-10-27): 96 scans (8 h) for Ap, 16 for Am and At; the D9 goal reaches the rating at the last of the 96 scans. The timeline takes a length per family (`TimelineBuilder(length={family: frames})`), with the same uniform placement and the attacked share counted in frames |
| N24 | reactive load plausibility | decided (Ben, 2026-10-27): the 0.5 load-plausibility cap also applies to reactive load, \|dQ\| <= 0.5 \|Q_true\|, in Am, Ap and At, enforced in the same active set. On the 80 paper windows it lowered Am's feasibility from 88% to 36% (with N25). Amended (Ben, 2026-10-27): the reactive cap has a floor, \|dQ\| <= 0.5 max(\|Q_true\|, 0.2 \|P_true\|), in all three families, so a load near unity power factor may still move its Q |
| N25 | zero-injection buses on an attack area's edge | decided (coordinator): a New York bus on the edge whose injection the attack may not change (a zero-injection bus, a load under the 1 MW floor) moves into the interior, its voltage free and its injection held, so the edge moves out a ring; repeated until every edge bus is a free injection, the slack (whose voltage is the reference) or a bus outside New York, which is the limit of the growth (`Attacker._absorb`, all three families). Correction (ours), which reverses part of our earlier N18 implementation: that build held every non-free bus of the edge at its true injection, buses outside New York included. N18's reason (the operator meters P and Q at every New York bus) does not cover them, since no meter sees an external bus's injection, and holding them over-constrained the grown areas (Am fell to 20% of the paper windows, all in zone A). A bus outside New York on the edge now absorbs the change again, as before N18; New York buses that are not free still hold |
| N26 | target line order | decided (Ben, 2026-10-27, on the coordinator's recommendation): Am and Ap try the eligible target lines in order of true loading (apparent flow over rating at the window's last scan), highest first, instead of at random (`MsFdia(target_order="loading")`, ours; [1] fixes its targets per scenario). On the 80 paper windows it raised Am's feasibility from 36% to 71% and the target's loading from 0.51 to 0.61, with no fall in tampered devices (about 10) or channels, so the count is the support's footprint, not the headroom |
| N27 | generator reactive realism | decided (Ben, 2026-10-27): on v7 about half the Am and Ap overloads (29 of 57, 28 of 53) were built mostly of reactive power, generators pretending swings of 190 MVAr at the median and 600 at the 90th percentile, reported Q/P on the target 1.3. Every family's pretended generator reactive change now stays within 0.2 of its units' Q range (`Attacker(gen_q_share=0.2)`), and under a step bound a generator's reported reactive output moves per scan by at most its MW ramp |
| N28 | load cap | decided (Ben, 2026-10-27): the plausibility cap is 0.1 of the load, P and Q (with N24's floor), replacing 0.5; on v7 the median attack moved some load by the full 50 percent |
| N29 | ratings | decided (Ben, 2026-10-27): unrated branches are rated at 1.05 times their 2019 peak apparent flow (`peak_ratings(factor=1.05)`), replacing 1.25, so a line runs close to its rating in its peak hours and an attack needs a small change. Amended by N47 to 1.0 Superseded by N48. |
| N30 | attack exit | decided (Ben, 2026-10-27): Am and Ap end with exit scans in which every pretended injection unwinds toward zero by at most its step bound per scan (natural load steps, generator ramps), Am's rise staying unbounded (`MsFdia(exit_scans=..., bound_rise=False)`); an attack not fully unwound by its last scan does not exist. Replaces the one-scan snap back to the truth |
| N31 | bus loads within the hour | decided (Ben, 2026-10-27, "why does this data look so smooth"): measured on pool_2019, the five-minute load steps of buses in one zone correlate at 0.95 to 0.99 and the zone explains 99 percent of a bus's step variance, because the bottom-up profiles were collapsed to hourly and each bus's share interpolated between hours. Now: the ResStock/ComStock prototypes are kept at their native 15 minutes, each bus gets a size-scaled AR(1) term calibrated on the real zones' fluctuation within the hour and industrial switching steps drawn from zone D's 2019 step statistics, and the buses are rescaled to sum to the zone's real five-minute load |
| N32 | generators within the hour | decided (Ben, 2026-10-27, replacing the AGC-participation version of the same day): every five-minute operating point comes from a ramp-constrained AC OPF run in sequence like NYISO's real-time dispatch (RTD), `AcSolver.dispatch` in `PoolBuilder.build`. Each frame minimizes NYgrid's generation cost (NYgrid stores no gencost; updateOpCond.m L470-504 builds it each hour: thermal units linear at the week's fuel price times HeatRateLM_1 (allocateGen.m 'lm'), nuclear 1 + 2 rand and hydro 10 rand $/MWh after rng("default"), so fixed numbers, external units at the hourly LBMP of NPX, O H, PJM, H Q) on that frame's N31 bus loads, each unit within Pmin/Pmax and within its five-minute ramp of the previous frame's basepoint. Ramps per unit by the rules of `generator_ramps` (N34: maxRampAgc, hydro 10% of nameplate per minute, imports 20 MW/min, nuclear 0, others unbounded), passed to `PoolBuilder` as `ramp [G]`. Nuclear holds its hourly NYgrid output. Unit reactive limits 0.312 Pmax (N34); a frame infeasible under them is solved again with NYgrid's limits and counted. A power flow at the basepoints leaves the reference bus only the residual mismatch. The pool stores each unit's basepoint `unit_p [T, G]`. Built in week chunks run in parallel, each starting from NYgrid's hourly AC point (`project`) at its first hour |
| N33 | generator pretended output | decided (Ben, 2026-10-27, "how fast realistically the generators can pump stuff in"): in every family a generator's pretended change moves per scan within its units' five-minute ramp (the Am rise included) and in total within a band around its dispatch basepoint, the band 3 percent of the units' upper operating limit (`generator_band`), NYISO's tolerance for a unit's output against its RTD basepoint (MST Rate Schedule 3-A, 15.3A.1; NYISO updates basepoints every 5 min and scans unit telemetry every 6 s, Manual 2, 4.3.1, 4.11.1). The operator knows each unit's instruction, so a load is the stealthier place to hide a change. Supersedes N27's 0.2 share of the Q range, which bound nothing |
| N34 | missing generator data | decided (Ben, 2026-10-27): NYgrid's Q limits are MATPOWER placeholders (plus or minus 9999 MVAr per unit), and hydro and imports carry no ramp, so 10 of the 24 New York generator buses were unbounded (one unit with no ramp made its bus inf). Q limits are plus or minus 0.312 Pmax, power factor 0.95 both ways (`generator_q_capability`; NYISO Manual 2, 3.6; FERC Order 827), hydro ramps 10 percent of nameplate per minute (USBR Hydro Model Data Project, 2013), an import 300 MW per 15 min (ISO-NE CROP.31001, the NY real-time interchange ramp; a model parameter for the other interfaces), and every unit is bounded separately (`generator_ramps`) |
| N35 | OPF everywhere | decided (Ben, 2026-10-27, "everything should honestly be opf"): (1) the grid, every five-minute operating point a ramp-constrained AC OPF run in sequence like NYISO's real-time dispatch, its output each unit's basepoint (replaces N32's AGC participation); (2) the attacker, `attacks/opf_attack.OpfFdia`: one multi-period AC OPF over the attack area per target set (Pyomo, Ipopt from the env), minimizing the total absolute pretended injection change (the convex stand-in for eq. 12's l0) under the goal (D9, with N30's reversed exit) and every bound of N22, N24, N28, N33, N34; [1] likewise has its attacker run a local OPF. Replaces MsFdia's support search in production |
| N36 | attack timing | decided (Ben, 2026-10-28): under every realistic bound (N27 to N35) an overload attack exists on 0 of 24 uniformly random onsets and 4 of 24 onsets where some eligible line is at 0.9 of its rating or more, so the attacker waits for a loaded grid. Onsets are still drawn uniformly at random, but only among scans where some eligible line is at 0.9 of its rating or more at the last rise scan (about 5% of 2019). Replaces uniform placement over the whole year for the overload families. Replaced by N48 (onsets from real 2019 interface loading). |
| N37 | basepoint band hard | decided (Ben, 2026-10-28): the 3% band of N33 stays a hard bound on the attacker though NYISO uses it for settlement, not as an alarm, since a reported output beyond it shows in performance tracking. Without it 5 of 24 random onsets have an attack. Also dropped, as untraced to any source, the reactive ramp equal to the active ramp. Reactive output is bounded only by the N34 capability. |
| N38 | reactive bound with no shunts | decided (Ben, 2026-10-28): the cited capability of N34 is infeasible for the grid itself in all but 28 frames of pool_2019_v2 (the reduced NYgrid model has no shunt compensation, and the reference bus needs 1.4 to 4.1 GVAr against 645 MVAr), so the grid keeps NYgrid's limits and the attacker's reported Q stays within max(0.312 Pmax, the unit's true Q) in magnitude. The paper states the missing shunts as a limitation. |
| N39 | dispatch anchor | decided (Ben, 2026-10-28): the pure economic five-minute OPF put hydro at its 4,270 MW nameplate all year and imports 1.9 GW high in January (19.6% below NYgrid's cost). Hydro and interchange now follow NYgrid's hourly output interpolated to five minutes, being energy and schedule limited, and the OPF re-dispatches only thermal units within the hour, like RTD following the hourly commitment. |
| N40 | switching base | decided (Ben, 2026-10-28): industrial switching stays calibrated per MW of the model's 125 MW industrial load in zone D, and the drop in the hourly validation (load factor 0.92 to 0.75, spectrum 0.78 to 0.61) is reported as the cost of visible switching. |
| N41 | splits by week | decided (Ben, 2026-10-28): under N36 the contiguous test block (mid-November to December) holds 42 near-limit frames of 5,336 (0.27%), room for about 6 Am windows. The year is cut into whole weeks assigned 70/15/15 to train, val and test by a seeded draw stratified on each week's near-limit frames, so every split holds summer, winter and loaded hours. No episode crosses a week block, so no window is shared between splits. Replaces the contiguous blocks. |
| N42 | generator output and onset redraw | done (2026-10-28, ours, a fix and Ben's "please fix this"): the v8 build on pool v2 designed 1 of 141 Am and 2 of 37 Ap. Two causes. First, a bug. At a generator bus that also carries load, `OpfFdia` bounded the net injection as if it were the units' output, so load at the bus held it under the units' minimum and the bus could never report less, while that load could not move at all. The units' limits, band and ramps now bound the units' own output (pool `unit_p`), and the load at the bus moves within the 0.1 load cap. Second, with every bound an attack exists at about 1 in 12 near-limit onsets (16 windows, one bound relaxed at a time: all 1, no band 5, no Q cap 6, none 15; no Q cap restores the fake reactive of v7 since 91% of NYgrid units carry a 9999 MVAr placeholder). Redrawing a failed onset at random filled under 1% per try, since the onsets with an attack cluster in time. So the builder first designs every allowed Am and Ap onset on an hourly grid (`survey`), then places the episodes uniformly among those with an attack and reuses their designs. Every bound stays as decided, and the survey gives the share of near-limit hours where a bounded attack exists. |
| N43 | loads only, tried and reverted; Am alone | decided (Ben, 2026-10-28). Ben asked to rule out generator and zero-injection buses (zero-injection buses were never pretended, since only loads above 1 MW are). With loads alone and every other bound, Am existed at 0 of 68 near-limit onsets over two July weeks, and at 16 near-limit onsets over the year at 1 of 16 with a load cap of 0.5 or 1.0 and 0 of 16 at 0.2 or 0.1. Near the loaded lines the attack area is mostly generator buses, and only 3 to 6 loads are left to carry the false state. In [wu2026dynamic], eqs. (13)-(14) tamper the injection at any node and (22)-(23) bound the attacked generator output, so the formulation fakes generator output within its limits. Ben reverted, so generators are pretended within their limits again (`OpfFdia(loads_only=False)`, the default; `loads_only=True` stays for the study). The dataset is built with Am alone first (`timeline_2019_v8_am.h5`, survey every 15 minutes), so one family is right before the next is added. |
| N44 | dead buses outside New York hold | done (Ben, 2026-10-28): bus 140 (TE, a PJM tie point near Warren, Pennsylvania, radial off bus 60 Dunkirk on a 550 MW line) has no load and no units, so line 60-140 carries no flow in the pool all year. The attacker let any bus outside New York absorb, so zone A Am attacks reported 3 to 83 MW on 60-140, and a per-meter step detector caught every one of them there. A bus outside New York with no unit and no injection at the state now holds its injection like a New York zero-injection bus (`Attacker._held_bus`, used by `_absorb`, `false_flow_state` and `OpfFdia`). Every other bus outside New York still absorbs. On a spot check of 4 zone A v8 onsets, 3 have no attack after the fix, and on the fourth 60-140 moves 0.05 MW (11.9 before). Bus 140 is not the Hydro-Quebec import, which NYgrid models at bus 48 (Moses). Rebuilt on pool v3 as `out/timeline_2019_v9_am.h5`: survey 84 of 1330 onsets, 21 Am built (train 9, val 11, test 1; zones A 10, I 11; v8 had 31, A 21, I 10), line 60-140 moves at most 0.06 MW. The step-detector control now gives the same result with or without the tie meters: the end is caught in 3 of 21 with the exit and 12 of 21 when cut off after the rise (`paper/build/exit_control_v9.csv`). |
| N45 | HQ import bound | done (Ben, 2026-10-29): pool v2 peaked the HQ import at 1825 MW against NYgrid's 1500. NYgrid bounds the external units at 1.5 times their setpoint (updateOpCond.m L360) and appends the HQ import afterwards with its own -1100 to 1500 MW (L367-377), but the projection and the 5-minute limits gave HQ the 1.5x rule too. HQ now keeps its case limits, widened only to admit NYISO's schedule (HQ-NY plus Cedars exceeds 1500 MW in 3407 of 8760 hours of 2019, max 1787 MW), so it goes past 1500 MW only where the schedule does (`AcSolver.project`, `PoolBuilder._limits`). On 2019-10-02, the day with the most overshoot in v2, every frame solved and HQ peaked at the schedule's 1600 MW. Rebuilt as `out/pool_2019_v3.h5`. |
| N46 | splits on where attacks exist | done (Ben, 2026-10-29): the v9 Am build put 1 of 21 episodes in test, because N41 stratified the weeks on near-limit hours and most near-limit hours hold no attack (test had 1 hit in 144 surveyed onsets). With a survey, the builder now surveys every onset whose window lies in one week first, then stratifies the weeks on the onsets where an attack exists (`TimelineBuilder.build`, `split_weeks(stratum=...)`, `week_index`). The 85 hits sit in 9 weeks, one of them holding 34 (40%), so a 20-week stratum still left test 2 episodes. Strata of 5 weeks (`HIT_STRATUM`, a 3/1/1 draw) give placed 15/3/3 in preview, at a cost: the week shares become about 60/20/20 instead of 70/15/15. Day blocks would keep 70/15/15 with balanced attacks, but adjacent days in different splits share weather, so they were not used without Ben. The v9 survey was reused (every new candidate was in it). Built as `out/timeline_2019_v10_am.h5`: 21 Am, train 15 (32 weeks), val 3 (11 weeks, March), test 3 (10 weeks, June and July, all zone A). The 21 onsets are the same as v9's, since the attacks cluster and 21 is about all that fit without overlap; only which split holds them changed. Exit control on v10: end caught in 2 of 21 with the exit and 11 of 21 when cut (`paper/build/exit_control_v10.csv`). |
| N47 | rating factor 1.0 | decided (Ben, 2026-11-01): unrated branches are rated at their 2019 peak apparent flow (`peak_ratings(factor=1.0)`), so each line's busiest scan of the year sits at its rating, the reason N29 gave for 1.05, without the 5% margin. At 1.05, v10 holds 21 Am, all in zones A and I. Measured on pool v3 with the N36 onset rule: scans where some candidate line carries at least 0.9 of its rating rise from 3.9% to 11.7% and allowed Am onsets from 3,937 to 12,008, while scans with a true flow above a rating stay at 0.05%, so the pool needs no rebuild. A random 120 of the allowed onsets held an attack in 10 (8.3%, against 85 of 1,325, 6.4%, at 1.05), in zones A, C, E and I. Factors of 0.95 and below put 3% or more of benign scans over a rating and wait for a dispatch that enforces branch limits. Built as `out/timeline_2019_v11_am.h5` (pool v3, the v10 settings, 12 workers, 5.2 h): the survey found an attack at 310 of 3,997 onsets (7.8%), and 79 Am were placed and built, train 53, val 14, test 12, in zones A 35, I 28, C 9, E 3, B 2, G 2 (test: I 8, A 2, C 2). 2,430 scans carry a label (2.31%; train 2.59%, val 1.94%, test 1.85%) and 11,914 bus labels; every client holds labels (NYSEG 4,485, NGRID 4,358, ConEd 2,435, LIPA 300, RGE 233, CHGE 53, ORU 50). Superseded by N48 (interface goal). |
| N48 | overload target: NYISO interfaces | decided (Ben, 2026-11-02): the overload goal moves from one branch's peak-flow rating (N8, N29, N47) to a NYISO interface. A rating at a factor of the 2019 peak flow is circular (the rating comes from a dispatch that saw no rating) and puts every line at its rating once a year; NYgrid's branches are reduced equivalents (no branch inside New York has a published rating, and the 345 kV candidates' 2019 peak flow is a median 3.1 times their surge impedance loading, up to 10), so no per-line physical rating can be recovered. (1) Am's goal: the reported signed P sum over an interface's branches (NYgrid `updateOpCond.m` L420-430) reaches that hour's real 2019 positive limit (`interflowHourly_2019.csv`), the number NYISO's real-time dispatch enforces. (2) Onsets: the last rise scan falls in an hour whose real 2019 flow is at least 90 percent of the limit (replaces N36's model near-limit rule). Real 2019 hours at or above 90 percent: Central East-VC 4,337 (never above 100), Sprain Brook/Dunwoodie South 115, UPNY-ConEd 34, Moses South 10, Total East and Dysinger East 0. (3) The pool's dispatch gets the interface limits, as NYgrid's own OPF does (open since N16); the operator-response design needs the same constraint. A limit only caps flow, so it does not lift the model's flows to the real ones: on pool v3 the model's median loading is Central East-VC 73.3 percent of the limit (real 89.9), Moses South 63.6 (59.8), UPNY-ConEd 41.4 (42.3), Sprain Brook/Dunwoodie South 43.5 (60.5). G4 still holds, since the goal is the interface sum and needs no per-branch number. Feasibility under every production bound (scratch `interface_feasibility.py`, `out/scratch_iface_capacity.csv`, 2 onsets per interface at real loading of at least 90 percent, pushes of 5 to 400 MW above the model flow): the real limit itself is out of reach (model flows sit 729 to 1,820 MW below it); the largest push found is Sprain Brook/Dunwoodie South 400 MW (both onsets, 18 meters), Moses South 25 MW, Central East-VC 5 MW, UPNY-ConEd none, since both its branches end at bus 74 (Indian Point, two nuclear units), the case's slack, which the attacker cannot move. Open for Ben: the goal as the model flow plus the hour's real 2019 headroom, and which interfaces |
| N49 | where the overload attacker strikes | decided (Ben, 2026-11-02, "realistically the attacker would target a line that is already close to his rating because he could change it by above the noise but a small amount each time to get to that rating easier, I think it's ok if we just have 1 or 2 places like that"): the attack bounds (N27 to N35) and the attacker's area stay as they are, and Am strikes only an interface already close to its limit, closing the last gap in above-noise steps over the rise. Under those bounds the reachable places are few: the largest push found is Sprain Brook/Dunwoodie South 400 MW, Moses South 25 MW, Central East-VC 5 MW, UPNY-ConEd none (N48), while the real 2019 grid never came within 50 MW of any internal interface limit (hours with at most 100 MW of headroom: Central East-VC 11, Moses South 7, the others 0). Proposed implementation, to confirm: since the model's flows differ from the real ones (N48), the goal is the model's interface flow plus that hour's real 2019 headroom (limit minus real flow), so the attacker adds what the real grid had left. Measured with that goal (`out/scratch_iface_headroom.csv`, 20 random onsets per interface among those whose last rise scan falls in a real 2019 hour at 90 percent of the limit or more): Sprain Brook/Dunwoodie South 19 of 20 (pushes 150 to 433 MW, 3 or 4 attacked buses, 17 to 32 tampered meters, from 1,380 such onsets in June to August), Moses South 0 of 20, Central East-VC 0 of 20 (real headroom 100 to 262 MW). Built as `attacks/interface_attack.InterfaceFdia` with `grid/interfaces.py` (2026-11-02). Two fixes to the minimization, after Ben asked whether 17 to 32 tampered meters was minimal: on 30 Jul 12:55 every support of 1 to 3 area buses (534) gives at best 18, with two buses' voltages moved (79, 82), where the search had kept 20, because it grew every support from all three of the interface's end buses and solved only 3 windows; supports now grow from one interface branch's ends at a time (6 windows each), and each window is solved with a meter objective, every P and Q flow meter and voltage meter around the interior counted beyond its noise as eq. 12 counts meters (the injection terms alone cover only part), falling back to OpfFdia's objective where Ipopt finds no point. Meters on four onsets, old search to new: 20 to 15, 18 to 18, 32 to 29, 19 to 20 |
| N50 | the operators (D1) | decided (Ben delegated, 2026-11-03, "make all the decisions for me", toward the Transactions paper): O1 the real-time dispatch on the falsified view, O2 the N-1 consequence of O1's outcome, O5 O1 acting 1 or 3 scans late. A separate corrective operator (O3) is O1 itself once the interface limit binds. Load shedding (O4) waits until O1 leaves a true violation it cannot clear. O2 measures and does not dispatch, since NYISO's interface limits are themselves set from contingency studies, so a second N-1 dispatch against the same limit would count the contingencies twice |
| N51 | the operator's model (D2) | decided (delegated): a DC dispatch on shift factors around the pool basepoint (NYISO's real-time dispatch is a DC dispatch on shift factors), checked on the true grid with an AC power flow at the true loads |
| N52 | the operator's interface limit (D3) | decided (delegated): the benign model flow plus that hour's real 2019 headroom, the quantity Am pushes toward (N49). The pool is not rebuilt |
| N53 | how the operator moves (ours) | decided (delegated): least total movement first, then least cost at that movement, so a scan that needs nothing moves nothing and the benign basepoint stays as the pool dispatched it. A least-cost dispatch from the basepoint would instead reshuffle units wherever the DC costs disagree with the AC pool dispatch, in the benign twin as much as in the attacked one |
| N54 | the true load and regulation (ours) | decided (delegated): the attack changes what the operator reads, never the true load, so the generation its dispatch removes or adds is picked up by regulation, the online free units in proportion to their five-minute ramp (automatic generation control) |
| N55 | the cost terms (D4, D5) | decided (delegated): physical first, the dispatch moved, the regulation called (MW, MWh), the interface flow the operator could not relieve, the true interface change and the N-1 change (MW). Then money, the production cost change at the units' costs and the energy change settled at the real 2019 hourly zonal LBMP (NYgrid's `priceHourly_2019.csv`). No value of lost load. Regulation energy is split by the utility whose territory holds each unit |
| N56 | closed loop (D6) | decided (delegated): closed loop on the dispatch (each scan from the operator's previous one, in the attacked and the benign twin alike), open loop on the data (the released scans are not changed). The attack was designed against the benign path, a limitation stated in the paper |
| N57 | overshoot what-if (ours) | decided (delegated): the attacked interface reading pushed 50, 100 or 200 MW past what the attack reached, in proportion to its push, to show how the consequence grows if an attacker overshoots the limit. A what-if on the reading, not a new attack design |
| N58 | the damaging variant (ours) | decided (Ben delegated, 2026-11-03, "don't we want attacks that are damaging", option A): the overload attack also in a damaging form whose goal is the limit plus an overshoot, designed under every bound as a real attack (`InterfaceFdia(overshoot=...)`), and per onset the largest overshoot the bounds allow (`max_overshoot`, bisection from 0 to 800 MW to 25 MW, which assumes feasibility falls with the overshoot). N57's overshoot stays the what-if on the reading in `tools/nyfed/operator_cost.py`. At zero the goal is N49's exactly |
| N59 | the operator's escalation | decided (delegated): redispatch first. While redispatch cannot relieve an interface, the flow over the limit is priced at the 2019 transmission shortage curve. Once the interface has been over its limit for 15 minutes (3 scans), load relief at the buses where a MW shed takes at least 0.1 MW off the interface. From the NYISO Emergency Operations Manual 4.5 (interface brought back within 15 minutes, load shedding ordered within 30 minutes of the overload) |
| N60 | prices and reference values | decided (delegated): the 2019 shortage curve (350 $/MWh for 5 MW, 1,175 for the next 15 MW, the 4,000 cap beyond, NYISO filing of April 2017), shed energy at the LBNL-6941E values for residential and medium and large C&I customers as a range (no northeast data in that study, stated), the regulation read against the 175 to 275 MW summer requirement (the table in force from September 2020, stated) |
| N61 | exit length (recorded) | set in the code with N30 and recorded here: an exit as long as the rise (16 scans each for Am, `build.RISE`), the pretended change unwinding to nothing at the last scan |
| N62 | the interchange-consistent attacker and the tie-line check | decided (delegated, 2026-11-03, after three internal reviews of the TSG draft found the at-the-limit damage rests on the attack moving New York's net interchange): N25 lets a bus outside New York on the attack's edge absorb the change because no New York meter sees it, but every tie is metered at both ends and the readings are exchanged with the neighbour, and the balancing authority computes load as generation plus net interchange. `Attacker(hold_external=True)` keeps every external injection at its true value (an external bus stays on the edge, its injection held by the solve, so every tie keeps its true flow). On the 60 drawn onsets the consistent attacker reaches the limit at 1 and overshoots at none. The operator gets the matching check, `tie_gap` (the interchange it reads at the New York ends less the neighbours' meters), alarming beyond four standard deviations of the benign day's gap |
| N63 | the regulation the operator holds | decided (delegated): regulation is capped at 275 MW, the summer requirement at its highest hour (N60). Imbalance beyond the cap is reported as an ACE excursion (MW and MW-minutes), not as regulation |
| N64 | every interface enforced, quick-start units | decided (delegated): the operator enforces all four mapped interfaces (each limit as N52, each with its own 15-minute clock and its own shortage price), and may start the small gas, oil and kerosene units of zones J and K (at most 200 MW, half their range per scan, a ten-minute start), which makes the load relief a lower bound |
| N65 | the defended operator | decided (delegated): with the interchange check the operator stops acting on the tampered telemetry from the first alarm and reads the scans as they would be without the attack, the most a defense that finds the attack at that scan can save. The damage avoided is the undefended run less the defended one |
| N66 | three operators | decided (delegated, after the second internal review found the AGC reading the true imbalance inconsistent with an attack on the tie meters): `respond(mode=...)`. Trusting: dispatch on the metered injections, AGC on the area control error read through the New York-end tie meters the attack falsifies, so the shortfall the false load causes is hidden and the neighbours supply it as inadvertent interchange. Accounting: load as generation plus the interchange agreed with the neighbours and AGC on the agreed values (NERC BAL-005-1 R7, a common-source meter for each tie), so neither carries the false interchange while the interface read is still the attacked one. Oracle: trusting until the interchange alarm, then the scans as they would be without the attack, an upper bound on what any defense that finds the attack then can save. Replaces N65's single defended operator |
| N67 | the neighbours' meters and the alarm's calibration | decided (delegated): the neighbour's reading of each tie goes through a meter of the same accuracy class as the New York end, with its own constant bias from a fixed stream and a fresh jitter per scan, which is conservative against BAL-005's common-source meter. The alarm's mean and standard deviation are fitted on January to March of the benign pool and applied to every day (`tools/nyfed/tie_check.py`): threshold 55 MW, no false alarm in 79,030 benign scans of April to December. The check catches a false value injected in New York's own data path; a compromise of the common-source meter itself is outside it |
| N68 | a certificate for the consistent attacker | decided (delegated): per onset and interface, the largest push of the reported interface flow any attacker can make in the DC model with every external injection held and New York's net injection unchanged, each attackable load within tau of its load and each unit within the band of its upper limit (an upper bound: stealth, AC and step bounds only shrink it), swept over tau 0.05 to 0.3 and the band 3 to 10 percent (`results/nygrid.push_bound.csv`). On Sprain Brook/Dunwoodie South at tau 0.1 and the 3 percent band the push reaches the hour's headroom at 9 percent of the onsets, at a 5 percent band at 90 percent: the basepoint band is the bound that protects the interface |
| N69 | where the inadvertent interchange enters, and the margin | decided (delegated, after the third internal review found the shortfall landing on the reference bus at Indian Point): the generation the dispatch removes and regulation does not restore is taken up by the synchronous units outside New York in proportion to their capacity (Hydro-Quebec excluded, an asynchronous tie), before the AC power flow of the true grid. Each scan also reports `margin`, the tightest true margin of any interface to its limit, and the summary `margin_min_mw` and `margin_lost_mw`. On the two draws the trusting operator then raises another interface by a median of 159 MW at the limit and 307 MW damaging (up to 468 MW) and loses up to 230 MW of the tightest margin, never below 72 MW; the accounting operator stays below 22 MW and 52 MW |
| N70 | the certificate's reach | decided (delegated): `tools/nyfed/push_bound.py` bounds four attackers that keep the interchange true, `net_local` (the DC form of the search's attacker: the angles of one subnetwork and the zero-injection buses it absorbs free, every other angle held, the best anchor taken), `net` (any New York bus), `net55` (as `net` plus up to the check's 55 MW entering from the external units by capacity) and `net_ties` (as `net` with every tie's flow held), with the search's own injection bounds: a generator bus moves its units within the band and their operating limits and the load it carries (units less injection, N42) within the cap, the reference bus holds. The first version gave a generator bus only the band, which made the bound far too tight (found in the third internal review, where the AC search's one consistent attack exceeded it). Corrected, at tau 0.1 and the 3 percent band the local attacker reaches the headroom on every Sprain Brook onset (median push 564 MW against 374 MW), 25 percent of Central East and none of Moses South or UPNY-ConEd; at tau 0.05 42 percent of Sprain Brook. The load cap decides it, not the band. A form with the ties held and external injections free was dropped: it is unbounded in the DC model |
| N71 | why the AC search rarely finds the consistent attack | measured (2026-11-03): with the whole subnetwork as its support the AC static solve is still infeasible on the onsets the DC bound opens, and lifting the generators' reactive limits does not change that, while widening every bus voltage limit by 0.02 pu makes it feasible on every onset tried. The NYC generator buses of the Sprain Brook subnetwork sit at vmax 1.10 in 85 to 100 percent of the benign frames (the reduced network carries no shunt compensation), so the consistent attack is closed by pinned voltages, a property of the reduced model. `damage_study.py --v-margin` widens the attacker's voltage limits to measure what the consistent attack does to the operators |
| N72 | where the inadvertent interchange enters | decided (delegated, after the fourth internal review): `respond(entry=...)` shares the shortfall among the synchronous external units by capacity (`capacity`, N69), by the areas' approximate 2019 peak loads (`peak`, a proxy for frequency bias), or within one region (`pjm`, `isone`, `west`); `tools/nyfed/entry_sensitivity.py` reruns the trusting operator on every attack day under each. The median true rise of the attacked interface at the limit stays between 102 and 139 MW; with entry from the west or PJM the damaging attack takes some interface over its true limit on up to 11 of 113 onsets, by up to 96 MW |
| N73 | the check over many meter fleets | measured: `tie_check.py --draws 20` redraws both ends' constant biases and calibrates each draw on its own January to March. The threshold ranges 36 to 86 MW and the check raises a median of 6 false-alarm scans on 4 of 275 days (at most 12 days), since a relative bias grows with summer flows; the default fleet's zero was one lucky draw. Reconciling each tie's two meters, or normalizing by the accuracy class at the flow, did not remove them (kept in the json as `reconciled_*`) |
| N74 | the voltage-limit alarm as a detector | measured: `tools/nyfed/voltage_check.py` sets the alarm at the largest benign voltage reading of January to March plus 0.002 pu (1.1058 pu); no false alarm on the 275 test days; it catches all 97 consistent attacks designed with the attacker's voltage limits widened by 0.02 pu (by at least 0.013 pu) and none of the 233 attacks that falsify the interchange. `tie_check.py --attacks` measures detection over the 20 meter draws: 99.1 and 100 percent in every draw, median delays 2 to 4 and 0 to 2 scans, at most 9 |
| N75 | generator voltage schedules | decided (delegated, after the fifth internal review): a generator bus regulates its voltage to a schedule the operator knows, so `OpfFdia.gen_v_tol` (`damage_study.py --gen-v-tol`) holds its false voltage within that much of the true one. With 0.005 pu and every other limit widened by 0.02 pu the consistent attack exists on 7 of 60 onsets, cutting the true transfer by a median of 30 and at most 59 MW, and clears the voltage alarm by only 0.0006 to 0.002 pu. `summarize` also reports the scans, longest run and energy an interface spends past its true limit (`over_*`): with western entry Central East, a median of 4 scans and at most 7 in a row |
| N76 | the closed loop | decided (delegated, after the sixth internal review): `respond(closed_loop=True)` lets the operator read each interface as it truly flows after its last dispatch (the part its shift factors did not predict enters the next reading); `damage_study.py`, `entry_sensitivity.py` and `operator_cost.py` take `--closed-loop`. Closed loop the damage barely moves (true rise 122 MW against 120 open loop, the damaging drop 180 against 231 MW), but the Central East violation of western entry shrinks to 3 onsets of one scan and 9 MW, since the operator now sees and relieves it. `tools/nyfed/closed_loop_bdd.py` rebuilds each attacked scan as the estimator receives it after the operator acts, h(x_closed) + a, and fits it by weighted least squares: open loop the residual is below 4e-8, closed loop the 99 percent chi-square test is more likely than not to fire within the window on 97 percent of the attacks at the limit and all damaging ones, first at a median of scan 10 and 7 |
| N77 | the residual test with the meters' own noise | measured: `closed_loop_bdd.py` now adds each window's own noise realization to the closed-loop scans and tests the benign twin the same way. The benign scans never fire (0 of 7,264). Against the trusting operator the test fires on 95 percent of the attacks at the limit and all damaging ones, first at a median of scan 11 and 7; an operator that falls back at that alarm (`respond(alarm_at=...)`) keeps 99 of 424 MWh of inadvertent interchange at the limit and 76 of 788 damaging, and 1 of 184 MW of forfeited transfer. Against the accounting operator it fires on 1 and 88 percent, on the consistent attack 0 and 67 percent |
| N78 | how the operator answers a residual alarm, and a calibrated test | decided (delegated, after the eighth internal review): `respond(after=...)` lets any operator answer an alarm at `alarm_at` by switching to the scans without the attack (benign), holding its dispatch while its AGC keeps reading the tie meters it reads (freeze) or holding it with the AGC on the common-source tie values (freeze_cs). `closed_loop_bdd.py` calibrates the residual test at the 99th percentile of benign January to March scans (420 against the chi-square 469, a constant bias keeping the residual near 395). Calibrated, the test fires on 99 percent of the attacks at the limit and all damaging ones against the trusting operator (scan 8 and 5), 75 and 97 percent against the accounting operator and 53 and 92 percent on the consistent attack; freezing with the AGC on the falsified ties keeps 425 of 424 and 706 of 788 MWh of inadvertent interchange, with the AGC on the common source 51 and 43 MWh |
| N79 | a persistence rule for the residual test | decided (delegated, after the ninth internal review found the per-scan calibrated test raising about nine alarms a day): `bdd_benign_series.py` gives the benign residual of every scan of whole days, and `bdd_rule.py` alarms when 2 of the last 3 residuals exceed the 99.9th percentile of January to March (428.7). No false alarm day on 40 fit days, 80 test days of April to December or the benign twin of any attack window. It catches 96 percent of the attacks at the limit and all damaging ones against the trusting operator (scan 10 and 7), 4 and 88 percent against the accounting operator and 0 and 73 percent of the consistent attack. Answering it: the trusting operator keeps 425/757 MWh frozen on the falsified ties, 95/82 on the common source; the accounting operator, frozen at the alarm, forfeits 77 rather than 171 MW, the consistent attack 70 rather than 104 |

## References

[1] S. Wu et al., "Dynamic PMU configuration for stealthy multi-snapshot FDIA mitigation," IEEE Trans. Smart Grid, vol. 17, no. 1, 2026. [doi:10.1109/TSG.2025.3615645](https://doi.org/10.1109/TSG.2025.3615645)
[2] H. Li et al., "The creation and validation of load time series for synthetic electric power systems," IEEE Trans. Power Syst., vol. 36, no. 2, 2021.
[3] M. V. Liu et al., "An open source representation for the NYS electric grid," IEEE Trans. Power Syst., vol. 38, no. 4, 2023. [arXiv:2112.06756](https://arxiv.org/abs/2112.06756)
[4] L. Che, X. Liu, Z. Li, "Mitigating false data attacks induced overloads using a corrective dispatch scheme," IEEE Trans. Smart Grid, vol. 10, no. 3, 2019. [doi:10.1109/TSG.2018.2817515](https://doi.org/10.1109/TSG.2018.2817515)
[5] Z. Chu, J. Zhang, O. Kosut, L. Sankar, "N-1 reliability makes it difficult for false data injection attacks to cause physical consequences," IEEE Trans. Power Syst., vol. 36, no. 5, 2021. [doi:10.1109/TPWRS.2021.3061480](https://doi.org/10.1109/TPWRS.2021.3061480)
