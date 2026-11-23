# Check the package yourself

This page lists what the package claims and the test that checks each claim. Run a test to see
the claim hold on your machine. Read the test to see exactly what is compared: each row says the
case the test runs on and the tolerance it uses.

## Set up

Work from a checkout, because the tests are not in the wheel.

```bash
git clone https://github.com/myersben9/fdia-graph
cd fdia-graph
pip install -e ".[generate,test]" -e "./analysis[test]"
```

Run one test by its name:

```bash
python -m pytest -q tests/test_timeline.py::test_a_stealthy_frame_is_the_benign_scan_plus_its_attack_vector
```

Run everything the install above can run (it downloads the IEEE pools it needs, once):

```bash
python -m pytest -q -n 4 -rs tests analysis/tests
```

`-rs` lists each skipped test with its reason. The "Needs" column below says what a test needs
beyond the install above.

| Needs | Meaning |
|---|---|
| nothing | runs after the install above; an IEEE pool is downloaded once |
| `[ny]` | also `pip install -e ".[ny]"`. Without it the whole folder `tests/nygrid` is left out of the run, and asking for one of its tests by name reports "not found" |
| Ipopt | also the Ipopt solver on your `PATH` ([what to install](../concepts/solving.md#what-to-install)). Without it these tests fail with an error |
| inputs | also the public New York inputs, fetched by `fdia_graph.data.inputs.fetch_inputs` ([Build the operating points](operating_points.md)) |

## One path writes every timeline

| Claim | Test | Needs |
|---|---|---|
| Every dataset and attribute of a file is named in `schema.py`, the required ones are present, and no dataset path is spelled outside `schema.py` | `tests/test_schema.py::test_the_fixture_carries_exactly_the_schema`, `test_no_module_but_schema_spells_a_dataset_path` | nothing |
| The same seed and settings give the same file (IEEE-14) | `tests/test_split_first.py::test_the_same_seed_gives_the_same_file` | nothing |
| One worker process and two write the same file, byte for byte (IEEE-14, overload episodes) | `tests/test_split_first.py::test_the_file_is_the_same_for_any_number_of_workers` | nothing |
| A run that fails at its last step leaves no file and keeps a file that was already there | `tests/test_one_path.py::test_a_failed_run_leaves_no_file_and_keeps_the_one_that_was_there` | nothing |
| A stored reference timeline is reproduced bit for bit | `tests/test_timeline_digests.py` | nothing; compared on the platform the snapshot was recorded on (same OS, processor, Python minor version and numpy version), skipped elsewhere |
| The New York case runs through the same source as an IEEE case | `tests/nygrid/test_any_case.py::test_the_original_generator_takes_the_new_york_case` | `[ny]` |

## Splits and placement

| Claim | Test | Needs |
|---|---|---|
| The splits follow their fractions and cover every frame | `tests/test_split_first.py::test_the_splits_are_cut_to_their_fractions_and_cover_the_timeline` | nothing |
| An episode lies whole inside one split and never overlaps another (one plan of 2,000 frames) | `tests/test_split_first.py::test_episodes_lie_whole_inside_their_split_and_never_overlap` | nothing |
| A split by weeks keeps each week whole, on an IEEE pool with times | `tests/test_any_case.py::test_an_ieee_timeline_is_split_by_whole_weeks_when_its_pool_has_time` | nothing |
| Every frame labelled attacked carries an attack | `tests/test_split_first.py::test_every_frame_labelled_attacked_carries_an_attack` | nothing |
| The file records what the placement asked for and what it built | `tests/test_split_first.py::test_the_file_records_what_the_placement_asked_for_and_got` | nothing |

## The attack designs

| Claim | Test | Needs |
|---|---|---|
| On a ramp frame, observed minus benign is the attack vector rebuilt from the frame's labels (to 0.01 MW or 0.1 %), and exactly zero on every untampered channel (first 20 ramp frames, IEEE-14) | `tests/test_timeline.py::test_a_stealthy_frame_is_the_benign_scan_plus_its_attack_vector` | nothing |
| The tamper masks agree with the layers | `tests/test_timeline.py::test_layers_and_tamper_masks_agree` | nothing |
| The local area keeps a boundary and leaves the slack bus out (IEEE-14) | `tests/test_timeline.py::test_the_hop_area_keeps_a_boundary_and_the_slack_fixed` | nothing |
| A ramp never targets a generator bus or the slack bus | `tests/test_timeline.py::test_stealthy_families_never_target_a_generator_bus`, `test_a_slack_bus_load_is_never_a_target` | nothing |
| The fewest-device search returns the cheapest of its candidate supports, as found by solving every candidate (IEEE-14) | `tests/test_minimize.py::test_the_search_equals_brute_force`, `tests/test_overload.py::test_the_search_on_a_flow_goal_equals_brute_force` | nothing |
| The overload reaches the line rating on the support it holds (relative tolerance 1e-4) | `tests/test_overload.py::test_am_reaches_the_rating_on_the_held_support` | nothing |
| A generator on the support's boundary whose P range is shut keeps its P, and a bus whose voltage range is shut keeps its voltage, while the flow goal is met (IEEE-14) | `tests/test_overload.py::test_a_boundary_generator_stays_inside_its_limits`, `test_a_voltage_is_held_at_its_limit_while_the_goal_is_met` | nothing |
| A ramp's false state keeps the generators one branch outside its support inside their limits, as an overload's does (IEEE-14) | `tests/test_timeline.py::test_a_ramp_state_keeps_the_generators_of_its_boundary_inside_their_limits` | nothing |
| A generator limit counts as passed only beyond the accuracy of the bus's injection meter at its true reading, and never within 0.001 MW (IEEE-14) | `tests/test_timeline.py::test_a_generator_limit_is_passed_only_by_more_than_the_injection_meter_resolves` | nothing |
| A bus with a shunt and no load and no producing unit is a zero-injection bus, and the pool and the admittance matrix inject nothing there (IEEE-30, 89, 118, 145 and 300) | `tests/test_timeline.py::test_a_bus_with_a_shunt_and_nothing_else_injects_nothing` | nothing |
| The load cap bounds the active change of every load the attack moves | `tests/test_overload.py::test_the_load_cap_bounds_every_load_the_attack_moves` | nothing |
| The two IEEE-14 scenarios of [WU26] are feasible, reach both ratings and tamper the paper's devices, with one difference the test names. The IEEE-118 scenario is feasible and reaches both ratings | `tests/test_overload_case_studies.py` | nothing; set `FDIA_SLOW=1` for IEEE-118 |
| On the New York case every labelled bus is a New York bus | `tests/nygrid/test_any_case.py::test_the_original_generator_takes_the_new_york_case` | `[ny]` |
| A design family of your own runs through the same source | `tests/test_any_case.py::test_a_family_of_your_own_takes_the_place_of_one_by_name` | nothing |

## Grid, meters and noise

| Claim | Test | Needs |
|---|---|---|
| The admittances of an IEEE case are those of pandapower's `makeYbus`, its port of MATPOWER's, bit for bit in the generator's summation order (all eight cases) | `tests/test_ieee_case.py::test_the_network_of_an_ieee_case_has_makeybus_admittances` | nothing |
| The admittance matrices the loader serves are those the file's static graph gives | `tests/test_formulas.py::test_loader_admittances_match_kernel` | nothing |
| The measurement Jacobian matches finite differences, on 16 sampled state columns | `tests/nygrid/test_sensing.py::test_jacobian_matches_finite_differences` | `[ny]` |
| A meter's error is a constant bias and a per-scan jitter whose squares sum to the square of its accuracy class | `tests/test_formulas.py::test_bias_and_jitter_keep_the_class_total`; `tests/nygrid/test_sensing.py::test_noise` checks it statistically on 4,000 scans | `[ny]` for the second |
| Only a PMU reads an angle | `tests/test_pmu_channels.py::test_a_scada_voltmeter_reads_no_angle_under_the_hybrid_model` | nothing |
| A client's view holds only its own meters, buses and labels | `tests/nygrid/test_dataset.py::test_client_view_holds_only_its_own_meters_buses_and_labels` | `[ny]` |

## Operating points and solving

| Claim | Test | Needs |
|---|---|---|
| The two OPF engines agree on IEEE-14, 30 and 118, within the tolerances [Solving](../concepts/solving.md) gives | `tests/nygrid/test_solve_engines.py::test_the_engines_agree_on_an_ieee_case` | `[ny]`, Ipopt |
| An OPF result does not depend on the solves before it (IEEE-14) | `tests/nygrid/test_solve_engines.py::test_an_opf_does_not_depend_on_the_solves_before_it` | `[ny]`, Ipopt |
| A solve choice the New York pool cannot be built with is refused, with the reason | `tests/nygrid/test_solve_engines.py::test_the_new_york_pool_refuses_what_it_cannot_be_solved_with` | `[ny]` |
| On three hours of 2019 the New York dispatch equals NYgrid's MATLAB output to 0.001 MW | `tests/nygrid/test_scenario.py::test_dispatch_matches_matlab` | `[ny]`, inputs |
| On the same hours the AC projection agrees with it to 5 MW per unit and 0.001 pu | `tests/nygrid/test_scenario.py::test_projection_matches_matlab` | `[ny]`, Ipopt, inputs |
| A harmless edit of the case file reads to the same case, and a breaking one is refused | `tests/nygrid/test_case_file.py::test_a_harmless_edit_reads_to_the_same_case`, `test_an_edit_that_breaks_the_case_is_refused` | `[ny]` |

## The rules of the code

| Claim | Test | Needs |
|---|---|---|
| No `Any` annotation outside a short list with reasons | `tests/test_schema.py::test_every_any_is_listed_with_its_reason` | nothing |
| No input is checked by hand outside the models | `tests/test_schema.py::test_no_input_is_checked_by_hand_outside_the_models` | nothing |
| Every public record is defined in `models/`, except in the New York modules the test lists | `tests/test_models_package.py::test_every_public_model_is_defined_in_the_package` | nothing |
| A wrong kind of value given to a settings or input model is accepted or refused with a `ConfigError` or the model's own named error, never a raw exception | `tests/test_malformed_inputs.py` | nothing |

## What these tests do not show

- **Byte identity across platforms.** The stored digests are compared bit for bit only on the
  platform they were recorded on (`tests/digests.py`). Elsewhere those tests are skipped, and the
  other tests on this page still run.
- **A New York pool build on every platform.** It repeats bit for bit on some platforms and only
  to a tolerance on others ([Repeating a solve](../concepts/solving.md#repeating-a-solve)).
- **The released data files.** A download checks the sha256 of the releases the package lists
  (`src/fdia_graph/download.py`). No test rebuilds a released file.
