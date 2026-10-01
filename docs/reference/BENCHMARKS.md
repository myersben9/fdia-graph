# Benchmarks

Per-record timings on the tiny IEEE-14 timeline, stored by `python tools/bench.py` as a run of the results store's `bench` experiment (`results/bench.csv`) and rendered below by `tools/results_docs.py`; `--check` fails when a timing is more than 3x slower than the last run. One machine's rows are comparable with each other, not with another machine's.

From 0.18 the generate column is milliseconds per timeline frame of the test suite's tiny timeline (`TIMELINE_KW` in `tests/conftest.py`, 1000 frames); earlier rows timed the record-shard writer per record and are not comparable to it.

From 0.20 that timeline has one-frame Aq and Al episodes, and its seed moved with them, so a later row's generate column times a different episode mix from the rows above it. From 0.21 it holds At and the overload Am only on the hybrid meters (seed 2, the search capped at 16 supports): other frames and another meter plan, so none of its columns, the estimators' included, compares with the rows above. `tools/bench.py` records the fixture recipe with each row, starts a new table for a new recipe, and `--check` compares only rows of the same machine and recipe; the first run on this recipe starts its baseline.

<!-- results: bench.table -->
| date | generate ms/record | wls ms/record | huber ms/record | prior+huber ms/record | version | machine | recipe |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-09-16 | 79.839 | 0.110 | 2.824 | 1.885 | 0.17.0 | AMD64 Intel64 Family 6 Model 143 Stepping 8, GenuineIntel, numpy 1.26.4, torch 2.12.0.dev20260314+cu128 | shards |
<!-- /results -->

## The fewest-tamper search

Median seconds per episode of the overload (`Am`) and ramp (`At`) search, before and after a change,
run side by side on the same fixed episodes (seed 1, hybrid meters, pool ratings, the D16 bounds,
two-line `Am`, budget 256, 20-snapshot windows; experiment `search.speed`). Every episode returned
the same answer before and after.

The search speed-up (#163):

<!-- results: search.speed 163 -->
| system | default threads, Am | default threads, At | one thread, Am | one thread, At |
|---|---:|---:|---:|---:|
| IEEE-14 (10 Am, 4 At) | 4.5 to 3.0 | 0.84 to 0.66 | 4.5 to 2.9 | 0.36 to 0.21 |
| IEEE-30 (5 Am, 3 At) | 8.5 to 4.8 | 1.1 to 0.98 | 8.4 to 5.1 | 0.54 to 0.41 |
| IEEE-118 (4 Am, 3 At) | 14.5 to 9.5 | 1.8 to 0.98 | 5.2 to 3.3 | 1.1 to 0.44 |
<!-- /results -->

BLAS held to one thread during the search (#164):

<!-- results: search.speed 164 -->
| system | default threads, Am | default threads, At | one thread, Am | one thread, At |
|---|---:|---:|---:|---:|
| IEEE-14 (10 Am, 4 At) | 3.1 to 3.1 | 0.58 to 0.28 | 3.0 to 2.7 | 0.31 to 0.31 |
| IEEE-30 (5 Am, 3 At) | 5.5 to 5.6 | 0.81 to 0.33 | 5.5 to 5.1 | 0.32 to 0.26 |
| IEEE-118 (4 Am, 3 At) | 7.2 to 3.1 | 0.89 to 0.44 | 3.3 to 3.3 | 0.38 to 0.51 |
<!-- /results -->
