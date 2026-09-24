# Benchmarks

Per-record timings on the tiny IEEE-14 shard, appended by `python tools/bench.py`; `--check` fails when a timing is more than 3x slower than the last row. One machine's rows are comparable with each other, not with another machine's.

From 0.18 the generate column is milliseconds per timeline frame of the test suite's tiny timeline (`frozen_spec.TIMELINE_KW`, 1000 frames, every family); earlier rows timed the record-shard writer per record and are not comparable to it.

From 0.20 that timeline is seed 23 with one-frame Aq episodes (the fixture moved from seed 4 when Aq became single-snapshot), so a later row's generate column times a different episode mix from the rows above it; the estimator columns are per record and stay comparable.

| date | generate ms/record | wls ms/record | huber ms/record | prior+huber ms/record | version | machine |
|---|---|---|---|---|---|---|
| 2026-09-16 | 79.839 | 0.110 | 2.824 | 1.885 | 0.17.0 | AMD64 Intel64 Family 6 Model 143 Stepping 8, GenuineIntel, numpy 1.26.4, torch 2.12.0.dev20260314+cu128 |
