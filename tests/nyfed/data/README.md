# New York grid test data

Files `tests/nyfed` reads that are checked in, so CI runs the tests on them with only the public inputs
(`tests/nyfed/nyfed_inputs.py` says which input each test needs).

| file | what it is | made by |
|---|---|---|
| `matlab_ref/` | operating points of three hours from NYgrid's MATLAB code with the AC recipe, which the Python port must match | `tools/nyfed/matlab/nygrid/dump_hours.m` |
| `days/am_48220_0.h5`, `days/am_48220_175.h5` | two days of the damage study (2019-06-17): the interface attack at onset 48220, at the limit and 175 MW past it | `production_builder(...).build_episode` on `out/pool_2019_v3.h5` |
| `days/sd_south_2019-07-05.h5` | the single-episode day of 2019-07-05 | `tools/nyfed/build_single_episodes.py` on `out/pool_2019_v3.h5`, copied from `out/single_episode/` |
| `days/loads_2019_v2_days.npz` | the bus reactive loads of `out/loads_2019_v2.npz` (the pool's load series) on the same two days, which the AC operator reads (`pool_frames`, `NYFED_LOADS`) | `np.savez_compressed` of the series' `time` and `q` on those days |
| `days/pool_2019_v3_days.h5` | the frames of `out/pool_2019_v3.h5` on 2019-06-17 and 2019-07-05 (576 frames), the pool `operator_day` reads beside a day | the slice below |

`operator_day` matches the pool to a day by time, so these two days of the pool give the operator and the detection
tools exactly what the whole pool does (the golden snapshots hold on both). To cut the slice again:

```python
import numpy as np
from fdia_graph.generation.nygrid.io import read_pool, write_pool
from fdia_graph.generation.nygrid.pool import Pool

p = read_pool("out/pool_2019_v3.h5")
keep = np.isin(p.time.astype("datetime64[D]"), np.array(["2019-06-17", "2019-07-05"], dtype="datetime64[D]"))
write_pool("pool_2019_v3_days.h5", Pool(**{k: getattr(p, k)[keep] for k in Pool.__dataclass_fields__}))
```

A change to the timeline format or the pool rebuilds these files from the same commands and rewrites the golden
snapshots in the same pull request.
