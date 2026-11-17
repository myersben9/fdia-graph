# New York grid test data

Files `tests/nygrid` reads that are checked in, so CI runs the tests on them with only the public inputs
(`tests/nygrid/inputs.py` says which input each test needs).

| file | what it is | made by |
|---|---|---|
| `matlab_ref/` | operating points of three hours from NYgrid's MATLAB code with the AC recipe, which the Python port must match | `pipelines/newyork/matlab/dump_hours.m` |
| `days/pool_2019_v3_days.h5` | the frames of `out/pool_2019_v3.h5` on 2019-06-17 and 2019-07-05 (576 frames), the pool the timeline digests build on (`tests/nygrid/test_timeline_digests.py`) | the slice below |

To cut the slice again:

```python
import numpy as np
from fdia_graph.data.pool import Pool, read_pool, write_pool

p = read_pool("out/pool_2019_v3.h5")
keep = np.isin(p.time.astype("datetime64[D]"), np.array(["2019-06-17", "2019-07-05"], dtype="datetime64[D]"))
write_pool("pool_2019_v3_days.h5", Pool(**{k: getattr(p, k)[keep] for k in Pool.__dataclass_fields__}))
```

A change to the pool rebuilds this file from the same commands and rewrites the timeline digests in the same pull
request.
