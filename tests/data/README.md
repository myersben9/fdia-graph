# Test data

Files the test suite reads but can no longer generate:

| file | layout | written by |
|---|---|---|
| `tiny_shard_v072.h5` | data release v0.7.2: a record shard with gap rows and the clean pool per pool timestep | the pre-0.18 shard writer |
| `tiny_timeline_v083.h5` | data release v0.8.3: a timeline with all seven families (Aq, Ad, As, Ar, At, Al, the held-redistribution Am) and the v0.8.3 meters; IEEE-14, 400 frames, seed 1, 20-frame ramps | the legacy recipe, `fg.generate(..., families=LEGACY_FAMILIES, am_attack="redistribution", min_tamper=False, redundancy={"meter_model": "v083"})`, at commit 7e4f14d |
| `tiny_stream_v071.npz` | a v0.7.x stream file: the arrays of `stream_of` over that timeline plus the pickled episode list | the same commit, `np.savez_compressed` |

They keep old releases readable (`tests/test_old_releases.py`). The generator makes only At and Am
from 0.21; to rebuild a file, check out the commit named here.
