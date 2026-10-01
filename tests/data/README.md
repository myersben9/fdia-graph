# Test data

Files the test suite reads but can no longer generate. Each is either written by the SDK that built
its data release or a slice of the published file, so the loader is tested on the layouts users
download (`tests/test_old_releases.py`).

| file | data release and layout | made by |
|---|---|---|
| `tiny_shard_v072.h5` | v0.7.2: a record shard with gap rows and the clean pool per pool timestep | the pre-0.18 shard writer |
| `tiny_stream_v071.npz` | v0.7.1: the first 100 frames of the published `stream_ieee118.npz` (the frame arrays and the episodes that end inside them) | sliced from the release asset |
| `graph_ieee118_v071.npz` | v0.7.1: the published graph sidecar of IEEE-118 (`edge_index`, `edge_attr`, `node_m`, `edge_m`), which the stream files of v0.7.1 and v0.7.2 lack | the release asset, unchanged |
| `tiny_timeline_v080.h5` | v0.8.0: a timeline, all seven families; IEEE-14, 400 frames, seed 17, 20-frame ramps | fdia-graph 0.18.0 (tag `v0.18.0`), `fg.generate` |
| `tiny_timeline_v081.h5` | v0.8.1: as above, seed 15 | fdia-graph 0.19.0 (tag `v0.19.0`), `fg.generate` |
| `tiny_timeline_v083.h5` | v0.8.3: a timeline, all seven families (the held-redistribution Am), the v0.8.3 meters; IEEE-14, 400 frames, seed 1, 20-frame ramps | the v0.8.3 recipe at commit 7e4f14d: `fg.generate(..., families=LEGACY_FAMILIES, am_attack="redistribution", min_tamper=False, redundancy={"meter_model": "v083"})` |

The generator makes only At and Am from 0.21; to rebuild a file, check out the tag or commit named
here and run `fg.generate("ieee14", ..., frames=400, ramp_len=20, seed=<seed>)`.
