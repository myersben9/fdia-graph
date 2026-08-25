#!/usr/bin/env python
"""60-second tour of fdia-graph: load a dataset, inspect it, and get it in whatever format you want.

Run: python quickstart.py   (downloads the IEEE-14 shard on first use, then caches it)
"""
from __future__ import annotations

import fdia_graph as fg

# 1) LOAD — auto-downloads the newest release, caches under ~/.cache/fdia_graph.
train = fg.load("ieee14", split="train")
test = fg.load("ieee14", split="test")
print("what's in it:", train.summary())          # system, N, E, record count, per-family counts

# 2) TRAIN-READY — a PyTorch DataLoader yielding the measurement graph + per-bus labels.
loader = train.loader(batch_size=32)
batch = next(iter(loader))
print("batch tensors:", {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")})

# 3) SLICE THE BENCHMARK — family subsets and the unseen-attack generalization protocol.
stealthy = fg.load("ieee14", split="test", families=["Ao", "ramp", "LRA"])   # the hard, ML-only attacks
print("stealthy-only test set:", len(stealthy), "records")
heldout = fg.load("ieee14", split="train", heldout=True)                     # As/Ar held out of training
print("held-out train set:", len(heldout), "records")

# 4) ANY FRAMEWORK — whole split as numpy / torch / pandas (tensorflow via .to_tf()).
arrays = test.to_numpy()
print("numpy node_x:", arrays["node_x"].shape)
df = test.to_pandas(flatten_features=False)       # one row per record + metadata
print("pandas head:\n", df.head())

# 5) VERSION CONTROL — pin an exact dataset release for reproducibility.
#    fg.load("ieee14", release="v0.1.0")

# 6) GENERATE A CUSTOM VARIANT (needs: pip install 'fdia-graph[generate]')
#    fg.generate("ieee14", name="my_run", per_family=2000, attack_intensity=0.2, families=["Ao","ramp","LRA"])
#    ds = fg.load("my_run", split="train")
