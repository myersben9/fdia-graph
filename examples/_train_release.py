#!/usr/bin/env python
"""Re-train the localizer benchmark + attention A/B on the v0.4.1 shards and assemble the report JSONs.

For each system: train the ARMA hybrid (with attention) and plain ARMA (--no_attn) on the LOCAL v0.4.1 shard
(train_arma.py trains in per-unit by default: --units pu), then assemble:
  ml_only_benchmark.json  <- the attention-hybrid runs (localization per family + detection)
  ml_only_attn_ab.json    <- {ieee{c}: {arma:{loc,det}, hybrid:{loc,det}}}
and touch results/models_v040.done so the report un-gates those two pages. Everything trained on real
v0.4.1 data; nothing hardcoded. Waits for each shard to exist before training it. Env: EPOCHS.
"""
import os, sys, json, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "..", "..", "venv", "python.exe")
SHARDS = os.path.join(HERE, "release_v0.4.1"); RES = os.path.join(HERE, "results")
ENV = dict(os.environ, PYTHONPATH=os.path.join(HERE, "..", "src"), KMP_DUPLICATE_LIB_OK="TRUE")
EPOCHS = os.environ.get("EPOCHS", "40")
SYS = [14, 118, 300]

def train(C, attn):
    tag = "hybrid" if attn else "arma"
    out = os.path.join(RES, f"_run_ieee{C}_{tag}.json")
    shard = os.path.join(SHARDS, f"ml_only_ieee{C}.h5")
    cmd = [PY, "train_arma.py", "--shard", shard,
           "--system", f"ieee{C}", "--epochs", EPOCHS, "--out", out]
    if not attn: cmd.append("--no_attn")
    print(f"[train] ieee{C} {tag}  shard={shard} ...", flush=True)
    subprocess.run(cmd, env=ENV, cwd=HERE, check=True)
    return json.load(open(out))

bench = []; attn_ab = {}
for C in SYS:
    shard = os.path.join(SHARDS, f"ml_only_ieee{C}.h5")
    while not os.path.exists(shard):        # wait for regeneration to produce this shard
        time.sleep(60)
    time.sleep(5)
    h = train(C, attn=True)                 # attention hybrid = the shipped localizer -> benchmark
    a = train(C, attn=False)                # plain ARMA = the ablation baseline
    bench.append(h)
    attn_ab[f"ieee{C}"] = {"arma":   {"loc": round(a["overall"]["swf1"], 3), "det": a["detection"]["det_f1"]},
                           "hybrid": {"loc": round(h["overall"]["swf1"], 3), "det": h["detection"]["det_f1"]}}
    json.dump(bench, open(os.path.join(RES, "ml_only_benchmark.json"), "w"), indent=2)
    json.dump(attn_ab, open(os.path.join(RES, "ml_only_attn_ab.json"), "w"), indent=2)
    print(f"[done] ieee{C}: hybrid swF1 {h['overall']['swf1']:.3f}  det-F1 {h['detection']['det_f1']:.3f}", flush=True)

open(os.path.join(RES, "models_v040.done"), "w").write("done")
print("[all] localizer benchmark + attention A/B trained on v0.4.1 (per-unit); models_v040.done written", flush=True)
