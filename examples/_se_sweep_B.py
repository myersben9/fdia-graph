#!/usr/bin/env python
"""Part B sweep driver: 300-bus benign-angle floor. Runs configs SEQUENTIALLY (never parallel -> no extra OOM
/ one-system-at-a-time), each writes results/se_300_exp_{TAG}.json. Order = cheapest-diagnostic first."""
import os, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); PY = os.path.join(HERE, "..", "..", "venv", "python.exe")
ENV = dict(os.environ, PYTHONPATH=os.path.join(HERE, "..", "src"), KMP_DUPLICATE_LIB_OK="TRUE", CASE="300", W_PHYS="0.2")
# (TAG, EPOCHS, HID, ARMA_LAYERS, BLOCKS, COSINE)
CONFIGS = [
    ("deep6",   "80",  "128", "6", "2", "1"),  # PRIMARY hyp: receptive field (ARMA num_layers 2->6) -> global angle
    ("ep120",   "120", "128", "2", "2", "1"),  # (i) isolate the training-budget lever (same arch, cosine)
    ("hid256",  "80",  "256", "2", "2", "1"),  # (iii) capacity/width
    ("blk4d4",  "80",  "128", "4", "4", "1"),  # deep+wide receptive field (4 blocks x 4 layers)
]
for tag, ep, hid, al, blk, cos in CONFIGS:
    t0 = time.time()
    print(f"\n===== [{tag}] EPOCHS={ep} HID={hid} ARMA_LAYERS={al} BLOCKS={blk} COSINE={cos} =====", flush=True)
    subprocess.run([PY, "_se_exp_v040.py"], cwd=HERE,
                   env=dict(ENV, TAG=tag, EPOCHS=ep, HID=hid, ARMA_LAYERS=al, BLOCKS=blk, COSINE=cos))
    print(f"[{tag}] done in {(time.time()-t0)/60:.1f} min", flush=True)
print("\n[sweep] all configs done", flush=True)
