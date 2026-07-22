#!/usr/bin/env python
"""Part B best-achievable-floor run: stack the two strongest levers (width 256 + deep 6-layer receptive field)
+ long cosine training, to establish the LOWEST 300 benign angle this architecture family can reach. If this
does not beat the meter (0.082 deg), the negative result is airtight. Runs after the sweep (one at a time)."""
import os, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); PY = os.path.join(HERE, "..", "..", "venv", "python.exe")
ENV = dict(os.environ, PYTHONPATH=os.path.join(HERE, "..", "src"), KMP_DUPLICATE_LIB_OK="TRUE", CASE="300", W_PHYS="0.2")
t0 = time.time()
subprocess.run([PY, "_se_exp_v040.py"], cwd=HERE,
               env=dict(ENV, TAG="kitchensink", EPOCHS="120", HID="256", ARMA_LAYERS="6", BLOCKS="2", COSINE="1"))
print(f"[kitchensink] done in {(time.time()-t0)/60:.1f} min", flush=True)
