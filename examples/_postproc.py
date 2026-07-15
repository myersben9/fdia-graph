#!/usr/bin/env python
"""Post-regeneration: as each system's shard finishes, run BDD on it; after all three, run the residuals.
Polls _regen_release.out for the generator's '[ieeeC] DONE' marker so we never read a half-written shard."""
import os, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "..", "..", "venv", "python.exe")
ENV = dict(os.environ, PYTHONPATH=os.path.join(HERE, "..", "src"), KMP_DUPLICATE_LIB_OK="TRUE")
LOG = os.path.join(HERE, "_regen_release.out")

def done(c):
    try: return f"[ieee{c}] DONE" in open(LOG, errors="ignore").read()
    except FileNotFoundError: return False

for C in (14, 118, 300):
    while not done(C):
        time.sleep(30)
    print(f"[postproc] BDD ieee{C}", flush=True)
    subprocess.run([PY, "_bdd_release.py"], env=dict(ENV, CASE=str(C), N_PER="80"), cwd=HERE)

print("[postproc] residuals (all systems)", flush=True)
subprocess.run([PY, "_residuals_release.py"], env=dict(ENV, PER_FAM="1200"), cwd=HERE)
print("[postproc] ALL DONE", flush=True)
