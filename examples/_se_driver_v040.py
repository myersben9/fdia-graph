#!/usr/bin/env python
"""Fill every missing v0.4.0 SE result: PINN (w_phys=0.2) + NN (w_phys=0) + WLS, for C in {14,118,300}.
Skips any results/se_*.json that already exists. Sequential so it never OOMs. Seed 123 inside each script."""
import os, sys, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
PY = os.path.join(HERE, "..", "..", "venv", "python.exe")
ENV = dict(os.environ, PYTHONPATH=os.path.join(HERE, "..", "src"), KMP_DUPLICATE_LIB_OK="TRUE", EPOCHS=os.environ.get("EPOCHS", "60"))

def need(fn): return not os.path.exists(os.path.join(RES, fn))

for C in (14, 118, 300):
    if need(f"se_{C}.json"):
        print(f"[se] ieee{C} PINN (w_phys=0.2)", flush=True)
        subprocess.run([PY, "_se_pinn_v040.py"], env=dict(ENV, CASE=str(C), W_PHYS="0.2"), cwd=HERE)
    if need(f"se_{C}_nophys.json"):
        print(f"[se] ieee{C} NN (w_phys=0)", flush=True)
        subprocess.run([PY, "_se_pinn_v040.py"], env=dict(ENV, CASE=str(C), W_PHYS="0"), cwd=HERE)
    if need(f"se_{C}_wls.json"):
        print(f"[se] ieee{C} WLS", flush=True)
        subprocess.run([PY, "_se_wls_v040.py"], env=dict(ENV, CASE=str(C)), cwd=HERE)
    print(f"[done] ieee{C} SE complete", flush=True)
print("[all] SE (PINN/NN/WLS x 3 systems) complete", flush=True)
