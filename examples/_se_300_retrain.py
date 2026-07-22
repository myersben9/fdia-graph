import os, subprocess
HERE=os.path.dirname(os.path.abspath(__file__)); PY=os.path.join(HERE,"..","..","venv","python.exe")
ENV=dict(os.environ, PYTHONPATH=os.path.join(HERE,"..","src"), KMP_DUPLICATE_LIB_OK="TRUE", CASE="300", EPOCHS="200")
import shutil
# run PINN + NN at 200 epochs -> overwrite the temp, then we compare
for wp,name in [("0.2","se_300.json"),("0","se_300_nophys.json")]:
    print(f"[retrain300] w_phys={wp} epochs=200", flush=True)
    subprocess.run([PY,"_se_pinn_v040.py"], env=dict(ENV, W_PHYS=wp), cwd=HERE)
    shutil.copy(os.path.join(HERE,"results",name), os.path.join(HERE,"results",name.replace(".json","_ep200.json")))
print("[retrain300] done", flush=True)
