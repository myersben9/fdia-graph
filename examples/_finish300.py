import os, json, subprocess
HERE=os.path.dirname(os.path.abspath(__file__)); PY=os.path.join(HERE,"..","..","venv","python.exe")
ENV=dict(os.environ, PYTHONPATH=os.path.join(HERE,"..","src"), KMP_DUPLICATE_LIB_OK="TRUE", PYTHONUNBUFFERED="1")
sh=os.path.join(HERE,"release_v0.4.1","ml_only_ieee300.h5")
for attn,tag in [(True,"hybrid"),(False,"arma")]:
    out=os.path.join(HERE,"results",f"_run_ieee300_{tag}.json")
    cmd=[PY,"train_arma.py","--shard",sh,"--system","ieee300","--epochs","40","--out",out]
    if not attn: cmd.append("--no_attn")
    print(f"[300] {tag} ...",flush=True); subprocess.run(cmd,env=ENV,cwd=HERE,check=True)
# assemble benchmark (14+118 from existing runs + 300) and attn_ab, write marker
bench=[]; attn_ab={}
for C in (14,118,300):
    h=json.load(open(os.path.join(HERE,"results",f"_run_ieee{C}_hybrid.json")))
    a=json.load(open(os.path.join(HERE,"results",f"_run_ieee{C}_arma.json")))
    bench.append(h); attn_ab[f"ieee{C}"]={"arma":{"loc":round(a["overall"]["swf1"],3),"det":a["detection"]["det_f1"]},
                                          "hybrid":{"loc":round(h["overall"]["swf1"],3),"det":h["detection"]["det_f1"]}}
json.dump(bench,open(os.path.join(HERE,"results","ml_only_benchmark.json"),"w"),indent=2)
json.dump(attn_ab,open(os.path.join(HERE,"results","ml_only_attn_ab.json"),"w"),indent=2)
open(os.path.join(HERE,"results","models_v040.done"),"w").write("done")
print("[300] done + benchmark assembled + marker written",flush=True)
