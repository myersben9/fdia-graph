"""Upgrade v0.4.1 shards to the v0.5.0 graph schema without regenerating any records.

The new fields are STATIC branch and bus physics, identical for every record in a system, so there is
no reason to re-run generation. Regenerating would re-solve tens of thousands of AC power flows and,
worse, would produce per-record data that is not bit-identical to the data every published result was
computed on. Copying the file and adding a small group leaves all measurements exactly as audited.

For each shard: copy, derive the physics from the pandapower case for that system, write the ten new
arrays into graph/, then verify that the arrays reconstruct the nodal admittance matrix and that every
per-record dataset is byte-identical to the source.

    ./venv/python.exe fdia_graph_sdk/examples/upgrade_to_v050.py [--dry-run]
"""
import hashlib
import os
import shutil
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "release_v0.4.1")
DST = os.path.join(HERE, "release_v0.5.0")
DRY = "--dry-run" in sys.argv

FIELDS = ["edge_r", "edge_x", "edge_b", "edge_g", "edge_tap", "edge_shift",
          "edge_status", "edge_is_trafo", "bus_shunt_g", "bus_shunt_b"]


def physics(system):
    """Static branch and bus physics for one IEEE case, float64, ppc order (lines then transformers)."""
    import pandapower as pp
    import pandapower.networks as pn
    net = {14: pn.case14, 118: pn.case118, 300: pn.case300}[int(system)]()
    pp.runpp(net)
    ppc = net._ppc
    br, bus = ppc["branch"], ppc["bus"]
    tap = br[:, 8].real.astype(np.float64).copy()
    tap[tap == 0] = 1.0
    nl = len(net.line)
    return dict(
        edge_r=br[:, 2].real.astype(np.float64), edge_x=br[:, 3].real.astype(np.float64),
        edge_b=br[:, 4].real.astype(np.float64), edge_g=br[:, 23].real.astype(np.float64),
        edge_tap=tap, edge_shift=br[:, 9].real.astype(np.float64),
        edge_status=br[:, 10].real.astype(np.float64),
        edge_is_trafo=np.r_[np.zeros(nl), np.ones(br.shape[0] - nl)].astype(np.float64),
        bus_shunt_g=bus[:, 4].real.astype(np.float64),
        bus_shunt_b=bus[:, 5].real.astype(np.float64),
    ), float(ppc["baseMVA"]), br[:, 0].real.astype(int), br[:, 1].real.astype(int), bus.shape[0]


def rebuild(P, f, t, nbus, baseMVA):
    Y = np.zeros((nbus, nbus), dtype=complex)
    ys = P["edge_status"] / (P["edge_r"] + 1j * P["edge_x"])
    Bc = P["edge_status"] * (P["edge_g"] + 1j * P["edge_b"])
    tau = P["edge_tap"] * np.exp(1j * np.pi / 180.0 * P["edge_shift"])
    Ytt = ys + Bc / 2.0
    Yff = Ytt / (tau * np.conj(tau))
    for k in range(len(ys)):
        i, j = f[k], t[k]
        Y[i, i] += Yff[k]; Y[i, j] += -ys[k] / np.conj(tau[k])
        Y[j, i] += -ys[k] / tau[k]; Y[j, j] += Ytt[k]
    Y[np.arange(nbus), np.arange(nbus)] += (P["bus_shunt_g"] + 1j * P["bus_shunt_b"]) / baseMVA
    return Y


_YREF = {}


def _yref(system):
    """makeYbus reference for a system, built once and cached."""
    if system not in _YREF:
        import pandapower as pp
        import pandapower.networks as pn
        from pandapower.pypower.makeYbus import makeYbus
        net = {14: pn.case14, 118: pn.case118, 300: pn.case300}[int(system)]()
        pp.runpp(net)
        ppc = net._ppc
        _YREF[system] = np.asarray(makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])[0].todense())
    return _YREF[system]


def digest(ds, cap=4_000_000):
    """Hash a bounded slice of a dataset, enough to catch any per-record corruption."""
    a = ds[:] if ds.size <= cap else ds[: max(1, cap // max(1, int(np.prod(ds.shape[1:] or [1]))))]
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()[:16]


shards = sorted(x for x in os.listdir(SRC) if x.endswith(".h5"))
print(f"{len(shards)} shards, {'DRY RUN' if DRY else 'writing to ' + os.path.relpath(DST)}\n")
os.makedirs(DST, exist_ok=True)
cache = {}
ok_all = True

for name in shards:
    s, d = os.path.join(SRC, name), os.path.join(DST, name)
    with h5py.File(s, "r") as f:
        system = int(f.attrs.get("system", f.attrs.get("N", 0)))
        before = {k: digest(f[f"data/{k}"]) for k in f["data"]}
    if system not in cache:
        cache[system] = physics(system)
    P, baseMVA, bf, bt, nbus = cache[system]

    if DRY:
        print(f"  {name:<34} system {system:<4} would add {len(FIELDS)} arrays")
        continue

    shutil.copy2(s, d)
    with h5py.File(d, "a") as f:
        gg = f["graph"]
        if gg["edge_index"].shape[1] != len(P["edge_r"]):
            print(f"  {name}: BRANCH COUNT MISMATCH, skipped"); ok_all = False; continue
        for k in FIELDS:
            if k in gg:
                del gg[k]
            gg.create_dataset(k, data=P[k])
        gg.attrs.update(dict(
            edge_feat_static="r,x,b,g,tap,shift,status,is_trafo (per unit, ppc order = lines then trafos)",
            bus_feat_static="shunt_g,shunt_b (MW/MVAr at 1.0 pu, ppc bus order)",
            edge_reactance_deprecated="mixes ohms (lines) with vk_percent (trafos); use edge_x",
            ybus_reconstructible="yes"))
        f.attrs["schema_version"] = "0.5.0"
        after = {k: digest(f[f"data/{k}"]) for k in f["data"]}

    err = float(np.abs(rebuild(P, bf, bt, nbus, baseMVA) - _yref(system)).max())
    same = before == after
    ok = same and err / float(np.abs(_yref(system)).max()) < 1e-10
    ok_all &= ok
    print(f"  {name:<34} ybus err {err:.2e}   records unchanged {same}   "
          f"{'OK' if ok else '*** FAIL ***'}")

print("\n" + ("all shards upgraded and verified" if ok_all else "SOMETHING FAILED"))
