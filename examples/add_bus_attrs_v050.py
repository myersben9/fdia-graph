"""Add static per-bus attributes to the v0.5.0 shards, in place, without touching any record.

Follows the upgrade_to_v050.py pattern exactly: the new arrays are STATIC per-system values, so no
regeneration is needed and every per-record dataset must stay byte-identical (verified by digest).
The bus ordering question is settled empirically before writing anything: for case14/118/300 the
pandapower bus index, the ppc row order, and our node axis all coincide (net._pd2ppc_lookups['bus']
is the identity on all three), so these arrays index the same buses as data/node_x and graph/bus_*.

What gets added to graph/ (length N, ppc order == pandapower order):
  bus_type         1 PQ, 2 PV, 3 reference, 4 isolated (ppc BUS_TYPE column)
  bus_vmin/vmax    voltage magnitude limits, pu
  bus_base_kv      nominal voltage, kV
  bus_is_zero_inj  1 where the generator's own zero-injection logic marks the bus
  bus_has_gen      1 where a generator (incl. slack ext_grid) sits on the bus
  bus_base_pd/qd   base-case load at the bus, MW / MVAr (summed over pandapower load rows)
  bus_attackable   1 where the generator would consider the bus an attack target, i.e. it carries a
                   load with |p_mw| > 0. This is the IEEE-300 reactive-only-bus rule (buses 141, 183
                   are loads with p=0 and are NOT attackable) made structural instead of a filter
                   buried in the generator.

zero-injection and attackable come from FdiaGenerator itself rather than being re-derived here, so
the shipped arrays cannot drift from the generator's behaviour.

    ./venv/python.exe fdia_graph_sdk/examples/add_bus_attrs_v050.py [--dry-run]
"""
import hashlib
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(HERE, "release_v0.5.0")
DRY = "--dry-run" in sys.argv

FIELDS = ["bus_type", "bus_vmin", "bus_vmax", "bus_base_kv", "bus_is_zero_inj",
          "bus_has_gen", "bus_base_pd", "bus_base_qd", "bus_attackable"]


def bus_attrs(system):
    """Static per-bus attributes for one IEEE case, sourced from the generator itself."""
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    from fdia_graph._core import FdiaGenerator
    import pandapower as pp
    import pandapower.networks as pn

    g = FdiaGenerator(system)                       # canonical zero_inj + attackable logic
    net = {14: pn.case14, 118: pn.case118, 300: pn.case300}[int(system)]()
    pp.runpp(net)
    bus = net._ppc["bus"]
    N = bus.shape[0]

    has_gen = np.zeros(N)
    for b in net.gen["bus"].values: has_gen[int(b)] = 1
    for b in net.ext_grid["bus"].values: has_gen[int(b)] = 1

    pd_ = np.zeros(N); qd = np.zeros(N)
    for r in net.load.itertuples():
        pd_[int(r.bus)] += r.p_mw; qd[int(r.bus)] += r.q_mvar

    zero_inj = np.zeros(N); zero_inj[list(g.zero_inj)] = 1
    attackable = np.zeros(N)
    attackable[net.load["bus"].values[g._attackable_mask]] = 1

    return dict(
        bus_type=bus[:, 1].real.astype(np.float64),
        bus_vmax=bus[:, 11].real.astype(np.float64),
        bus_vmin=bus[:, 12].real.astype(np.float64),
        bus_base_kv=bus[:, 9].real.astype(np.float64),
        bus_is_zero_inj=zero_inj, bus_has_gen=has_gen,
        bus_base_pd=pd_, bus_base_qd=qd, bus_attackable=attackable,
    )


def digest(ds, cap=4_000_000):
    a = ds[:] if ds.size <= cap else ds[: max(1, cap // max(1, int(np.prod(ds.shape[1:] or [1]))))]
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()[:16]


shards = sorted(x for x in os.listdir(DST) if x.endswith(".h5"))
print(f"{len(shards)} shards in {os.path.relpath(DST)} {'(DRY RUN)' if DRY else ''}\n")
cache, ok_all = {}, True

for name in shards:
    p = os.path.join(DST, name)
    with h5py.File(p, "r") as f:
        system = int(f.attrs.get("system", f.attrs.get("N", 0)))
        before = {k: digest(f[f"data/{k}"]) for k in f["data"]}
    if system not in cache:
        cache[system] = bus_attrs(system)
    A = cache[system]

    if DRY:
        print(f"  {name:<34} system {system:<4} would add {len(FIELDS)} arrays "
              f"(attackable={int(A['bus_attackable'].sum())}, zero_inj={int(A['bus_is_zero_inj'].sum())})")
        continue

    with h5py.File(p, "a") as f:
        gg = f["graph"]
        if len(A["bus_type"]) != f.attrs["N"]:
            print(f"  {name}: BUS COUNT MISMATCH, skipped"); ok_all = False; continue
        for k in FIELDS:
            if k in gg: del gg[k]
            gg.create_dataset(k, data=A[k])
        gg.attrs["bus_attr_static"] = ("type(1PQ,2PV,3ref),vmin,vmax,base_kv,is_zero_inj,has_gen,"
                                       "base_pd(MW),base_qd(MVAr),attackable — pandapower==ppc bus order")
        after = {k: digest(f[f"data/{k}"]) for k in f["data"]}

    same = before == after
    ok_all &= same
    print(f"  {name:<34} attackable={int(A['bus_attackable'].sum()):<4} "
          f"zero_inj={int(A['bus_is_zero_inj'].sum()):<4} records unchanged: {same}")

print("\nALL OK" if ok_all else "\nFAILURES ABOVE", flush=True)
sys.exit(0 if ok_all else 1)
