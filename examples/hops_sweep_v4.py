"""
#2  Attacker hop-radius sweep: hops 1, 2, 3 on ieee14.
#5  Test: Passes if and only if attacks change values by 2 to 20 percent
    from the unattacked value.

Run:  python examples/hops_sweep_v3.py     (or: pytest examples/hops_sweep_v3.py)
"""
import functools
import pathlib
import sys
import tempfile
import time

import h5py
import numpy as np

import fdia_graph as fg

GRID = "ieee14"
N_FRAMES = 3000
HOPS = (1, 2, 3)


def _box(headers, body):
    """Plain-text grid table with right-aligned cells (used for the printed output)."""
    widths = [max(len(h), *(len(r[i]) for r in body)) for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt(cells):
        return "|" + "|".join(f" {c:>{w}} " for c, w in zip(cells, widths)) + "|"

    return "\n".join([sep, fmt(headers), sep, *(fmt(r) for r in body), sep])

# -------------------------- 2 (hops 1, 2, 3 on ieee14)

def _find_attr(f, key):
    """Find an HDF5 attribute on the root or on any group/dataset."""
    if key in f.attrs:
        return f.attrs[key]
    found = []
    f.visititems(lambda _name, obj: found.append(obj.attrs[key]) if key in obj.attrs else None)
    return found[0] if found else None


def generate_small_timeline(hops, out_dir):
    """Generate one small ieee14 timeline for this hops value and record its characteristics."""
    out = out_dir / f"timeline_{GRID}_hops{hops}.h5"

    # --------------------- timeline generation 
    t0 = time.perf_counter()
    ret = fg.generate(GRID, name=f"{GRID}_hops{hops}", frames=N_FRAMES, hops=hops, out=str(out))
    generation_time = time.perf_counter() - t0          # (1) generation time

    path = pathlib.Path(str(ret)) if ret is not None else out
    if not path.is_file():
        path = out
    if not path.is_file():
        raise FileNotFoundError(f"could not find the generated file (returned {ret!r}, out={out})")

    with h5py.File(path, "r") as f:
        y = f["data/y"][:]                               # [T, N] 1 on each attacked bus
        stealthy = f["data/stealthy"][:].astype(bool)    # [T]    1 for Aq/At/Al/Am frames
        node_tamper = f["attack/node_tamper"][:]         # 1 where the attacker wrote a meter
        edge_tamper = f["attack/edge_tamper"][:]
        fallback = _find_attr(f, "fallback_benign")

    T = len(y)
    attacked = y.reshape(T, -1).any(axis=1)              # frame has at least one attacked bus

    #     (a file attribute written by the generator).
    fallback_benign = None if fallback is None else int(np.asarray(fallback).ravel()[0])

    # (3) attacked fraction: share of frames that carry an attack.
    attacked_fraction = float(attacked.mean())

    # (4) mean tampered meters per stealthy frame: tampered meters (node + edge)
    #     in each attacked stealthy frame, averaged over those frames.
    tampered_per_frame = node_tamper.reshape(T, -1).sum(axis=1) + edge_tamper.reshape(T, -1).sum(axis=1)
    stealthy_attacked = stealthy & attacked
    mean_tampered = (float(tampered_per_frame[stealthy_attacked].mean())
                     if stealthy_attacked.any() else float("nan"))

    return {
        "hops": hops,
        "path": path,
        "generation_time": generation_time,
        "fallback_benign": fallback_benign,
        "attacked_fraction": attacked_fraction,
        "mean_tampered": mean_tampered,
    }


@functools.lru_cache(maxsize=1)
def run_sweep():
    """Generate the three timelines once (the #5 test below reuses the same files)."""
    out_dir = pathlib.Path(tempfile.mkdtemp(prefix="fdia_hops_sweep_"))
    return tuple(generate_small_timeline(h, out_dir) for h in HOPS)


def print_sweep(rows):
    headers = ["hops", "time (s)", "fallback_benign", "attacked fraction", "tampered / stealthy frame"]
    body = [[str(r["hops"]),
             f"{r['generation_time']:.1f}",
             "n/a" if r["fallback_benign"] is None else str(r["fallback_benign"]),
             f"{r['attacked_fraction']:.3f}",
             f"{r['mean_tampered']:.2f}"] for r in rows]
    print(_box(headers, body))

    first, last = rows[0]["mean_tampered"], rows[-1]["mean_tampered"]
    word = "rises" if last > first else "falls" if last < first else "stays flat"
    print(f"Trend: mean tampered meters per stealthy frame {word} "
          f"from {first:.2f} at hops=1 to {last:.2f} at hops=3.")

# Test: attacks change values by 2 to 20 percent from the unattacked value
# Value must be within threadhold for tampered P and Q values to pass 

LOWER_BOUND = 0.02
UPPER_BOUND = 0.20
EPS = 1e-9  # float slack on the band edges

# Families whose per-meter change is bounded to 2-20 
CHECK_FAMILIES = (1, 2, 3, 6)
FAMILY_NAMES = {1: "Aq", 2: "Ad", 3: "As", 6: "Al"}
P_CH, Q_CH = 1, 2  # node channel order: |V|, P_inj, Q_inj, theta
CHANNEL_NAMES = {P_CH: "P", Q_CH: "Q"}


def check_magnitudes():
    """Compare every tampered value with its unattacked value.

    Returns (stats, skipped_zero, violations), where
    stats[(hops, family)] = (values checked, values outside the 2-20% band).
    """
    stats, violations, skipped_zero = {}, [], 0

    for row in run_sweep():
        with h5py.File(row["path"], "r") as f:
            attacked_val = f["data/node_x"][:]               # observed (attacked)
            unattacked_val = f["benign/node_benign"][:]      # same noise draw, attack removed
            tamper = f["attack/node_tamper"][:].astype(bool)
            family = f["data/family"][:]

        if tamper.ndim == 2:  # [T, N] mask -> apply to every channel
            tamper = np.broadcast_to(tamper[..., None], attacked_val.shape)

        # Tampered P and Q values in the families under test.
        sel = np.isin(family, CHECK_FAMILIES)[:, None, None] & tamper
        sel[..., [c for c in range(attacked_val.shape[-1]) if c not in (P_CH, Q_CH)]] = False
        idx = np.argwhere(sel)
        true_v, att_v = unattacked_val[sel], attacked_val[sel]

        nonzero = true_v != 0                                # percent change is undefined at 0
        skipped_zero += int((~nonzero).sum())
        idx, true_v, att_v = idx[nonzero], true_v[nonzero], att_v[nonzero]

        pct_change = np.abs(att_v - true_v) / np.abs(true_v)
        outside = (pct_change < LOWER_BOUND - EPS) | (pct_change > UPPER_BOUND + EPS)

        fam_at = family[idx[:, 0]]
        for fam in CHECK_FAMILIES:
            m = fam_at == fam
            stats[(row["hops"], fam)] = (int(m.sum()), int((outside & m).sum()))
        for (t, bus, ch), p in zip(idx[outside], pct_change[outside]):
            violations.append((row["hops"], int(t), int(family[t]), int(bus), int(ch), float(p)))

    return stats, skipped_zero, violations


def test_attack_magnitude_within_2_to_20_percent():
    """PASS if and only if every tampered value is within 2-20% of its unattacked value."""
    stats, skipped_zero, violations = check_magnitudes()
    checked = sum(n for n, _ in stats.values())
    assert checked > 0, "no tampered P/Q values were found to check"
    assert not violations, (
        f"{len(violations)} of {checked} tampered value(s) outside "
        f"[{LOWER_BOUND:.0%}, {UPPER_BOUND:.0%}] (skipped {skipped_zero} with a zero unattacked value)"
    )


def print_test():
    """Print the test result; return True on pass."""
    stats, skipped_zero, violations = check_magnitudes()
    checked = sum(n for n, _ in stats.values())

    headers = ["hops", "family", "values checked", "outside 2-20%", "% outside"]
    body = [[str(hops), FAMILY_NAMES[fam], str(n), str(nbad), f"{100 * nbad / n:.1f}%" if n else "-"]
            for (hops, fam), (n, nbad) in sorted(stats.items())]
    print(_box(headers, body))
    print(f"\nTotal: {len(violations)} of {checked} values outside the band "
          f"({skipped_zero} skipped because the unattacked value was 0).")

    if violations:
        print("\nFirst 10 violations:")
        for hops, t, fam, bus, ch, p in violations[:10]:
            print(f"  hops={hops}  frame={t:<5} {FAMILY_NAMES[fam]}  bus={bus:<3} "
                  f"{CHANNEL_NAMES[ch]}  change={100 * p:.2f}%")
        print("\nResult: Fail")
        return False

    print("\nResult: Pass - every checked tampered value is within 2-20% of its unattacked value.")
    return True

# ----------------------------------- MAIN

if __name__ == "__main__":
    print(f"\nAttacker reach sweep: hops {', '.join(map(str, HOPS))} "
          f"on {GRID} ({N_FRAMES} frames each)...", flush=True)
    rows = run_sweep()
    print()
    print_sweep(rows)

    print(f"\n\nTest attack magnitude: tampered values must change "
          f"{LOWER_BOUND:.0%} to {UPPER_BOUND:.0%} from the unattacked value "
          f"({', '.join(FAMILY_NAMES[f] for f in CHECK_FAMILIES)}, P and Q only)...\n")
    sys.exit(0 if print_test() else 1)