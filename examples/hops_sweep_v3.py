"""
#2 + #5 -- hops sweep and attack-magnitude test (ieee14).

Sweep: one 3000-frame timeline per hops (1, 2, 3); the table goes into
docs/reference/DATA_DICTIONARY.md under the hops knob.
Test: passes iff tampered values change by 2-20% from their unattacked value.

Run:  python examples/hops_sweep.py     (or: pytest examples/hops_sweep.py)
"""
import functools
import pathlib
import re
import sys
import tempfile
import time

import h5py
import numpy as np

import fdia_graph as fg

GRID = "ieee14"
N_FRAMES = 3000
HOPS = (1, 2, 3)

LOWER_BOUND = 0.02
UPPER_BOUND = 0.20
EPS = 1e-9  # float slack on the band edges

# Families whose per-meter change the guide bounds to 2-20%:
# Aq=1, Ad=2, As=3, Al=6. At (0.2%/frame ramp), Am (<=1.8%/frame steps) and
# Ar (unbounded replay) are excluded by design.
CHECK_FAMILIES = (1, 2, 3, 6)
P_CH, Q_CH = 1, 2  # node channel order: |V|, P_inj, Q_inj, theta

REPO = pathlib.Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "reference" / "DATA_DICTIONARY.md"
START = "<!-- hops-sweep:start -->"
END = "<!-- hops-sweep:end -->"


# ----------------------------------------------------------------- sweep ---

def _find_attr(f, key):
    """Look for an HDF5 attribute on the root or any group/dataset."""
    if key in f.attrs:
        return f.attrs[key]
    found = []

    def visit(_name, obj):
        if key in obj.attrs:
            found.append(obj.attrs[key])

    f.visititems(visit)
    return found[0] if found else None


def _run_one(hops, out_dir):
    out = out_dir / f"timeline_{GRID}_hops{hops}.h5"
    t0 = time.perf_counter()
    # generate(system, name, frames=None, ...): frames must be passed by keyword,
    # otherwise the count lands in `name` and the full 72,000-frame pool is used.
    ret = fg.generate(GRID, name=f"{GRID}_hops{hops}", frames=N_FRAMES,
                      hops=hops, out=str(out))
    seconds = time.perf_counter() - t0

    path = pathlib.Path(str(ret)) if ret is not None else out
    if not path.is_file():
        path = out
    if not path.is_file():
        raise FileNotFoundError(f"could not find the generated file (returned {ret!r}, out={out})")

    with h5py.File(path, "r") as f:
        y = f["data/y"][:]
        stealthy = f["data/stealthy"][:].astype(bool)
        node_t = f["attack/node_tamper"][:]
        edge_t = f["attack/edge_tamper"][:]
        fallback = _find_attr(f, "fallback_benign")

    T = len(y)
    attacked = y.reshape(T, -1).any(axis=1)
    tampered = node_t.reshape(T, -1).sum(axis=1) + edge_t.reshape(T, -1).sum(axis=1)
    sel = stealthy & attacked

    return {
        "hops": hops,
        "path": path,
        "seconds": seconds,
        "fallback_benign": None if fallback is None else int(np.asarray(fallback).ravel()[0]),
        "attacked_frac": float(attacked.mean()),
        "mean_tampered": float(tampered[sel].mean()) if sel.any() else float("nan"),
    }


@functools.lru_cache(maxsize=1)
def _sweep():
    """Run the three generations once; the test reuses the same files."""
    out_dir = pathlib.Path(tempfile.mkdtemp(prefix="fdia_hops_sweep_"))
    return tuple(_run_one(h, out_dir) for h in HOPS)


# ------------------------------------------------------------------ docs ---

def _table(rows):
    lines = [
        "| hops | generation time (s) | fallback_benign | attacked fraction | mean tampered meters per stealthy frame |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        fb = "n/a" if r["fallback_benign"] is None else r["fallback_benign"]
        lines.append(f"| {r['hops']} | {r['seconds']:.1f} | {fb} | "
                     f"{r['attacked_frac']:.3f} | {r['mean_tampered']:.2f} |")
    return "\n".join(lines)


def _trend(rows):
    first, last = rows[0]["mean_tampered"], rows[-1]["mean_tampered"]
    if np.isnan(first) or np.isnan(last):
        return "The mean number of tampered meters per stealthy frame could not be computed."
    word = "rises" if last > first else "falls" if last < first else "stays flat"
    return (f"On ieee14 with {N_FRAMES} frames, the mean number of tampered meters per stealthy "
            f"frame {word} from {first:.2f} at hops=1 to {last:.2f} at hops=3.")


def write_doc(rows):
    """Put the sweep table in the doc's hops section, replacing what was there."""
    block = f"{START}\n\n{_table(rows)}\n\n{_trend(rows)}\n\n{END}"
    text = DOC.read_text(encoding="utf-8")

    # Drop the block (and its auto-added heading) left by an earlier run.
    text = re.sub(r"(?:^## `hops` sweep\n+)?" + re.escape(START) + r".*?" + re.escape(END) + r"\n*",
                  "", text, flags=re.S | re.M)

    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines) if re.match(r"^#+\s.*hops", ln, re.I)), None)
    if start is None:
        text = text.rstrip("\n") + f"\n\n## Hops knob\n\n{block}\n"
    else:
        level = len(re.match(r"^#+", lines[start]).group(0))
        end = len(lines)
        for j in range(start + 1, len(lines)):
            m = re.match(r"^(#+)\s", lines[j])
            if m and len(m.group(1)) <= level:
                end = j
                break
        lines[start + 1:end] = ["", block, ""]
        text = "\n".join(lines).rstrip("\n") + "\n"
    DOC.write_text(text, encoding="utf-8")


# ------------------------------------------------------------------ test ---

FAMILY_NAMES = {1: "Aq", 2: "Ad", 3: "As", 6: "Al"}
CHANNEL_NAMES = {P_CH: "P", Q_CH: "Q"}


def _check_magnitudes():
    """Return (stats, skipped_zero, violations) over the sweep files.

    stats[(hops, family)] = (values checked, values outside the band)
    """
    violations = []
    stats = {}
    skipped_zero = 0

    for row in _sweep():
        with h5py.File(row["path"], "r") as f:
            x = f["data/node_x"][:]              # observed (attacked)
            benign = f["benign/node_benign"][:]  # same noise draw, attack removed
            tamper = f["attack/node_tamper"][:].astype(bool)
            family = f["data/family"][:]

        if tamper.ndim == 2:  # [T, N] mask -> apply to every channel
            tamper = np.broadcast_to(tamper[..., None], x.shape)

        sel = np.isin(family, CHECK_FAMILIES)[:, None, None] & tamper
        sel[..., [c for c in range(x.shape[-1]) if c not in (P_CH, Q_CH)]] = False

        true_val = benign[sel]
        att_val = x[sel]
        idx = np.argwhere(sel)

        nonzero = true_val != 0
        skipped_zero += int((~nonzero).sum())
        pct = np.abs(att_val - true_val)[nonzero] / np.abs(true_val)[nonzero]
        idx = idx[nonzero]

        bad = (pct < LOWER_BOUND - EPS) | (pct > UPPER_BOUND + EPS)
        fam_at = family[idx[:, 0]]
        for fam in CHECK_FAMILIES:
            m = fam_at == fam
            stats[(row["hops"], fam)] = (int(m.sum()), int((bad & m).sum()))

        for (t, bus, ch), p in zip(idx[bad], pct[bad]):
            violations.append((row["hops"], int(t), int(family[t]), int(bus), int(ch), float(p)))

    return stats, skipped_zero, violations


def test_attack_magnitude_within_2_to_20_percent():
    stats, skipped_zero, violations = _check_magnitudes()
    checked = sum(c for c, _ in stats.values())
    assert checked > 0, "no tampered P/Q values were found to check"
    assert not violations, (
        f"{len(violations)} of {checked} tampered value(s) outside "
        f"[{LOWER_BOUND:.0%}, {UPPER_BOUND:.0%}] (skipped {skipped_zero} with a zero unattacked value)"
    )


# --------------------------------------------------------------- console ---

def _box(headers, body):
    """Plain-text grid table with right-aligned cells."""
    widths = [max(len(h), *(len(r[i]) for r in body)) for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt(cells):
        return "|" + "|".join(f" {c:>{w}} " for c, w in zip(cells, widths)) + "|"

    return "\n".join([sep, fmt(headers), sep, *(fmt(r) for r in body), sep])


def _sweep_console(rows):
    headers = ["hops", "time (s)", "fallback_benign", "attacked frac", "tampered / stealthy frame"]
    body = [[str(r["hops"]),
             f"{r['seconds']:.1f}",
             "n/a" if r["fallback_benign"] is None else str(r["fallback_benign"]),
             f"{r['attacked_frac']:.3f}",
             f"{r['mean_tampered']:.2f}"] for r in rows]
    return _box(headers, body)


def _report_magnitude():
    """Print the magnitude test result; return True on pass."""
    stats, skipped_zero, violations = _check_magnitudes()
    checked = sum(c for c, _ in stats.values())

    print(f"\nChecking attack magnitudes ({LOWER_BOUND:.0%} to {UPPER_BOUND:.0%}, "
          f"{', '.join(FAMILY_NAMES[f] for f in CHECK_FAMILIES)}, P and Q only)...\n")

    headers = ["hops", "family", "values checked", "outside band", "% outside"]
    body = []
    for (hops, fam), (n, nbad) in sorted(stats.items()):
        body.append([str(hops), FAMILY_NAMES[fam], str(n), str(nbad),
                     f"{100 * nbad / n:.1f}%" if n else "-"])
    print(_box(headers, body))

    nbad_total = len(violations)
    print(f"\nTotal: {nbad_total} of {checked} values outside the band "
          f"({skipped_zero} skipped because the unattacked value was 0).")

    if violations:
        print("\nFirst 10 violations:")
        for hops, t, fam, bus, ch, p in violations[:10]:
            print(f"  hops={hops}  frame={t:<5} {FAMILY_NAMES[fam]}  bus={bus:<3} "
                  f"{CHANNEL_NAMES[ch]}  change={100 * p:.2f}%")
        print(f"\nRESULT: FAIL")
        return False

    print("\nRESULT: PASS - all checked tampered values are within 2-20% of their unattacked value.")
    return True


# ------------------------------------------------------------------ main ---

if __name__ == "__main__":
    print(f"\nSweeping hops {', '.join(map(str, HOPS))} on {GRID} ({N_FRAMES} frames each)...", flush=True)
    rows = _sweep()
    print()
    print(_sweep_console(rows))
    print(f"\nTrend: {_trend(rows)}")

    print("\nUpdating DATA_DICTIONARY.md...")
    write_doc(rows)
    print(f"Done: {DOC}")

    sys.exit(0 if _report_magnitude() else 1)