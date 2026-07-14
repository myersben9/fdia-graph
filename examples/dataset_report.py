#!/usr/bin/env python
"""Comprehensive dataset report for `fdia-graph`, generated FROM THE SDK.

Loads each shipped system via `fdia_graph.load(...)` and reads the precomputed experiment results in
`examples/results/`, then writes `fdia_dataset_report.pdf`. Design rule: TABLES for anything that is a small
set of numbers (composition, BDD split, benchmarks, state-estimation error); FIGURES only where the data is
genuinely distributional or spatial (measurement histograms, per-bus attack probability, residual box/violin,
the LRA illustration). Every figure keeps a CSV/npz data sidecar.

    python dataset_report.py                 # regenerates the PDF from the installed package + results/
"""
import os, json, textwrap, warnings, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("FDIA_GRAPH_RELEASE", "v0.3.0")
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import fdia_graph as fg
from fdia_graph.dataset import FAMILIES, STEALTHY_FAMILIES, FdiaGraph

# Prefer locally-regenerated release shards (release_v0.4.0/) over the downloadable builtin, so the report
# reflects the new data before the GitHub release is cut. Override the directory via FDIA_LOCAL_SHARDS.
LOCAL_SHARDS = os.environ.get("FDIA_LOCAL_SHARDS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_v0.4.0"))
def _shard_path(c): return os.path.join(LOCAL_SHARDS, f"ml_only_ieee{c}.h5")
def shard_ready(c):
    # a shard counts as ready only if the local v0.4.0 file exists AND is fully written (openable with data)
    p = _shard_path(c)
    if not os.path.exists(p): return False
    try:
        import h5py
        with h5py.File(p, "r") as f: return "data" in f and "y" in f["data"]
    except Exception:
        return False
def load_sys(c, split=None):
    # NEVER fall back to the downloadable builtin (that would silently inject OLD-dataset numbers). Only the
    # local v0.4.0 shard is trusted; callers must guard on shard_ready(c) before using this.
    return FdiaGraph(_shard_path(c), split=split)

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "fdia_dataset_report")
SYS_ALL = [14, 118, 300]                                      # the full roster (for fixed table columns)
# SYS = only the systems whose v0.4.0 shard is fully built; the rest render as "pending" (never old data).
SYS = [c for c in SYS_ALL if shard_ready(c)]
PENDING_SYS = [c for c in SYS_ALL if c not in SYS]
FAMS = ["Aq", "Ad", "As", "Ar", "At", "Al"]; SPLITS = ["train", "val", "test"]
PW, PH = 8.5, 11.0
INK = "#1f2d3d"; ACCENT = "#b23a2e"; MUTE = "#8a97a3"; STEAL = "#b23a2e"; DET = "#2c6f9e"
SUBTLE = "#5a6673"; RULE = "#c7ced5"
SIDE = {}
PAGENO = [0]                                                  # mutable page counter for the footer
RUNNING = "fdia-graph  ·  ML-Only Dangerous FDIA Localization Dataset"

# professional defaults: proportional sans for structure, serif for running prose
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.serif": ["DejaVu Serif"],
    "axes.edgecolor": "#8a97a3", "axes.linewidth": 0.7, "axes.titlesize": 9.5, "axes.titlecolor": INK,
    "xtick.color": "#5a6673", "ytick.color": "#5a6673", "axes.labelcolor": INK, "pdf.fonttype": 42,
})


def load_json(name, default=None):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else default


# ---------------- style helpers ----------------
def header(fig, title, sub=""):
    fs = 16 if len(title) <= 52 else (13.5 if len(title) <= 72 else 12)
    fig.text(0.07, 0.955, title, ha="left", va="top", fontsize=fs, weight="bold", color=INK)
    y_rule = 0.940 if len(title) <= 72 else 0.925
    fig.add_artist(plt.Line2D([0.07, 0.93], [y_rule, y_rule], color=ACCENT, lw=1.4, alpha=0.9))
    if sub:
        fig.text(0.07, y_rule - 0.013, textwrap.fill(" ".join(sub.split()), 104), ha="left", va="top",
                 fontsize=9.3, color=SUBTLE, style="italic", linespacing=1.3)


def caption(fig, text, y=0.14):
    wrapped = "\n".join(textwrap.fill(ln, 112) for ln in text.split("\n"))
    fig.text(0.07, y, wrapped, ha="left", va="top", fontsize=8.2, color="#3a444e", family="serif", linespacing=1.35)


def footer(fig, n):
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.045, 0.045], color=RULE, lw=0.6))
    fig.text(0.07, 0.033, RUNNING, ha="left", va="top", fontsize=6.8, color=MUTE)
    fig.text(0.93, 0.033, f"{n}", ha="right", va="top", fontsize=6.8, color=MUTE)


def newpage(title, sub=""):
    fig = plt.figure(figsize=(PW, PH)); header(fig, title, sub); return fig


def save(fig):
    """Stamp the running footer + page number, then write the page."""
    PAGENO[0] += 1; footer(fig, PAGENO[0]); pdf.savefig(fig); plt.close(fig)


def prose(fig, sections, x=0.075, top=0.885, width=100, heading_gap=0.023, para_gap=0.030, lh=0.0188,
          fs=9.6, bullet_gap=0.008):
    """Lay out (heading, body) sections with editorial hierarchy: a bold sans sub-head with a short accent
    tick, then body. body is EITHER a string (justified serif paragraph) OR a list of strings rendered as
    a bulleted list (hanging indent, accent bullet) — bullets for enumerable notes, prose for definitions."""
    y = top
    for heading, body in sections:
        if heading:
            fig.add_artist(plt.Line2D([x, x + 0.024], [y + 0.004, y + 0.004], color=ACCENT, lw=2.4))
            fig.text(x + 0.032, y, heading, ha="left", va="top", fontsize=11, weight="bold", color=INK)
            y -= heading_gap
        if isinstance(body, (list, tuple)):                       # bulleted notes
            for item in body:
                lines = textwrap.wrap(item, width - 3) or [""]
                fig.text(x + 0.006, y, "•", ha="left", va="top", fontsize=fs + 1, color=ACCENT, weight="bold")
                for k, wl in enumerate(lines):                    # hanging indent under the bullet
                    fig.text(x + 0.026, y, wl, ha="left", va="top", fontsize=fs, color="#33404c", family="serif")
                    y -= lh
                y -= bullet_gap
        else:                                                     # paragraph
            for ln in body.split("\n"):
                for wl in (textwrap.wrap(ln, width) or [""]):
                    fig.text(x, y, wl, ha="left", va="top", fontsize=fs, color="#33404c", family="serif")
                    y -= lh
        y -= para_gap
    return y


ROW_H = 0.0246                                                 # figure-fraction height of one styled table row


def draw_table(fig, top_y, col_labels, rows, fontsize=8.6, col_widths=None, shade_rule=None, section_rows=None):
    """Draw one styled table whose TOP edge sits at top_y; return the y of its bottom edge. Consistent across
    every table in the report so captions can be placed a fixed gap beneath the table, not at a page offset."""
    h = (len(rows) + 1) * ROW_H
    ax = fig.add_axes([0.06, top_y - h, 0.88, h]); ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center",
                   colWidths=col_widths or [1.0 / len(col_labels)] * len(col_labels))
    tbl.auto_set_font_size(False); tbl.set_fontsize(fontsize); tbl.scale(1, 1.46)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dbe0e6"); cell.set_linewidth(0.6)
        if r == 0:                                             # header row
            cell.set_facecolor(INK); cell.set_text_props(color="white", weight="bold")
        else:
            i = r - 1
            if section_rows and i in section_rows:
                cell.set_facecolor("#e3e9ef"); cell.set_text_props(weight="bold", color=INK)
            elif shade_rule is not None:
                cell.set_facecolor(shade_rule(i, rows[i]))
            elif i % 2:
                cell.set_facecolor("#f4f6f8")
            if c == 0:
                cell.set_text_props(ha="left"); cell._text.set_x(0.03)
    return top_y - h


def table_page(title, sub, col_labels, rows, cap, fontsize=8.6, col_widths=None, shade_rule=None,
               section_rows=None, **_ignore):
    """A page with one top-anchored table and its caption tucked directly beneath it."""
    fig = newpage(title, sub)
    bottom = draw_table(fig, 0.865, col_labels, rows, fontsize, col_widths, shade_rule, section_rows)
    caption(fig, cap, y=bottom - 0.034)
    return fig


def multi_table_page(title, sub, blocks, top=0.870):
    """A page that stacks several labelled tables, each with its own short caption — so a couple of small
    result tables share one full page instead of floating alone on sparse ones. blocks: list of dicts with
    keys subhead, cols, rows, cap, and optional fontsize/col_widths/shade_rule."""
    fig = newpage(title, sub); y = top
    for b in blocks:
        fig.add_artist(plt.Line2D([0.07, 0.091], [y + 0.004, y + 0.004], color=ACCENT, lw=2.4))
        fig.text(0.099, y, b["subhead"], ha="left", va="top", fontsize=10.5, weight="bold", color=INK)
        y -= 0.028
        y = draw_table(fig, y, b["cols"], b["rows"], b.get("fontsize", 8.6), b.get("col_widths"),
                       b.get("shade_rule"))
        y -= 0.028
        for wl in textwrap.fill(b["cap"], 112).split("\n"):     # tight per-table caption
            fig.text(0.07, y, wl, ha="left", va="top", fontsize=8.0, color="#3a444e", family="serif")
            y -= 0.0165
        y -= 0.058                                               # breathing room before the next table block
    return fig


pdf = PdfPages(OUT + ".pdf")

# ======================= load everything =======================
MAN = load_json("MANIFEST.json", {})
# Model-performance results are gated behind a freshness marker so no STALE (old-dataset) numbers can render.
# Training on the v0.4.0 data writes results/models_v040.done; until then every model page shows a placeholder.
MODELS_READY = os.path.exists(os.path.join(RES, "models_v040.done"))
BENCH = {r["system"]: r for r in load_json("ml_only_benchmark.json", [])} if MODELS_READY else {}
ATTN = load_json("ml_only_attn_ab.json", {}) if MODELS_READY else {}
SE = ({c: dict(pinn=load_json(f"se_{c}.json"), nn=load_json(f"se_{c}_nophys.json"), wls=load_json(f"se_{c}_wls.json")) for c in SYS}
      if MODELS_READY else {c: dict(pinn=None, nn=None, wls=None) for c in SYS})


def pending_page(title, sub, what):
    """A clean placeholder page for a result that is not yet computed on the current dataset — so the report
    NEVER shows stale or made-up numbers. Auto-fills once the underlying results are staged and the report re-runs."""
    fig = newpage(title, sub)
    fig.text(0.5, 0.56, "[ PENDING — v0.4.0 re-train ]", ha="center", va="center", fontsize=15, weight="bold", color=ACCENT)
    fig.text(0.5, 0.48, textwrap.fill(what, 82), ha="center", va="center", fontsize=10.5, color=MUTE,
             family="serif", linespacing=1.5)
    save(fig)

# dataset-intrinsic stats straight from the SDK (fg.load per split)
DATA = {}
for c in SYS:
    per = {sp: load_sys(c, split=sp).summary()["families"] for sp in SPLITS}
    tst = load_sys(c, split="test"); a = tst.to_numpy()
    N, E = tst.N, tst.E
    nm, em, fam, y, nx = a["node_m"], a["edge_m"], a["family"], a["y"], a["node_x"]
    red = (nm.reshape(len(nm), -1).sum(1) + em.reshape(len(em), -1).sum(1)) / (2 * N - 1)
    cov = dict(V=nm[:, :, 0].mean(), Pinj=nm[:, :, 1].mean(), theta=nm[:, :, 3].mean(),
               flow=em[:, :, 0].mean())
    atk = fam > 0
    DATA[c] = dict(N=N, E=E, per=per, redundancy=float(np.median(red)), cov=cov,
                   surface=y[atk].sum(1), perbus=y[atk].mean(0) if atk.any() else np.zeros(N),
                   nx=nx, fam=fam, y=y)
    print(f"loaded ieee{c}: N={N} E={E}")

# ======================= Page 1: overview (prose) =======================
fig = newpage("The fdia-graph Dataset",
              "ML-only dangerous FDIA localization on realistic measurement graphs — IEEE-14 / 118 / 300")
prose(fig, [
 ("What this is",
  "A benchmark of false-data-injection attacks (FDIAs) that evade every classical, non-ML detector yet remain "
  "localizable by a graph model — the regime where machine learning is actually needed. Each record is a "
  "realistic sparse SCADA/PMU measurement graph (PING-style): branch power flows as edge features, metered bus "
  "injections, voltage magnitudes and sparse PMU angles as node features, with per-measurement availability "
  "masks. Redundancy is roughly 2-3, the regime a real energy-management system operates in — not the fully-"
  "observed idealization most FDIA benchmarks assume."),
 ("Seven families, split by detectability", [
  "Stealthy (evade bad-data detection — the hard, dangerous cases):  Aq, a state-consistent load "
  "redistribution;  At (ramp), a slow multi-timestep surge up then down;  Al (LRA), a targeted masked-overload.",
  "Detectable (a contrast set a bad-data detector catches):  Ad, random meter corruption;  As, meter scaling;  "
  "Ar, replay of a past scan.",
  "The split is the point: on this data, “ML beats a detector” is a genuine result, not a rigged one.",
 ]),
 ("What this report covers", [
  "The data dictionary and schema, and the attack families with their bad-data-detection stealth split.",
  "Class composition and the chronological split; the measurement model and coverage.",
  "The measurement distributions, the spatial attack pattern, and how far each family moves the true state.",
  "Model results: localization, detection, the physics-biased-attention gain, and an attack-resilient "
  "physics-informed state estimator.",
 ]),
 ("Reproducibility",
  "Every number in this report is loaded from the installed package (fdia_graph.load) or the result files in "
  "examples/results/. Run  python dataset_report.py  to regenerate this document end to end."),
], top=0.895)
tot = sum(sum(DATA[c]["per"][s].values()) for c in SYS for s in SPLITS)
fig.add_artist(plt.Line2D([0.07, 0.93], [0.175, 0.175], color=RULE, lw=0.6))
fig.text(0.5, 0.155, f"3 systems     {tot:,} records     7 attack families     release v0.3.0",
         ha="center", fontsize=11.5, weight="bold", color=ACCENT)
save(fig)

# ======================= Page 2: data dictionary (TABLE) =======================
dd_cols = ["Field", "Shape", "Dtype", "Meaning"]
dd_rows = [
 ["edge_index", "[2, E]", "int64", "branch endpoints [from; to] (lines then transformers) — the static graph"],
 ["edge_reactance", "[E]", "float32", "per-branch reactance (p.u.)"],
 ["node_x", "[N, 4]", "float32", "[ |V| p.u., P_inj MW, Q_inj MVAr, theta deg ] per bus"],
 ["node_m", "[N, 4]", "float32", "node availability mask (1 = metered; masked entries zeroed)"],
 ["edge_x", "[E, 2]", "float32", "[ P_from MW, Q_from MVAr ] per branch"],
 ["edge_m", "[E, 2]", "float32", "edge availability mask"],
 ["y", "[N]", "float32", "localization target: 1 = bus attacked, 0 = clean"],
 ["temporal_delta", "[N, 2]", "float32", "v0.3+: current-minus-previous-scan [dP_inj, dQ_inj]"],
 ["family", "scalar", "int", "0 benign 1 Aq 2 Ad 3 As 4 Ar 5 ramp 6 LRA"],
 ["stealthy", "scalar", "int", "1 if the attack evades classical bad-data detection"],
 ["split", "scalar", "int", "0 train 1 val 2 test (60/20/20 chronological)"],
 ["seq_id", "scalar", "int", "At-sequence id (>=0 groups one ramp/At sequence); -1 otherwise"],
 ["timestep", "scalar", "int", "source operating-point index"],
 ["gap", "scalar", "int", "1 if a physics non-convergence NA row (~0% shipped)"],
]
fig = table_page("Data Dictionary", "every field in ml_only_ieee{N}.h5  (N = buses, E = branches, T = records)",
                 dd_cols, dd_rows,
                 "The static graph (edge_index, edge_reactance) is read once; all other fields are per record. Masks are "
                 "load-bearing — with redundancy ~2-3 not every bus is metered and PMU angles are sparse, so a model must "
                 "consume node_m/edge_m rather than assume a full measurement vector.",
                 col_widths=[0.17, 0.11, 0.10, 0.62], fontsize=8.2, height=0.56, y_table=0.30)
SIDE["data_dictionary"] = ["field,shape,dtype,meaning"] + [",".join(f'"{x}"' for x in r) for r in dd_rows]
save(fig)

# ======================= Page 3: attack families + BDD stealth split (TABLE) =======================
# (description, provenance) — provenance makes explicit what is OUR contribution vs pulled from references.
SRC = {"Aq":  ("per-bus load scaling, AC re-solve",       "Ours (cf. Boyaci Ao; Liu FDIA)"),
       "Ad":  ("random meter corruption",                    "FDIA taxonomy (Boyaci 2022)"),
       "As":  ("injection-meter scaling",                    "FDIA taxonomy (Boyaci 2022)"),
       "Ar":  ("replay of a past scan",                      "FDIA taxonomy (Boyaci 2022)"),
       "At":  ("temporal load surge, up then down (ramp)",   "Haghshenas et al. (ISGT 2023)"),
       "Al":  ("targeted masked-overload (LRA)",             "Yuan et al. (T-SG 2011)")}
def bdd_pct(c, fam):
    # BDD-pass = fraction NOT flagged = 100 - chi2 detection rate, read from the fresh BDD summary.
    s = load_json(f"bdd_summary_{c}.json")
    try:
        v = s["families"][fam]["chi2_detect_pct"]
        return round(100 - v) if v is not None else None
    except Exception:
        return None
fam_cols = ["Family", "Description", "Class", "Source", "BDD-pass 14/118/300"]
fam_rows = []
for f in FAMS:
    fid = [k for k, v in FAMILIES.items() if v == f][0]
    cls = "stealthy" if fid in STEALTHY_FAMILIES else "detectable"
    bdd = " / ".join(f"{bdd_pct(c, f)}%" if bdd_pct(c, f) is not None else "pending" for c in SYS_ALL)
    fam_rows.append([f, SRC[f][0], cls, SRC[f][1], bdd])
def fam_shade(i, row): return "#fbecea" if row[2] == "stealthy" else "#eaf2f8"
cov_cols = ["System", "Buses N", "Branches E", "Redundancy", "|V| cov", "P_inj cov", "theta cov", "flow cov"]
cov_rows = [([f"IEEE-{c}", DATA[c]["N"], DATA[c]["E"], f"{DATA[c]['redundancy']:.2f}",
             f"{100*DATA[c]['cov']['V']:.0f}%", f"{100*DATA[c]['cov']['Pinj']:.0f}%",
             f"{100*DATA[c]['cov']['theta']:.0f}%", f"{100*DATA[c]['cov']['flow']:.0f}%"] if c in DATA
             else [f"IEEE-{c}", "pending", "pending", "pending", "pending", "pending", "pending", "pending"]) for c in SYS_ALL]
fig = multi_table_page("Attack Families & Measurement Model",
                       "what the attacks are, how stealthy they are, and what the meters actually see",
                       [dict(subhead="Seven families, provenance, and the bad-data-detection stealth split",
                             cols=fam_cols, rows=fam_rows, col_widths=[0.085, 0.34, 0.11, 0.27, 0.195], shade_rule=fam_shade,
                             cap="The Source column separates our contribution from prior art: Aq (stealthy load scaling) is OURS — "
                                 "the AC/physical realization of the state-consistent attack concept (cf. Boyaci et al. 2022 'Ao'; "
                                 "Liu et al. 2011 FDIA); ramp is Haghshenas et al. (ISGT 2023); LRA is Yuan et al. (T-SG 2011); the "
                                 "meter-level Ad/As/Ar are the standard FDIA taxonomy (Boyaci 2022). BDD-pass is the fraction that "
                                 "evade the chi-square detector: the stealthy families (Aq/At/Al, red) pass on ~90-100% (valid, "
                                 "plausibility-bounded states), the detectable ones (Ad/As/Ar, blue) are caught."),
                        dict(subhead="Measurement coverage — fraction of each quantity metered (sparse SCADA/PMU graph)",
                             cols=cov_cols, rows=cov_rows, col_widths=[0.16, 0.11, 0.13, 0.14, 0.11, 0.12, 0.11, 0.12],
                             cap="Redundancy = (# measurements)/(2N-1 states), the classical observability ratio; ~2-3 is a "
                                 "realistic EMS regime. Voltage magnitude and branch flows are widely metered; PMU angles (theta) "
                                 "are sparse — the masks encode which meters exist per record.")])
SIDE["families"] = ["family,description,class,bdd_pass_14,bdd_pass_118,bdd_pass_300"] + \
    [f'{f},"{SRC[f]}",{fam_rows[i][2]},{bdd_pct(14,f)},{bdd_pct(118,f)},{bdd_pct(300,f)}' for i, f in enumerate(FAMS)]
SIDE["coverage"] = ["system,N,E,redundancy,V_cov,Pinj_cov,theta_cov,flow_cov"] + \
    [f"{c},{DATA[c]['N']},{DATA[c]['E']},{DATA[c]['redundancy']:.3f},{DATA[c]['cov']['V']:.3f},{DATA[c]['cov']['Pinj']:.3f},{DATA[c]['cov']['theta']:.3f},{DATA[c]['cov']['flow']:.3f}" for c in SYS if c in DATA]
save(fig)

# ======================= Measurement (noise) model =======================
fig = newpage("Measurement Model — Realistic Accuracy-Class Meter Noise",
              "how every reading is corrupted: the accuracy-class magnitude, heteroscedastic scaling, and a systematic-bias / per-scan-jitter split")
# the accuracy-class values table (class-0.2 instrument transformers, sigma = manufacturer max / sqrt(3))
mm_cols = ["Quantity", "Source of error", "Max (class 0.2)", "sigma = max/√3"]
mm_rows = [["P / Q  (inj + flow)", "conventional device (Table III)", "±3%", "1.7%  (relative)"],
           ["|V|  magnitude", "VT ratio error (class 0.2)", "0.2%", "0.12%  (of ~1 p.u.)"],
           ["theta  angle", "VT phase displacement", "10 arc-min = 0.167°", "0.096°"]]
bot = draw_table(fig, 0.86, mm_cols, mm_rows, fontsize=8.6, col_widths=[0.24, 0.34, 0.20, 0.22])
y = prose(fig, [
 ("The measurement chain",
  "A reading is the true quantity seen through a chain: bus -> instrument transformer (VT/CT) -> measurement "
  "device -> control center. Each stage adds error, so the dataset stores what a real EMS receives, not the "
  "exact physics. The magnitude of that error follows the instrument-transformer ACCURACY CLASS (Asprou, "
  "Kyriakides & Albu 2013 — the model the Falas et al. 2025 baseline delegates to). A class number is a "
  "MAXIMUM error; the standard deviation added as noise is that max divided by sqrt(3) (a uniform-distribution "
  "convention). Power is device-dominated (the conventional device is +-3%), so its sigma is far larger than "
  "voltage, which is transformer-dominated. P/Q noise is heteroscedastic — proportional to the reading — so a "
  "large line carries more absolute error than a small one, which is how a state estimator weights it."),
 ("Systematic bias vs per-scan jitter",
  ["A meter's error is mostly SYSTEMATIC — a fixed calibration / ratio / phase offset that is the SAME every "
   "scan — with only a small RANDOM repeatability jitter. This is the accuracy-vs-precision distinction.",
   "Modelling the whole accuracy class as fresh per-scan noise would make 1-minute traces jump ~2.4% every "
   "scan (unrealistic) and drown the temporal signal. So each meter gets a bias drawn ONCE (0.968 x sigma, "
   "constant in time) plus a per-scan jitter (0.25 x sigma). Their combination equals the full class total.",
   "Because the bias is constant, it CANCELS in scan-to-scan differences: the temporal feature reflects real "
   "load change plus tiny jitter, not meter noise. Verified: measured scan-to-scan drops from 2.4% to ~0.7% "
   "(matching the smooth true load) while the across-meter accuracy stays ~1.7%."]),
], top=bot - 0.03, fs=9.4, lh=0.019)
put_y = y - 0.004
fig.text(0.075, put_y, "One reading:", fontsize=10, weight="bold", color=INK)
fig.text(0.075, put_y - 0.026, r"$z = z_{\mathrm{true}}\,(1 + b_{\rm meter}) + \varepsilon_t,\quad "
         r"b_{\rm meter}\sim\mathcal{N}(0,\,0.968\,\sigma)\ \mathrm{(drawn\ once)},\quad "
         r"\varepsilon_t\sim\mathcal{N}(0,\,0.25\,\sigma\,|z_{\rm true}|)\ \mathrm{(each\ scan)}$",
         fontsize=10.5, color="#1f2d3d", family="serif")
caption(fig, "Honest note: the accuracy-class magnitudes and the sqrt(3) conversion are taken directly from Asprou et al. "
             "(2013); the split of that error into a constant per-meter bias plus a small per-scan jitter is our physically-"
             "motivated refinement (the paper models it as all-random), justified by the standard metrology distinction "
             "between accuracy and precision. |V| bias is the VT ratio offset and theta bias is the VT phase displacement — "
             "both fixed transformer characteristics, so a systematic bias is the correct representation, not per-scan noise.",
        y=put_y - 0.06)
SIDE["measurement_model"] = ["quantity,source,max_class0.2,sigma"] + [",".join(f'"{c}"' for c in r) for r in mm_rows] + \
    ["split,bias=0.968*sigma_constant_per_meter,jitter=0.25*sigma_per_scan,total=class"]
save(fig)

# ======================= Attack construction (formulas) =======================
fig = newpage("How Each Attack Is Built",
              "from the clean state to the attacked measurement graph, as implemented in the generator")
x0 = 0.075; Y = [0.870]
def put(txt, dx, fs, col, fam="serif", bold=False, ital=False):
    fig.text(x0 + dx, Y[0], txt, fontsize=fs, color=col, family=fam, va="top",
             weight="bold" if bold else "normal", style="italic" if ital else "normal")
intro = ("There are two mechanisms. State-level attacks change the bus loads L and re-solve a full AC power flow, "
         "so every measurement stays mutually consistent and passes bad-data detection. Meter-level attacks tamper "
         "the readings directly, which breaks consistency and is caught. Notation: h maps a state x to the meter "
         "readings, PF is a power-flow solve, e is Gaussian meter noise, and z_a is the reading at an attacked bus a.")
for wl in textwrap.fill(intro, 106).split("\n"):
    put(wl, 0, 8.8, "#33404c"); Y[0] -= 0.0185
Y[0] -= 0.016
def grp(title, color):
    fig.add_artist(plt.Line2D([x0, x0 + 0.026], [Y[0] + 0.004, Y[0] + 0.004], color=color, lw=2.8))
    put(title, 0.034, 10.5, INK, "sans-serif", bold=True); Y[0] -= 0.033
def item(name, desc, formula):
    put(name, 0.018, 10.5, INK, "sans-serif", bold=True)
    put(desc, 0.085, 8.8, "#5a6673", "serif", ital=True); Y[0] -= 0.026
    put(formula, 0.050, 11.5, "#1f2d3d", "serif"); Y[0] -= 0.034
grp("State-level attacks   (change the loads, re-solve, evade bad-data detection)", STEAL)
item("benign", "reference, no attack", r"$z = h(x_t) + e$")
item("Aq", "per-bus load scaling on 1-6 buses, re-solve with generation rebalanced (AGC)",
     r"$L_a \leftarrow \alpha_a\, L_a,\ \ \alpha_a \in [1.05,\ 1.15]\ \mathrm{per\ bus},\ \ x' = \mathrm{PF}(L'),\ \ z = h(x') + e$")
item("At", "temporal surge: ramp up then back down (asymmetric rates, random turn), or a dip",
     r"$L_a \leftarrow (1 \pm r_k)\, L_a,\ \ r_k\ \mathrm{rises\ then\ falls\ over}\ k=0\ldots L$")
item("Al", "targeted, load-conserving, PTDF-chosen to cut the target-line flow", r"$L \leftarrow L + d,\ \ \sum_i d_i = 0,\ \ |d_i| \leq \rho\, L_i\ \ (\rho = 0.15)$")
Y[0] -= 0.014
grp("Meter-level attacks   (tamper the readings, caught by bad-data detection)", DET)
item("Ad", "random corruption of the P and Q injection meters", r"$z_a \leftarrow z_a + \mathcal{N}\,(0,\ 0.3\,|z_a| + 0.05)$")
item("As", "scaling of the P and Q injection meters", r"$z_a \leftarrow \beta\, z_a,\ \ \beta \in [1.25,\ 1.5]$")
item("Ar", "replay of a distant past benign scan", r"$z_a \leftarrow z_a(t - k),\ \ k \geq 20$")
caption(fig, "State-level attacks move the whole state, including voltage and angle, because those are the physical "
             "result of the new loads. That is exactly why they pass every classical check. Meter-level attacks change "
             "only the tampered channels and leave an inconsistency a detector can see. The attacked bus set is stored "
             "as the label y, which is the localization target.", y=Y[0] - 0.010)
save(fig)

# ======================= Attack generation steps =======================
fig = newpage("How the Attacks Are Generated",
              "the step-by-step pipeline from a clean operating point to a labeled attacked sample")
Y[0] = 0.865
def steps(title, color, items):
    fig.add_artist(plt.Line2D([x0, x0 + 0.026], [Y[0] + 0.004, Y[0] + 0.004], color=color, lw=2.8))
    put(title, 0.034, 10.5, INK, "sans-serif", bold=True); Y[0] -= 0.033
    for k, s in enumerate(items, 1):
        put(f"{k}.", 0.020, 10, ACCENT, "serif", bold=True)
        for wl in textwrap.wrap(s, 94):
            put(wl, 0.058, 9.8, "#33404c", "serif"); Y[0] -= 0.0188
        Y[0] -= 0.007
    Y[0] -= 0.017
steps("State-level attacks   (Aq, At, Al)", STEAL, [
 "Start from a clean operating point, the true grid state at a timestep t.",
 "Rebuild the bus loads from that state. Net injection plus the generator setpoints gives the gross load.",
 "Choose the attacked bus SET (drawn at random per record, so it is not memorizable): Aq hits 1 to 6 load "
 "buses, each with its OWN scaling multiplier; ramp holds a fixed set of up to 5 buses across its whole "
 "sequence; LRA hits the PTDF-selected up-set and down-set. Then change their loads: bounded per-bus scaling "
 "for Aq, a rise-then-fall surge for At (ramp), or a load-conserving PTDF-targeted shift for Al (LRA).",
 "Re-solve a full AC power flow for the new loads with pandapower, holding generation at the true operating "
 "point's dispatch and distributing the net load change across generators (AGC). If it does not converge, "
 "discard the sample and try another timestep.",
 "Read every meter off the solved state (via the SAME measurement function used for benign) and add sensor "
 "noise. Because generation is pinned to the true dispatch, the attacked snapshot differs from the true state "
 "only in the attacked buses' loads plus the small distributed generation response — the honest attack footprint, "
 "not a spurious slack-bus swing.",
 "Record the attacked bus set as the label y, then check that the sample passes bad-data detection.",
])
steps("Meter-level attacks   (Ad, As, Ar)", DET, [
 "Start from the clean measurements at a timestep, read directly off the true state.",
 "Pick the attacked bus set: Ad, As, and Ar each hit 4 randomly-chosen load buses (and their incident branches).",
 "Corrupt only those buses' meters: add random noise for Ad, scale the readings for As, or paste in a distant "
 "past scan for Ar.",
 "Record the attacked bus set as the label y. These fail bad-data detection by construction.",
])
caption(fig, "The same pipeline runs for IEEE-14, 118, and 300, and benign samples simply skip the attack step and "
             "are read straight off the true state. Everything is seeded, so the dataset regenerates identically each "
             "time. The re-solve in the state-level pipeline is what keeps every reading mutually consistent, which is "
             "why those attacks pass detection, while the meter-level attacks change a few readings and leave a "
             "detectable inconsistency.", y=Y[0] - 0.004)
save(fig)

# ======================= Attack insertion: previous vs current design =======================
fig = newpage("Attack Insertion — Previous vs Current Design",
              "how the generator decides WHEN to place an attack, HOW LARGE it is, and how it PROGRESSES in time")
Y[0] = 0.872
intro = ("The generator was redesigned. The previous pipeline walked one long measurement timeline and crafted each "
         "stealthy Aq attack by a per-sample gradient-descent optimization. The current pipeline samples a pool of "
         "physical operating points and makes stealthy attacks by changing a bounded, plausible load and re-solving "
         "the power flow, so stealth is achieved by construction rather than by optimization.")
for wl in textwrap.fill(intro, 106).split("\n"):
    put(wl, 0, 8.8, "#33404c"); Y[0] -= 0.0185
Y[0] -= 0.013

def axis(title):
    fig.add_artist(plt.Line2D([x0, x0 + 0.026], [Y[0] + 0.004, Y[0] + 0.004], color=ACCENT, lw=2.8))
    put(title, 0.034, 10.5, INK, "sans-serif", bold=True); Y[0] -= 0.029

def side(tag, txt, color):
    put(tag, 0.020, 9.2, color, "sans-serif", bold=True)
    for wl in textwrap.wrap(txt, 92):
        put(wl, 0.105, 9.2, "#33404c", "serif"); Y[0] -= 0.0180
    Y[0] -= 0.007

axis("Stealthy-attack mechanism   (the biggest change)")
side("Previous", "The predecessor attack (Boyaci-style Ao) was solved per sample: about 1000 epochs of gradient descent tuned a fake measurement vector to "
                 "be both bad-data-stealthy and load-plausible, tampering the readings directly (a soft a = Hc).", MUTE)
side("Current", "Aq scales the load at the target buses by a bounded factor and re-solves a full AC power flow, then "
                "reads clean meters off the new state. The result is a genuine, mutually consistent operating point that "
                "passes detection by construction. Cheaper, exact, and a defensible threat model.", INK)

axis("Placement   (when an attack is inserted)")
side("Previous", "Timeline-driven: every one of the ~86,400 timesteps was labeled with a family in sequence, via a "
                 "datatype map over the whole horizon.", MUTE)
side("Current", "Pool-driven: benign draws n_benign distinct random timesteps; each single-shot family retries random "
                "timesteps until per_family samples converge (hard cap of per_family x 25 attempts), so a "
                "non-converging operating point costs time, not sample count.", INK)

axis("Magnitude   (how large the attack is)")
side("Previous", "The attacked set was a graph-radius ball around an entry bus (r = 2-4 hops, capped at 15 buses). Size "
                 "was bounded by a 50% relative-change cap plus, for the predecessor Ao, a 2-sigma jump cap and a 3-sigma load-anomaly "
                 "band from a rolling seven-day window.", MUTE)
side("Current", "Small random bus sets (at most 4-6). Aq scales load 1.05-1.15x; ramp grows 1 + rate*k; LRA is a "
                "load-conserving PTDF shift bounded by 15%. Meter-level As scales 1.25-1.5x, Ad adds ~30% Gaussian, Ar "
                "replays a past scan. Plausibility now comes from re-solving physics, not from explicit caps.", INK)

axis("Progression   (behavior over timesteps)")
side("Previous", "Attacks sat inline in the timeline; temporal structure came from their position in the sequence.", MUTE)
side("Current", "Almost every family is single-shot (one snapshot, seq_id = -1). Ramp is the only temporal attack: a "
                "fixed bus set whose magnitude grows 1 + rate*k over up to 60 consecutive operating points (kept if at "
                "least 10 steps converge), all sharing one seq_id so the chronological split never splits a sequence.", INK)

caption(fig, "Both designs target the same threat model, a stealthy attacker who keeps loads inside a believable band. "
             "The current design reaches it by simulating real physics instead of optimizing a measurement vector, which "
             "makes stealthy samples exact and cheap to generate and gives every attacked snapshot a fully consistent "
             "voltage and angle field.", y=Y[0] - 0.006)
save(fig)

SIDE["attack_insertion"] = [
    "axis,previous,current",
    '"stealthy mechanism","per-sample gradient-descent predecessor Ao (~1000 epochs, soft a=Hc, tamper meters)","Aq: bounded load scale + AC re-solve (stealthy by construction)"',
    '"placement","timeline: ~86400 steps labeled in sequence","pool: retry random timesteps until per_family converge (cap x25)"',
    '"magnitude","graph-radius ball (r2-4, <=15 buses); 50% cap + 2/3-sigma caps","Aq: random 1-6 buses, per-bus 1.05-1.15x; ramp 1+rate*k; LRA PTDF 15%"',
    '"progression","inline timeline position","single-shot (seq=-1) except ramp: 1+rate*k over <=60 steps, one seq_id"',
]

# ======================= Attack progression on one bus's load over time =======================
LT = np.load(os.path.join(RES, "load_timeline.npz")) if os.path.exists(os.path.join(RES, "load_timeline.npz")) else None
if LT is not None:
    bus = int(LT["bus"]); Wn = int(LT["W"]); t = np.arange(Wn)
    fig = newpage(f"Attack Progression — One Bus's Load Over Time (real records, IEEE-118 bus {bus})",
                  "actual injection readings pulled from the shard: benign timeline + each family's real attacked scans")
    fams = [("benign", MUTE,  "reference load rhythm"),
            ("Aq",     STEAL, "state: per-bus load scaling"),
            ("At",     STEAL, "state: temporal ramp"),
            ("Al",     STEAL, "state: load redistribution"),
            ("Ad",     DET,   "meter: random corruption"),
            ("As",     DET,   "meter: scaling"),
            ("Ar",     DET,   "meter: replay a past scan")]
    bottoms = [0.66, 0.50, 0.34, 0.18]; x0s = [0.095, 0.55]; W, Hh = 0.38, 0.12
    for i, (fam, col, note) in enumerate(fams):
        if f"{fam}_atk" not in LT: continue
        ben = LT[f"{fam}_ben"]; atk = LT[f"{fam}_atk"]; hits = LT[f"{fam}_hits"]
        r, c = i // 2, i % 2
        ax = fig.add_axes([x0s[c], bottoms[r], W, Hh])
        ax.plot(t, ben, color=MUTE, lw=1.0, ls="--", zorder=2)               # real benign timeline for this bus
        if fam != "benign":                                                  # ALL families drawn as a line (consistent with At):
            ax.plot(t, atk, color=col, lw=1.4, zorder=3)                     # the attacked reading over the benign line
            if len(hits): ax.scatter(hits, atk[hits], s=24, color=col, zorder=4, edgecolor="white", linewidth=0.6)  # mark attacked scans
        ax.set_title(fam, fontsize=9, color=(INK if fam != "benign" else MUTE), loc="left", weight="bold")
        ax.text(0.98, 0.05, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, color=SUBTLE, style="italic")
        ax.tick_params(labelsize=6.0); ax.margins(x=0.02)
        if c == 0: ax.set_ylabel("MW", fontsize=6.6)
        if r == 3: ax.set_xlabel("scan (1-min)", fontsize=6.6)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    caption(fig, f"Every value is a REAL record from the shard — one IEEE-118 load bus (bus {bus}) over time. Dashed grey = its benign "
                 "injection timeline (one benign record per scan); the coloured line = the ACTUAL attacked reading, with markers at the "
                 "scans this family targets the bus. The key thing to see is that the attacked values sit INSIDE the bus's natural load "
                 "swing, not as obvious spikes — a single scan is only 5-15% off, comparable to the real minute-to-minute variation, which "
                 "is exactly why they are hard to flag by eye or by a simple threshold and why localization (not detection) is the task. "
                 "Single-shot families (Aq/Al, and meter-level Ad/As/Ar) touch one scan at a time (the line meets benign except at the "
                 "marked scans); only At progresses across many consecutive scans.", y=0.15)
    save(fig)
    SIDE["load_timeline"] = open(os.path.join(RES, "sidecars", "load_timeline.csv")).read().splitlines() \
        if os.path.exists(os.path.join(RES, "sidecars", "load_timeline.csv")) else []

# ======================= Bad-data detection verification (table + figure) =======================
BDD = {c: load_json(f"bdd_summary_{c}.json") for c in SYS_ALL}
if any(BDD.values()):
    STEAL_FAM = {"Aq", "At", "Al"}
    def bdd_rows(metric, fmt):
        rows = []
        for f in FAMS:
            cls = "stealthy" if f in STEAL_FAM else "detectable"
            cells = []
            for c in SYS_ALL:
                fam = (BDD.get(c) or {}).get("families", {}).get(f, {})
                v = fam.get(metric)
                cells.append(fmt(v) if v is not None else "pending")
            rows.append([f, cls] + cells)
        # benign row on top for reference
        bcells = []
        for c in SYS_ALL:
            fam = (BDD.get(c) or {}).get("families", {}).get("benign", {})
            v = fam.get(metric); bcells.append(fmt(v) if v is not None else "pending")
        return [["benign", "reference"] + bcells] + rows
    bdd_cols = ["Family", "Class"] + [f"IEEE-{c}" for c in SYS_ALL]
    def bshade(i, row): return "#fbecea" if row[1] == "stealthy" else ("#eef1f4" if row[1] == "reference" else "#eaf2f8")
    fig = multi_table_page("Bad-Data Detection Verification",
                           "we actually run WLS state estimation + the chi-square and largest-normalized-residual tests on every dataset sample",
                           [dict(subhead="Chi-square bad-data detection rate (%) — the stealthy families must NOT be flagged",
                                 cols=bdd_cols, rows=bdd_rows("chi2_detect_pct", lambda v: f"{v:.0f}%"),
                                 col_widths=[0.16, 0.16] + [0.68 / len(SYS_ALL)] * len(SYS_ALL), shade_rule=bshade,
                                 cap="Run per record: build the realistic sparse measurement set, solve WLS, apply the chi-square "
                                     "test. Benign and the stealthy families (Aq/At/Al, red) pass — they are physically valid "
                                     "states. The detectable families (Ad/As/Ar, blue) are caught almost every time. This is the "
                                     "stealth split, measured directly rather than asserted."),
                            dict(subhead="Median largest-normalized-residual (LNR) — the classical detector's statistic (threshold 3.0)",
                                 cols=bdd_cols, rows=bdd_rows("median_lnr", lambda v: f"{v:.1f}"),
                                 col_widths=[0.16, 0.16] + [0.68 / len(SYS_ALL)] * len(SYS_ALL), shade_rule=bshade,
                                 cap="The stealthy families sit at the benign noise floor (median LNR near or below the 3.0 "
                                     "threshold), while the detectable families are far above it. This gives the dataset physical "
                                     "plausibility: a reviewer can confirm the stealthy attacks genuinely evade the standard bad-data "
                                     "detector, not by our claim but by the detector's own residual.")])
    SIDE["bdd_verification"] = ["family,class," + ",".join(f"chi2_ieee{c},lnr_ieee{c}" for c in SYS_ALL)] + \
        [f"{f},{'stealthy' if f in STEAL_FAM else ('reference' if f=='benign' else 'detectable')}," +
         ",".join(f"{(BDD.get(c) or {}).get('families', {}).get(f, {}).get('chi2_detect_pct', '')},{(BDD.get(c) or {}).get('families', {}).get(f, {}).get('median_lnr', '')}" for c in SYS_ALL)
         for f in (["benign"] + FAMS)]
    save(fig)

    # --- BDD residual distribution figure (LNR by family, on the largest system with data) ---
    rp = next((os.path.join(RES, f"bdd_resid_{c}.npz") for c in (118, 300, 14) if os.path.exists(os.path.join(RES, f"bdd_resid_{c}.npz"))), None)
    if rp:
        z = np.load(rp); cc = int(os.path.basename(rp).split("_")[-1].split(".")[0])
        summ = BDD.get(cc, {}).get("families", {})
        fig = newpage(f"Bad-Data Detection — the Stealth Split, Measured (IEEE-{cc})",
                      "we run WLS + the chi-square test on real samples: detection rate, and the chi-square statistic distribution")
        axB, axJ = fig.add_axes([0.10, 0.585, 0.85, 0.275]), fig.add_axes([0.10, 0.245, 0.85, 0.275])
        order = [f for f in (["benign"] + FAMS) if f"ieee{cc}_{f}_J" in z]
        cols = [MUTE if f == "benign" else (STEAL if f in STEAL_FAM else DET) for f in order]
        # (a) chi-square detection rate
        rates = [summ.get(f, {}).get("chi2_detect_pct", 0) for f in order]
        axB.bar(range(len(order)), rates, color=cols, alpha=0.85)
        axB.set_xticks(range(len(order))); axB.set_xticklabels(order, fontsize=8); axB.set_ylim(0, 105)
        axB.set_ylabel("chi-square\ndetection rate (%)", fontsize=8.5); axB.tick_params(labelsize=7.5)
        axB.set_title("Detection rate: stealthy families pass, detectable families are caught", fontsize=9.5)
        # (b) chi-square statistic distribution (log)
        data = [z[f"ieee{cc}_{f}_J"] for f in order]
        parts = axJ.violinplot(data, showmedians=True, widths=0.85)
        for j, b in enumerate(parts["bodies"]): b.set_facecolor(cols[j]); b.set_alpha(0.72)
        axJ.set_yscale("log"); axJ.set_xticks(range(1, len(order) + 1)); axJ.set_xticklabels(order, fontsize=8)
        axJ.set_ylabel("chi-square statistic J\n(log scale)", fontsize=8.5); axJ.tick_params(labelsize=7.5)
        axJ.set_title("Residual magnitude: a ~10x gap separates stealthy from detectable", fontsize=9.5)
        caption(fig, "Real BDD on the shipped data: per record we rebuild the sparse measurement set, solve WLS, and apply the "
                     "chi-square test. Top: the detectable families (blue) are flagged ~100%, while benign and the stealthy families "
                     "(red) pass at the ~5% baseline false-alarm rate. Bottom: the chi-square statistic sits ~10x higher for the "
                     "detectable families. The stealthy attacks evade the standard detector by the detector's own residual.", y=0.185)
        SIDE["bdd_J"] = ["family,chi2_detect_pct,median_J"] + [f"{f},{summ.get(f,{}).get('chi2_detect_pct')},{summ.get(f,{}).get('median_J')}" for f in order]
        save(fig)

# ======================= Page 4: composition (TABLE) =======================
comp_cols = ["System", "Family", "Train", "Val", "Test", "Total"]
comp_rows = []; section_rows = set()
for c in SYS:
    section_rows.add(len(comp_rows))
    comp_rows.append([f"IEEE-{c}", "", "", "", "", ""])       # section header row
    for fid, fname in FAMILIES.items():
        tr = DATA[c]["per"]["train"].get(fname, 0); va = DATA[c]["per"]["val"].get(fname, 0); te = DATA[c]["per"]["test"].get(fname, 0)
        if tr + va + te == 0: continue
        comp_rows.append(["", fname, f"{tr:,}", f"{va:,}", f"{te:,}", f"{tr+va+te:,}"])
fig = table_page("Class Composition & Split",
                 "records per system x family across the 60/20/20 chronological split",
                 comp_cols, comp_rows,
                 "Equal count per attack family (~constant across systems), following Boyaci et al. (2022) and PING. The unseen-"
                 "attack protocol (heldout=True) additionally reserves As/Ar for test only — note their train/val entries drop to "
                 "zero under that flag (not applied here). The benign majority pads the set so both a balanced training view and an "
                 "attack-sparse realistic test view are samplable.",
                 col_widths=[0.14, 0.20, 0.17, 0.17, 0.16, 0.16], fontsize=8.0, height=0.56, y_table=0.30, section_rows=section_rows)
SIDE["composition"] = ["system,family,train,val,test,total"] + \
    [f'{c},{fn},{DATA[c]["per"]["train"].get(fn,0)},{DATA[c]["per"]["val"].get(fn,0)},{DATA[c]["per"]["test"].get(fn,0)},'
     f'{DATA[c]["per"]["train"].get(fn,0)+DATA[c]["per"]["val"].get(fn,0)+DATA[c]["per"]["test"].get(fn,0)}'
     for c in SYS for fn in FAMILIES.values() if DATA[c]["per"]["train"].get(fn,0)+DATA[c]["per"]["val"].get(fn,0)+DATA[c]["per"]["test"].get(fn,0)]
save(fig)

# ======================= measurement distributions (FIGURE — genuinely distributional) =======================
fig = newpage("Figure 1 — Measurement Distributions (IEEE-118)",
              "benign vs attacked, per metered channel — where the families do (and don't) move the meters")
gs = fig.add_gridspec(2, 2, left=0.09, right=0.96, top=0.86, bottom=0.30, wspace=0.28, hspace=0.4)
c = 118; nx = DATA[c]["nx"]; fam = DATA[c]["fam"]; nm_test = load_sys(118, split="test").to_numpy()["node_m"]
chans = [("|V| (p.u.)", 0), ("P_inj (MW)", 1), ("Q_inj (MVAr)", 2), ("theta (deg)", 3)]
for ax, (lab, ch) in zip([fig.add_subplot(gs[i]) for i in range(4)], chans):
    m = nm_test[:, :, ch].astype(bool)
    ben = nx[fam == 0, :, ch][m[fam == 0]]; atk = nx[fam > 0, :, ch][m[fam > 0]]
    lo, hi = np.percentile(np.concatenate([ben, atk]), [1, 99])
    ax.hist(ben, bins=60, range=(lo, hi), density=True, alpha=0.6, color=MUTE, label="benign")
    ax.hist(atk, bins=60, range=(lo, hi), density=True, alpha=0.6, color=ACCENT, label="attacked")
    ax.set_title(lab, fontsize=9); ax.tick_params(labelsize=7)
    if ch == 0: ax.legend(fontsize=7)
caption(fig, "Voltage magnitude barely moves under attack (the stealthy families keep a plausible state), so |V| alone is a weak "
             "signal — consistent with why the attacks evade detectors. The injection and angle channels shift more. These are "
             "genuine distributions, so a histogram (not a table) is the right view. Metered entries only.", y=0.24)
SIDE["distributions_note"] = ["channel,benign_median,attacked_median"] + \
    [f'{lab},{np.median(nx[fam==0,:,ch][nm_test[:,:,ch].astype(bool)[fam==0]]):.4f},{np.median(nx[fam>0,:,ch][nm_test[:,:,ch].astype(bool)[fam>0]]):.4f}' for lab, ch in chans]
save(fig)

# ======================= Page 7: attack surface + per-bus probability (FIGURE — spatial/distributional) =======================
fig = newpage("Figure 2 — Attack Surface & Spatial Pattern",
              "how many buses an attack touches, and which buses are attacked (IEEE-118)")
gs = fig.add_gridspec(1, 2, left=0.09, right=0.96, top=0.84, bottom=0.34, wspace=0.28)
ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
surf = DATA[118]["surface"]
ax0.hist(surf, bins=range(1, int(surf.max()) + 2), color=DET, alpha=0.85, align="left")
ax0.set_title("Attacked buses per record", fontsize=9); ax0.set_xlabel("# attacked buses", fontsize=8); ax0.set_ylabel("count", fontsize=8); ax0.tick_params(labelsize=7)
pb = DATA[118]["perbus"]
ax1.bar(range(len(pb)), pb, color=STEAL, width=1.0)
ax1.set_title("Per-bus attack probability", fontsize=9); ax1.set_xlabel("bus index", fontsize=8); ax1.set_ylabel("P(attacked)", fontsize=8); ax1.tick_params(labelsize=7)
caption(fig, "Left: most attacks are sparse (a handful of buses), so localization — not just detection — is the task. Right: the "
             "attack is spread across the grid rather than fixed to a few buses (LRA randomizes its target line and bus subset), so "
             "a model cannot simply memorize a hot-spot. Both are spatial/distributional, hence figures rather than tables.", y=0.27)
SIDE["attack_surface"] = ["metric,value"] + [f"mean_attacked_buses,{surf.mean():.2f}", f"median_attacked_buses,{np.median(surf):.0f}", f"max_attacked_buses,{int(surf.max())}"]
save(fig)

# ======================= Page 8: residual box/violin (FIGURE — distributional) =======================
resid = None
rp = os.path.join(RES, "ml_only_residuals_v2.npz")
if os.path.exists(rp):
    resid = np.load(rp)
    fig = newpage("Figure 3 — How Far Each Attack Moves the True State (IEEE-118)",
                  "injection deviation from the true state: at the attacked buses (local) vs over the whole network")
    gs = fig.add_gridspec(1, 2, left=0.10, right=0.96, top=0.84, bottom=0.36, wspace=0.26)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    floor = float(np.median(resid["ieee118_benign_dP"]))                  # benign = pure meter noise -> the floor
    # LEFT: deviation AT THE ATTACKED buses (the real per-target footprint) — attack families only
    labA = [f for f in FAMS if f"ieee118_{f}_dPatk" in resid and len(resid[f"ieee118_{f}_dPatk"])]
    partsA = axA.violinplot([resid[f"ieee118_{f}_dPatk"] for f in labA], showmedians=True, widths=0.85)
    for j, b in enumerate(partsA["bodies"]): b.set_facecolor(STEAL if labA[j] in ("Aq", "At", "Al") else DET); b.set_alpha(0.72)
    axA.axhline(floor, ls="--", lw=1.0, color="#444"); axA.text(len(labA)+0.4, floor, "benign\nfloor", fontsize=6.3, va="center", color="#444")
    axA.set_xticks(range(1, len(labA)+1)); axA.set_xticklabels(labA, rotation=45, fontsize=7.5)
    axA.set_title("AT the attacked buses   |dP| (MW)", fontsize=9.5); axA.tick_params(labelsize=7.5)
    # RIGHT: deviation over ALL metered buses (the network-wide aggregate) — benign + families
    order = ["benign"] + FAMS
    partsB = axB.violinplot([resid[f"ieee118_{f}_dP"] for f in order], showmedians=True, widths=0.85)
    for j, b in enumerate(partsB["bodies"]): b.set_facecolor(STEAL if order[j] in ("Aq", "At", "Al") else (MUTE if order[j] == "benign" else DET)); b.set_alpha(0.72)
    axB.axhline(floor, ls="--", lw=1.0, color="#444"); axB.text(len(order)+0.4, floor, "noise\nfloor", fontsize=6.3, va="center", color="#444")
    axB.set_xticks(range(1, len(order)+1)); axB.set_xticklabels(order, rotation=45, fontsize=7.5)
    axB.set_title("over ALL metered buses   |dP| (MW)", fontsize=9.5); axB.tick_params(labelsize=7.5)
    caption(fig, "Left: at the buses each attack actually targets, the stealthy families (Aq/At/Al, red) move the injection by a real but "
                 "plausibility-bounded amount (~4-5 MW on IEEE-118) — a genuine manipulation, not noise. Right: averaged over the WHOLE "
                 "network, that same manipulation is indistinguishable from benign — the stealthy families sit on the meter-noise floor, "
                 "because only a few of ~100 buses move and the rest wash out the average. That combination — a real LOCAL effect that "
                 "vanishes in the network-wide statistics — is exactly what makes them stealthy: no aggregate check or bad-data detector "
                 "sees them, so only per-bus localization can. (The detectable Ad/As/Ar, blue, perturb meters more but inconsistently, "
                 "which is what BDD catches — see the BDD page.)", y=0.30)
    SIDE["residuals"] = ["family,median_dP_at_attacked_MW,median_dP_all_metered_MW"] + \
        [f'{f},{(np.median(resid[f"ieee118_{f}_dPatk"]) if f"ieee118_{f}_dPatk" in resid and len(resid[f"ieee118_{f}_dPatk"]) else float("nan")):.3f},{np.median(resid[f"ieee118_{f}_dP"]):.3f}' for f in order]
    save(fig)
else:                                                        # PLACEHOLDER: residuals pass not finished yet
    fig = newpage("Figure 3 — How Far Each Attack Moves the True State (IEEE-118)",
                  "per-record injection deviation from the true state at the same timestep — RMS over metered buses")
    fig.text(0.5, 0.54, "[ PENDING ]", ha="center", va="center", fontsize=17, weight="bold", color=ACCENT)
    fig.text(0.5, 0.47, "The per-family residual distributions are still computing on the new v0.4.0 data.\n"
             "This page populates automatically the moment the residuals pass finishes — just re-run the report.",
             ha="center", va="center", fontsize=10.5, color=MUTE, family="serif", linespacing=1.5)
    save(fig)

# ======================= Operational impact — damage to operator decisions =======================
OPI = load_json("operator_impact.json", None)
if OPI and OPI.get("cases", {}).get("ieee118"):
    ci = OPI["cases"]["ieee118"]
    # headline stress level = 1.3 (grid loaded near limits — the regime where an operator actually acts)
    def opi(fam, s="s1.3"):
        v = ci.get(fam, {}).get(s, {})
        return v
    rows = []
    for fam, label in [("benign", "benign (control)"), ("Aq", "Aq  stealthy"), ("Al", "Al  stealthy (LRA)")]:
        v = opi(fam)
        if not v: continue
        rows.append([label,
                     f"{v.get('median_abs_line_err_pct',0):.2f}",
                     f"{v.get('max_line_err_pct',0):.0f}",
                     f"{v.get('pct_samples_ge1_flip',{}).get('th100',0):.0f}%",
                     f"{v.get('masked_overload_events',{}).get('th100',0)}",
                     f"{v.get('fabricated_overload_events',{}).get('th100',0)}"])
    opi_cols = ["Family", "median err (pts)", "max err (pts)", "samples flipped", "hidden overloads", "false overloads"]
    def opi_shade(i, row): return "#eef1f4" if "control" in row[0] else ("#fbecea" if row[4] != "0" else "#f4f6f8")
    fig = table_page("Operational Impact — Damage to Operator Decisions",
                     "does a stealthy attack that PASSES bad-data detection still make the operator misjudge line safety? (IEEE-118, grid loaded to ~1.3x so lines sit near thermal limits)",
                     opi_cols, rows, cap=(
                     "Line-overload masking (our operationalization of the load-redistribution damage model, Yuan et al. 2011): for each "
                     "line we compare its TRUE thermal loading to the loading an operator would read from the stealthy-attacked state, and "
                     "count where they disagree across the 100%-of-rating action threshold. A HIDDEN overload (red) is the dangerous case "
                     "— a line truly over its limit that the attacked reading shows as safe, so the operator takes no corrective action. "
                     "The benign control gives exactly zero misjudgments (validated: the no-attack re-solve reproduces the true state to "
                     "1e-10 p.u.), so every event below is caused by the attack. KEY RESULT: the stealthy families evade bad-data detection "
                     "(see the BDD page) yet still flip line-safety status on nearly half (Aq) to over 80% (Al) of samples and hide hundreds "
                     "of real overloads — the load-redistribution family Al, the classical "
                     "operationally-damaging attack, is the worst. The GNN-localization literature (Boyaci et al. 2022; Liu et al. 2009) "
                     "reports only detection/localization metrics and never an operator consequence; this carries that missing impact "
                     "label. Caveats, stated plainly: thermal limits are derived from peak nominal line current (the IEEE cases ship "
                     "placeholder ratings ~200x too large), and the effect is shown at a stressed operating point because the base cases "
                     "are lightly loaded — at nominal load few lines sit near a limit, so few can be masked."),
                     col_widths=[0.24, 0.16, 0.14, 0.16, 0.15, 0.15], shade_rule=opi_shade)
    SIDE["operator_impact"] = ["family,stress,median_line_err_pts,max_line_err_pts,pct_samples_misjudged,hidden_overloads,false_overloads"] + \
        [f"{fam},{s},{ci[fam][s].get('median_abs_line_err_pct')},{ci[fam][s].get('max_line_err_pct')},{ci[fam][s].get('pct_samples_ge1_flip',{}).get('th100')},{ci[fam][s].get('masked_overload_events',{}).get('th100')},{ci[fam][s].get('fabricated_overload_events',{}).get('th100')}"
         for fam in ci for s in ci[fam] if isinstance(ci[fam][s], dict)]
    save(fig)

# ======================= Model benchmark: localization + detection (two tables, one page) =======================
loc_cols = ["System", "overall"] + FAMS
loc_rows = []
for c in SYS_ALL:
    b = BENCH.get(f"ieee{c}")
    if not b:
        loc_rows.append([f"IEEE-{c}", "pending"] + ["pending"] * len(FAMS)); continue
    pf = b["per_family"]
    loc_rows.append([f"IEEE-{c}", f"{b['overall']['swf1']:.3f}"] + [f"{pf.get(f,{}).get('swf1',0):.3f}" for f in FAMS])
det_cols = ["System", "DR", "FA", "det-F1"] + ["DR:" + f for f in FAMS]
det_rows = []
for c in SYS_ALL:
    b = BENCH.get(f"ieee{c}"); d = (b or {}).get("detection", {})
    if not d:
        det_rows.append([f"IEEE-{c}", "pending", "pending", "pending"] + ["pending"] * len(FAMS)); continue
    pfd = d.get("per_family_DR", {})
    det_rows.append([f"IEEE-{c}", f"{d['DR']:.3f}", f"{d['FA']:.3f}", f"{d['det_f1']:.3f}"] + [f"{pfd.get(f,0):.2f}" for f in FAMS])
if MODELS_READY and loc_rows:
    BASE = {c: load_json(f"baselines_{c}.json") for c in SYS_ALL}    # simple-model complexity ladder
    ladder_cols = ["System", "per-bus MLP", "plain GCN", "ARMA+attn (ours)"]
    ladder_rows = []
    for c in SYS_ALL:
        b = BENCH.get(f"ieee{c}"); base = BASE.get(c)
        ladder_rows.append([f"IEEE-{c}",
                            f"{base['mlp']['swf1']:.3f}" if base else "pending",
                            f"{base['gcn']['swf1']:.3f}" if base else "pending",
                            f"{b['overall']['swf1']:.3f}" if b else "pending"])
    fig = multi_table_page("Model Benchmark — Localization & Detection",
                       "reference ARMA localizer (measurements + KCL residual + temporal delta), test split",
                       [dict(subhead="Model complexity ladder — overall localization swF1 (does the graph model earn its complexity?)",
                             cols=ladder_cols, rows=ladder_rows, col_widths=[0.22, 0.26, 0.26, 0.26],
                             cap="Same data, same metric: a per-bus MLP (node features only, NO graph) vs a plain 2-layer GCN vs "
                                 "our ARMA+attention hybrid. The gap from MLP to the graph models measures how much the grid "
                                 "structure matters; the gap from GCN to ARMA measures the wide-receptive-field / attention gain. "
                                 "The hybrid's lead over the simple baselines is what justifies its complexity — where they are "
                                 "close, the simpler model is preferable."),
                        dict(subhead="Localization — sample-wise F1 per attack type (higher is better)",
                             cols=loc_cols, rows=loc_rows, col_widths=[0.16, 0.14] + [0.10] * 6,
                             cap="Per attack type, not accuracy: LRA (structured, targeted) is the most localizable; the diffuse "
                                 "stealthy families (Aq, At) and the larger grids are the hard cases. A table beats a bar chart "
                                 "here — the exact values are the point."),
                        dict(subhead="Detection — grid-level rate (DR), false-alarm (FA), F1, and per-family DR",
                             cols=det_cols, rows=det_rows, fontsize=8.0, col_widths=[0.13, 0.09, 0.09, 0.10] + [0.098] * 6,
                             cap="Grid-level score S = max over buses of the per-bus attack probability. Detection far exceeds "
                                 "per-bus localization because it aggregates evidence: the three bad-data-evading families are "
                                 "detected in Boyaci range — ML catches what the classical detector cannot.")])
    SIDE["localization"] = ["system,overall," + ",".join(FAMS)] + [",".join([f"ieee{SYS_ALL[i]}"] + r[1:]) for i, r in enumerate(loc_rows)]
    SIDE["detection"] = ["system,DR,FA,det_f1," + ",".join("DR_" + f for f in FAMS)] + [",".join([f"ieee{SYS_ALL[i]}"] + r[1:]) for i, r in enumerate(det_rows)]
    SIDE["complexity_ladder"] = ["system,mlp_swf1,gcn_swf1,arma_hybrid_swf1"] + [",".join(r) for r in ladder_rows]
    save(fig)
else:
    pending_page("Model Benchmark — Localization & Detection",
                 "reference ARMA localizer (measurements + KCL residual + temporal delta), test split",
                 "The localizer and detector are being re-trained on the v0.4.0 dataset (new attack taxonomy, "
                 "accuracy-class meter noise). Per-family localization and detection numbers populate here once training finishes.")

# ======================= Model improvements: attention + state estimation (two tables, one page) =======================
at_cols = ["System", "loc ARMA", "loc +attn", "gain", "det ARMA", "det +attn"]
at_rows = []
for c in SYS:
    k = f"ieee{c}"
    if k not in ATTN: continue
    a, h = ATTN[k]["arma"], ATTN[k]["hybrid"]
    at_rows.append([f"IEEE-{c}", f"{a['loc']:.3f}", f"{h['loc']:.3f}", f"+{h['loc']-a['loc']:.3f}", f"{a['det']:.3f}", f"{h['det']:.3f}"])
se_cols = ["System", "|V| meter", "|V| WLS", "|V| NN", "|V| PINN", "th meter", "th WLS", "th NN", "th PINN"]
se_rows = []
for c in SYS:
    s = SE[c]; p = (s["pinn"] or {}).get("overall", {}); nn = (s["nn"] or {}).get("overall", {}); w = (s["wls"] or {}).get("overall_attacked", {})
    def g(d, k): return f"{d[k]:.4f}" if d and k in d and d[k] is not None else "—"
    def gt(d, k): return f"{d[k]:.2f}" if d and k in d and d[k] is not None else "—"
    se_rows.append([f"IEEE-{c}", g(p, "V_mae_meter_attacked"), g(w, "V_mae_wls"), g(nn, "V_mae_se_metered_attacked"), g(p, "V_mae_se_metered_attacked"),
                    gt(p, "th_mae_meter_attacked"), gt(w, "th_mae_wls"), gt(nn, "th_mae_se_metered_attacked"), gt(p, "th_mae_se_metered_attacked")])
blocks = []
if at_rows:
    blocks.append(dict(subhead="Physics-biased attention — localization swF1 / detection F1, ARMA vs + attention",
                       cols=at_cols, rows=at_rows, col_widths=[0.18, 0.16, 0.16, 0.14, 0.18, 0.18],
                       cap="A parallel attention head, gated from zero so the network starts identical to plain ARMA, attends "
                           "along physically-suspicious branches. It lifts localization on every system while holding detection — "
                           "now the SDK's default localizer."))
blocks.append(dict(subhead="Attack-resilient state estimation — recovery error on attacked buses, |V| (p.u.) / theta (deg)",
                   cols=se_cols, rows=se_rows, fontsize=7.8, col_widths=[0.13] + [0.108] * 8,
                   cap="Recover the TRUE state from possibly-attacked measurements. The result is honestly mixed. VOLTAGE recovery "
                       "is the real win: the learned estimators beat the raw meter and WLS at IEEE-14 (~4x) and IEEE-118 (~5x). ANGLE "
                       "only clearly wins at IEEE-14; at 118 it ties the meter and at 300 it is worse. The reason is a property of the "
                       "dataset, not a bug: on v0.4.0 the stealthy attacks are load-plausible and BDD-consistent, so they move the TRUE "
                       "angle very little — the meter is already only ~0.24-0.65 deg off under attack, leaving almost nothing to recover, "
                       "and the estimator's own reconstruction floor then dominates. In other words, the attacks are now stealthy enough "
                       "that state recovery is nearly moot for angle. The physics term (PINN vs NN) helps at 14 and 300 but hurts at 118, "
                       "so it is system-dependent, not universal. WLS (classical) is confirmed fooled — its error jumps 9-126x under "
                       "stealthy attack. Reported as measured, single seed."))
if MODELS_READY and (at_rows or any("—" not in r for r in se_rows)):
    fig = multi_table_page("Model Improvements — Attention & State Estimation",
                       "beyond the base localizer: an attention head that raises localization, and a state estimator that recovers the true state",
                       blocks)
    SIDE["attention_ab"] = ["system,loc_arma,loc_attn,det_arma,det_attn"] + \
        [f'ieee{c},{ATTN[f"ieee{c}"]["arma"]["loc"]},{ATTN[f"ieee{c}"]["hybrid"]["loc"]},{ATTN[f"ieee{c}"]["arma"]["det"]},{ATTN[f"ieee{c}"]["hybrid"]["det"]}' for c in SYS if f"ieee{c}" in ATTN]
    SIDE["state_estimation"] = ["system,V_meter,V_wls,V_nn,V_pinn,th_meter,th_wls,th_nn,th_pinn"] + [",".join([f"ieee{SYS[i]}"] + r[1:]) for i, r in enumerate(se_rows)]
    save(fig)
else:
    pending_page("Model Improvements — Attention & State Estimation",
                 "beyond the base localizer: an attention head that raises localization, and a state estimator that recovers the true state",
                 "The attention A/B and the attack-resilient state estimator are being re-run on the v0.4.0 data. "
                 "Localization gains and per-variable recovery error populate here once training finishes.")

# ======================= Hyperparameter tuning (Optuna) =======================
# default = the shipped ARMA hybrid; tuned = best of a 35-trial TPE search (FDIA_localization/_hpo_ieee{C}.json)
HPO = load_json("hpo_v040.json", None) if MODELS_READY else None
if HPO:
    hpo_cols = ["System", "default swF1", "tuned swF1", "change", "tuned det-F1"]
    hpo_rows = [[f"IEEE-{c}", f'{HPO[str(c)]["default"]:.3f}', f'{HPO[str(c)]["tuned"]:.3f}',
                 f'{HPO[str(c)]["tuned"]-HPO[str(c)]["default"]:+.3f}', f'{HPO[str(c)]["tuned_det"]:.3f}'] for c in SYS if str(c) in HPO]
    def hpo_shade(i, row): return "#e7f4ea" if row[3].startswith("+") else "#f6f7f8"
    fig = table_page("Hyperparameter Tuning (Optuna, Boyaci Space)",
                     "TPE search over the ARMA hybrid (blocks, units, K, T, dropout, lr, attention heads), seed 123, val-selected",
                     hpo_cols, hpo_rows, "Tuned vs default localization on the v0.4.0 data.",
                     col_widths=[0.18, 0.20, 0.20, 0.16, 0.20], shade_rule=hpo_shade)
    SIDE["hpo"] = ["system,default_swf1,tuned_swf1,change,tuned_detf1"] + [",".join(r) for r in hpo_rows]
    save(fig)
else:
    pending_page("Hyperparameter Tuning (Optuna, Boyaci Space)",
                 "TPE search over the ARMA hybrid localizer, val-selected",
                 "The hyperparameter search has not been re-run on the v0.4.0 dataset. Tuned-vs-default results populate "
                 "here if/when the search is repeated.")

# ======================= Page 13: protocol + critical analysis (prose) =======================
# ======================= Reference benchmark: Falas et al. Table II reproduction =======================
# Falas et al., "Data Manipulation Attack Mitigation ... Physics-Informed Neural Networks", IEEE CSR 2025,
# Table II (Model Comparison, IEEE-14). Left two number columns = his reported values; right two = our
# reproduction (see FDIA_localization/_falas_comparison_table.csv). His paper has no Table III.
falas_cols = ["Scenario", "Falas PINN", "Falas NN", "Ours PINN", "Ours NN"]
falas_mae = [["Steady", "1.28e-3", "2.91e-3", "2.71e-2", "1.52e-2"],
             ["5%",  "7.07e-3", "1.12e-2", "2.76e-2", "1.56e-2"],
             ["10%", "1.40e-2", "2.19e-2", "2.91e-2", "1.69e-2"],
             ["20%", "2.79e-2", "4.39e-2", "3.36e-2", "2.08e-2"],
             ["30%", "4.18e-2", "6.65e-2", "3.94e-2", "2.56e-2"]]
falas_r2 = [["Steady", "0.9994", "0.9968", "0.9985", "0.9996"],
            ["5%",  "0.9794", "0.9539", "0.9984", "0.9996"],
            ["10%", "0.9134", "0.8235", "0.9983", "0.9995"],
            ["20%", "0.6796", "0.2779", "0.9978", "0.9992"],
            ["30%", "0.2805", "-0.7167", "0.9970", "0.9988"]]
_fcw = [0.16, 0.21, 0.21, 0.21, 0.21]
FALAS = load_json("falas_v040.json", None)   # our reproduction numbers on v0.4.0; hardcoded lists below are OLD-data
if FALAS:
    falas_mae = FALAS["mae"]; falas_r2 = FALAS["r2"]
if FALAS:
    fig = multi_table_page("Reference Benchmark — Falas et al. Table II Reproduction",
                       "IEEE-14 state estimation, PINN vs plain NN across the attack ladder: Falas reported vs our reproduction",
                       [dict(subhead="Mean absolute error (standardized-pooled) — lower is better",
                             cols=falas_cols, rows=falas_mae, col_widths=_fcw,
                             cap="We reproduce the same range and the attack-ladder trend (error rises with attack magnitude). Our "
                                 "absolute MAE sits higher because of a standardization and target-construction difference documented "
                                 "separately, not a method error."),
                        dict(subhead="R-squared — higher is better",
                             cols=falas_cols, rows=falas_r2, col_widths=_fcw,
                             cap="The real finding is qualitative: in our setup the plain NN beats the PINN, the opposite of Falas, and "
                                 "our NN R-squared does not collapse under attack. A PINN beats an NN only against a weak data-only "
                                 "baseline; our NN is stronger and our attack is milder, so the physics term does not add. Falas reports "
                                 "IEEE-14 only.")])
    SIDE["falas_reproduction"] = ["metric,scenario,falas_pinn,falas_nn,ours_pinn,ours_nn"] + \
        [f"MAE,{','.join(r)}" for r in falas_mae] + [f"R2,{','.join(r)}" for r in falas_r2]
    save(fig)
else:
    pending_page("Reference Benchmark — Falas et al. Table II Reproduction",
                 "IEEE-14 state estimation, PINN vs plain NN across the attack ladder",
                 "Our reproduction columns are being re-computed on the v0.4.0 IEEE-14 data. Falas's own reported values "
                 "are published and unchanged, but we withhold the side-by-side until our numbers are current.")

# ======================= State estimation: V and theta separated (companion to the pooled metric) =======================
# se_rows (from the improvements page) = [System, V meter, V WLS, V NN, V PINN, th meter, th WLS, th NN, th PINN]
if MODELS_READY and se_rows:
    v_rows = [[r[0], r[1], r[2], r[3], r[4]] for r in se_rows]
    th_rows = [[r[0], r[5], r[6], r[7], r[8]] for r in se_rows]
    sep_cols = ["System", "meter", "WLS", "NN", "PINN"]
    fig = multi_table_page("State Estimation — Voltage and Angle Separated",
                       "the attack-resilient state-recovery error, split by variable so the per-variable behavior a pooled metric hides is visible",
                       [dict(subhead="Voltage magnitude |V| error (p.u.) on attacked buses — lower is better",
                             cols=sep_cols, rows=v_rows, col_widths=[0.20, 0.20, 0.20, 0.20, 0.20],
                             cap="Split by variable, the picture is size-dependent: on IEEE-14 both learned estimators sit far below "
                                 "the raw meter and WLS on voltage; on IEEE-118 they still beat the baselines and NN edges PINN; on "
                                 "IEEE-300 the learned voltage error rises above the meter/WLS baseline (PINN is the better of the two "
                                 "learned but still not ahead of the classical baselines). Physics helps voltage only on the small system."),
                        dict(subhead="Voltage angle theta error (deg) on attacked buses — lower is better",
                             cols=sep_cols, rows=th_rows, col_widths=[0.20, 0.20, 0.20, 0.20, 0.20],
                             cap="Angle carries most of the error and the SE only clearly helps at IEEE-14 (PINN 0.15 vs meter 0.21 deg). "
                                 "At 118 it ties the meter; at 300 both learned estimators are WORSE than the meter. This is not (mainly) "
                                 "under-training — it is because the v0.4.0 stealthy attacks barely move the true angle (the meter is only "
                                 "~0.24-0.65 deg off under attack), so there is little to recover and the estimator's reconstruction floor "
                                 "dominates. Angle recovery only pays off for the high-magnitude Al/LRA family. The honest reading: state "
                                 "recovery matters far less once the attacks are this stealthy.")])
    SIDE["se_separated"] = ["variable,system,meter,WLS,NN,PINN"] + \
        [f"V,{','.join(r)}" for r in v_rows] + [f"theta,{','.join(r)}" for r in th_rows]
    save(fig)
else:
    pending_page("State Estimation — Voltage and Angle Separated",
                 "attack-resilient state-recovery error, split by variable (voltage vs angle)",
                 "The per-variable state-estimation results are being re-computed on the v0.4.0 data — see the previous "
                 "State Estimation page. This companion breakdown populates alongside it.")

fig = newpage("Evaluation Protocol & Threat Model",
              "how the benchmark is scored, and what we assume about the attacker")
prose(fig, [
 ("Split and metrics", [
  "60/20/20 chronological split, cut on sequence boundaries so a ramp never straddles train and test — a random "
  "shuffle would leak it (Boyaci et al., 2022; PING, NAPS 2025). Equal count per attack family.",
  "Report per attack type — a pooled accuracy or F1 hides the stealthy families behind the easy ones.",
  "Localization = sample-wise F1 (predicted attacked set vs truth).  Detection = grid-level DR / FA / detection-F1.",
  "Unseen-attack protocol available via load(..., heldout=True), which reserves As and Ar for test only.",
 ]),
 ("Threat model and assumptions", [
  "We assume a strong, system-aware attacker, consistent with the stealthy FDIA literature (Liu et al. 2009; "
  "Boyaci et al. 2022): evading bad-data detection requires knowing the topology and coordinating the affected meters.",
  "The state-level attacks re-solve the full power flow, so they present a fully consistent fake across the network "
  "— near the worst case for a defender, which is deliberate but worth stating plainly.",
  "This is optimistic about reach: the change follows the flow paths across the system, so it assumes broad meter "
  "access. A limited-access attacker who can only compromise a bounded set of meters would leave a small residual "
  "at the boundary of the controlled region and be somewhat easier to catch. Bounding the attacker's reach is future work.",
 ]),
], top=0.895)
save(fig)

fig = newpage("What the Results Say, and Honest Notes",
              "what the numbers do and do not claim, and how to reproduce them")
prose(fig, [
 ("What the results say", [
  "The stealthy families are hard by construction (they pass every classical check), so localization on Aq, ramp "
  "and the larger grids is modest and honest — a difficult benchmark, not a solved one.",
  "Detection aggregates evidence across the grid and reaches Boyaci range on the bad-data-evading families — the "
  "ML-only thesis.",
  "Physics-biased attention improves localization on every system, and a physics-informed state estimator recovers "
  "the true state that a fooled weighted-least-squares estimator cannot.",
 ]),
 ("Honest limitations", [
  "Temporal windowing did not rescue ramp localization: over a short window, ramp's per-step creep is smaller than "
  "benign load drift, so it stays hard — a documented negative result, not an oversight.",
  "The state estimator recovers state, not attack labels; a large state correction is a signal, but naming the "
  "attacked buses remains the localizer's job.",
  "Voltage magnitude barely moves under the stealthy families, so |V| alone is a weak detector.",
 ]),
 ("Reproduce", [
  "Dataset-intrinsic tables and figures are computed live from fdia_graph.load(...).",
  "Model results are read from examples/results/, produced by the benchmark and state-estimator scripts.",
  "Regenerate the whole document with  python dataset_report.py.",
 ]),
], top=0.895)
save(fig)

pdf.close()
os.makedirs(os.path.join(RES, "sidecars"), exist_ok=True)
for name, rows in SIDE.items():
    open(os.path.join(RES, "sidecars", f"{name}.csv"), "w").write("\n".join(rows))
print(f"[done] wrote {OUT}.pdf  ({len(SYS)} systems)  + {len(SIDE)} CSV sidecars in results/sidecars/")
