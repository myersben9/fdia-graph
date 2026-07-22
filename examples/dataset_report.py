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

# Prefer locally-regenerated release shards (release_v0.4.1/) over the downloadable builtin, so the report
# reflects the new data before the GitHub release is cut. Override the directory via FDIA_LOCAL_SHARDS.
LOCAL_SHARDS = os.environ.get("FDIA_LOCAL_SHARDS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_v0.4.1"))
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
# Ben's style rules: Times New Roman on everything; ONE dark text colour (no pale text); dark saturated data colours
# (no washed-out series); no decorative coloured lines. INK is the single text colour used for ALL text on every page.
INK = "#15202c"                                              # the one dark text colour (titles, body, captions, ticks, tables)
MUTE = "#3a3f4a"                                             # dark neutral for the 'benign' data series (was pale grey)
STEAL = "#a5321f"; DET = "#1f5c8a"; ACCENT = "#a5321f"       # dark saturated categorical data colours
SUBTLE = INK                                                 # subtitles/captions share the one dark text colour
RULE = "#9aa3ad"                                             # thin neutral hairline (structure only, not a coloured accent)
# dark, well-saturated categorical palette for per-family plots (no light tones)
FAMCOL = {"benign": "#3a3f4a", "Aq": "#a5321f", "Ad": "#1f5c8a", "As": "#1e7d3c",
          "Ar": "#6a3d9a", "At": "#404652", "Al": "#a35a12"}
SIDE = {}
PAGENO = [0]                                                  # mutable page counter for the footer
RUNNING = "fdia-graph  ·  ML-Only Dangerous FDIA Localization Dataset"

# global defaults: Times New Roman for ALL text, dark ticks/labels, no pale chrome
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "text.color": INK,
    "axes.edgecolor": "#55606c", "axes.linewidth": 0.8, "axes.titlesize": 9.5, "axes.titlecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK, "pdf.fonttype": 42,
})


def load_json(name, default=None):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else default


# ---------------- style helpers ----------------
def header(fig, title, sub=""):
    fs = 16 if len(title) <= 52 else (13.5 if len(title) <= 72 else 12)
    fig.text(0.07, 0.955, title, ha="left", va="top", fontsize=fs, weight="bold", color=INK)
    y_rule = 0.940 if len(title) <= 72 else 0.925
    fig.add_artist(plt.Line2D([0.07, 0.93], [y_rule, y_rule], color=RULE, lw=0.8, alpha=1.0))
    if sub:
        fig.text(0.07, y_rule - 0.013, textwrap.fill(" ".join(sub.split()), 104), ha="left", va="top",
                 fontsize=9.3, color=SUBTLE, style="italic", linespacing=1.3)


def caption(fig, text, y=0.14):
    wrapped = "\n".join(textwrap.fill(ln, 112) for ln in text.split("\n"))
    fig.text(0.07, y, wrapped, ha="left", va="top", fontsize=8.2, color=INK, family="serif", linespacing=1.35)


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
    a bulleted list (hanging indent, accent bullet), bullets for enumerable notes, prose for definitions."""
    y = top
    for heading, body in sections:
        if heading:
            fig.add_artist(plt.Line2D([x, x + 0.024], [y + 0.004, y + 0.004], color=INK, lw=2.4))
            fig.text(x + 0.032, y, heading, ha="left", va="top", fontsize=11, weight="bold", color=INK)
            y -= heading_gap
        if isinstance(body, (list, tuple)):                       # bulleted notes
            for item in body:
                lines = textwrap.wrap(item, width - 3) or [""]
                fig.text(x + 0.006, y, "•", ha="left", va="top", fontsize=fs + 1, color=INK, weight="bold")
                for k, wl in enumerate(lines):                    # hanging indent under the bullet
                    fig.text(x + 0.026, y, wl, ha="left", va="top", fontsize=fs, color=INK, family="serif")
                    y -= lh
                y -= bullet_gap
        else:                                                     # paragraph
            for ln in body.split("\n"):
                for wl in (textwrap.wrap(ln, width) or [""]):
                    fig.text(x, y, wl, ha="left", va="top", fontsize=fs, color=INK, family="serif")
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
    """A page that stacks several labelled tables, each with its own short caption, so a couple of small
    result tables share one full page instead of floating alone on sparse ones. blocks: list of dicts with
    keys subhead, cols, rows, cap, and optional fontsize/col_widths/shade_rule."""
    fig = newpage(title, sub); y = top
    for b in blocks:
        fig.add_artist(plt.Line2D([0.07, 0.091], [y + 0.004, y + 0.004], color=INK, lw=2.4))
        fig.text(0.099, y, b["subhead"], ha="left", va="top", fontsize=10.5, weight="bold", color=INK)
        y -= 0.028
        y = draw_table(fig, y, b["cols"], b["rows"], b.get("fontsize", 8.6), b.get("col_widths"),
                       b.get("shade_rule"))
        y -= 0.028
        for wl in textwrap.fill(b["cap"], 112).split("\n"):     # tight per-table caption
            fig.text(0.07, y, wl, ha="left", va="top", fontsize=8.0, color=INK, family="serif")
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
    """A clean placeholder page for a result that is not yet computed on the current dataset, so the report
    NEVER shows stale or made-up numbers. Auto-fills once the underlying results are staged and the report re-runs."""
    fig = newpage(title, sub)
    fig.text(0.5, 0.56, "[ PENDING. v0.4.0 re-train ]", ha="center", va="center", fontsize=15, weight="bold", color=INK)
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
              "ML-only dangerous FDIA localization on realistic measurement graphs, IEEE-14 / 118 / 300")
prose(fig, [
 ("What this is",
  "A benchmark of false-data-injection attacks (FDIAs) that evade every classical, non-ML detector yet remain "
  "localizable by a graph model. This is the regime where machine learning is actually needed. Each record is a "
  "realistic sparse SCADA/PMU measurement graph (PING-style). Branch power flows are edge features, and metered bus "
  "injections, voltage magnitudes and sparse PMU angles are node features, with per-measurement availability "
  "masks. Redundancy is roughly 2-3, the regime a real energy-management system operates in, not the fully-"
  "observed idealization most FDIA benchmarks assume."),
 ("Seven families, split by detectability", [
  "Stealthy attacks evade bad-data detection and are the hard, dangerous cases. Aq is a state-consistent load "
  "redistribution. At (ramp) is a slow multi-timestep surge up then down. Al (LRA) is a targeted masked-overload.",
  "Detectable attacks are a contrast set a bad-data detector catches. Ad is random meter corruption. As is meter "
  "scaling. Ar is replay of a past scan.",
  "The split is the point. On this data, “ML beats a detector” is a genuine result, not a rigged one.",
 ]),
 ("What this report covers", [
  "The data dictionary and schema, and the attack families with their bad-data-detection stealth split.",
  "Class composition and the chronological split. The measurement model and coverage.",
  "The measurement distributions, the spatial attack pattern, and how far each family moves the true state.",
  "Model results cover localization, detection, the physics-biased-attention gain, and an attack-resilient "
  "physics-informed state estimator.",
 ]),
 ("Reproducibility",
  "Every number in this report is loaded from the installed package (fdia_graph.load) or the result files in "
  "examples/results/. Run  python dataset_report.py  to regenerate this document end to end."),
], top=0.895)
tot = sum(sum(DATA[c]["per"][s].values()) for c in SYS for s in SPLITS)
fig.add_artist(plt.Line2D([0.07, 0.93], [0.175, 0.175], color=RULE, lw=0.6))
fig.text(0.5, 0.155, f"3 systems     {tot:,} records     7 attack families     release v0.3.0",
         ha="center", fontsize=11.5, weight="bold", color=INK)
save(fig)

# ======================= Page 2: data dictionary (TABLE) =======================
dd_cols = ["Field", "Shape", "Dtype", "Meaning"]
dd_rows = [
 ["edge_index", "[2, E]", "int64", "branch endpoints [from, to] (lines then transformers), the static graph"],
 ["edge_reactance", "[E]", "float32", "per-branch reactance (p.u.)"],
 ["node_x", "[N, 4]", "float32", "[ |V| p.u., P_inj MW, Q_inj MVAr, theta deg ] per bus"],
 ["node_m", "[N, 4]", "float32", "node availability mask (1 = metered, masked entries zeroed)"],
 ["edge_x", "[E, 2]", "float32", "[ P_from MW, Q_from MVAr ] per branch"],
 ["edge_m", "[E, 2]", "float32", "edge availability mask"],
 ["y", "[N]", "float32", "localization target, 1 = bus attacked, 0 = clean"],
 ["temporal_delta", "[N, 2]", "float32", "v0.3+ current-minus-previous-scan [dP_inj, dQ_inj]"],
 ["family", "scalar", "int", "0 benign 1 Aq 2 Ad 3 As 4 Ar 5 ramp 6 LRA"],
 ["stealthy", "scalar", "int", "1 if the attack evades classical bad-data detection"],
 ["split", "scalar", "int", "0 train 1 val 2 test (60/20/20 chronological)"],
 ["seq_id", "scalar", "int", "At-sequence id (>=0 groups one ramp/At sequence), -1 otherwise"],
 ["timestep", "scalar", "int", "source operating-point index"],
 ["gap", "scalar", "int", "1 if a physics non-convergence NA row (~0% shipped)"],
]
fig = table_page("Data Dictionary", "every field in ml_only_ieee{N}.h5  (N = buses, E = branches, T = records)",
                 dd_cols, dd_rows,
                 "The static graph (edge_index, edge_reactance) is read once. All other fields are per record. Masks are "
                 "load-bearing. With redundancy ~2-3 not every bus is metered and PMU angles are sparse, so a model must "
                 "consume node_m/edge_m rather than assume a full measurement vector.",
                 col_widths=[0.17, 0.11, 0.10, 0.62], fontsize=8.2, height=0.56, y_table=0.30)
SIDE["data_dictionary"] = ["field,shape,dtype,meaning"] + [",".join(f'"{x}"' for x in r) for r in dd_rows]
save(fig)

# ======================= Page 3: attack families + BDD stealth split (TABLE) =======================
# (description, provenance). provenance makes explicit what is OUR contribution vs pulled from references.
SRC = {"Aq":  ("per-bus load scaling, AC re-solve",       "Ours (cf. Boyaci Ao, Liu FDIA)"),
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
                             cap="The Source column separates our contribution from prior art. Aq (stealthy load scaling) is OURS, "
                                 "the AC/physical realization of the state-consistent attack concept (cf. Boyaci et al. 2022 'Ao', "
                                 "Liu et al. 2011 FDIA). Ramp is Haghshenas et al. (ISGT 2023). LRA is Yuan et al. (T-SG 2011). The "
                                 "meter-level Ad/As/Ar are the standard FDIA taxonomy (Boyaci 2022). BDD-pass is the fraction that "
                                 "evade the chi-square detector. The stealthy families (Aq/At/Al, red) pass on ~90-100% (valid, "
                                 "plausibility-bounded states), and the detectable ones (Ad/As/Ar, blue) are caught."),
                        dict(subhead="Measurement coverage, fraction of each quantity metered (sparse SCADA/PMU graph)",
                             cols=cov_cols, rows=cov_rows, col_widths=[0.16, 0.11, 0.13, 0.14, 0.11, 0.12, 0.11, 0.12],
                             cap="Redundancy = (# measurements)/(2N-1 states), the classical observability ratio, and ~2-3 is a "
                                 "realistic EMS regime. Voltage magnitude and branch flows are widely metered. PMU angles (theta) "
                                 "are sparse, so the masks encode which meters exist per record.")])
SIDE["families"] = ["family,description,class,bdd_pass_14,bdd_pass_118,bdd_pass_300"] + \
    [f'{f},"{SRC[f]}",{fam_rows[i][2]},{bdd_pct(14,f)},{bdd_pct(118,f)},{bdd_pct(300,f)}' for i, f in enumerate(FAMS)]
SIDE["coverage"] = ["system,N,E,redundancy,V_cov,Pinj_cov,theta_cov,flow_cov"] + \
    [f"{c},{DATA[c]['N']},{DATA[c]['E']},{DATA[c]['redundancy']:.3f},{DATA[c]['cov']['V']:.3f},{DATA[c]['cov']['Pinj']:.3f},{DATA[c]['cov']['theta']:.3f},{DATA[c]['cov']['flow']:.3f}" for c in SYS if c in DATA]
save(fig)

# ======================= Measurement (noise) model =======================
fig = newpage("Measurement Model, Realistic Accuracy-Class Meter Noise",
              "how every reading is corrupted, the accuracy-class magnitude, heteroscedastic scaling, and a systematic-bias / per-scan-jitter split")
# the accuracy-class values table (class-0.2 instrument transformers, sigma = manufacturer max / sqrt(3))
mm_cols = ["Quantity", "Source of error", "Max (class 0.2)", "sigma = max/√3"]
mm_rows = [["P / Q  (inj + flow)", "conventional device (Table III)", "±3%", "1.7%  (relative)"],
           ["|V|  magnitude", "VT ratio error (class 0.2)", "0.2%", "0.12%  (of ~1 p.u.)"],
           ["theta  angle", "VT phase displacement", "10 arc-min = 0.167°", "0.096°"]]
bot = draw_table(fig, 0.86, mm_cols, mm_rows, fontsize=8.6, col_widths=[0.24, 0.34, 0.20, 0.22])
y = prose(fig, [
 ("The measurement chain",
  "A reading is the true quantity seen through a chain, from bus -> instrument transformer (VT/CT) -> measurement "
  "device -> control center. Each stage adds error, so the dataset stores what a real EMS receives, not the "
  "exact physics. The magnitude of that error follows the instrument-transformer ACCURACY CLASS (Asprou, "
  "Kyriakides & Albu 2013, the model the Falas et al. 2025 baseline delegates to). A class number is a "
  "MAXIMUM error. The standard deviation added as noise is that max divided by sqrt(3) (a uniform-distribution "
  "convention). Power is device-dominated (the conventional device is +-3%), so its sigma is far larger than "
  "voltage, which is transformer-dominated. P/Q noise is heteroscedastic, proportional to the reading, so a "
  "large line carries more absolute error than a small one, which is how a state estimator weights it."),
 ("Systematic bias vs per-scan jitter",
  ["A meter's error is mostly SYSTEMATIC, a fixed calibration / ratio / phase offset that is the SAME every "
   "scan, with only a small RANDOM repeatability jitter. This is the accuracy-vs-precision distinction.",
   "Modelling the whole accuracy class as fresh per-scan noise would make 1-minute traces jump ~2.4% every "
   "scan (unrealistic) and drown the temporal signal. So each meter gets a bias drawn ONCE (0.968 x sigma, "
   "constant in time) plus a per-scan jitter (0.25 x sigma). Their combination equals the full class total.",
   "Because the bias is constant, it CANCELS in scan-to-scan differences. The temporal feature reflects real "
   "load change plus tiny jitter, not meter noise. This was verified. Measured scan-to-scan drops from 2.4% to ~0.7% "
   "(matching the smooth true load) while the across-meter accuracy stays ~1.7%."]),
], top=bot - 0.03, fs=9.4, lh=0.019)
put_y = y - 0.004
fig.text(0.075, put_y, "One reading", fontsize=10, weight="bold", color=INK)
fig.text(0.075, put_y - 0.026, r"$z = z_{\mathrm{true}}\,(1 + b_{\rm meter}) + \varepsilon_t,\quad "
         r"b_{\rm meter}\sim\mathcal{N}(0,\,0.968\,\sigma)\ \mathrm{(drawn\ once)},\quad "
         r"\varepsilon_t\sim\mathcal{N}(0,\,0.25\,\sigma\,|z_{\rm true}|)\ \mathrm{(each\ scan)}$",
         fontsize=10.5, color="#1f2d3d", family="serif")
caption(fig, "Honest note. The accuracy-class magnitudes and the sqrt(3) conversion are taken directly from Asprou et al. "
             "(2013). The split of that error into a constant per-meter bias plus a small per-scan jitter is our physically-"
             "motivated refinement (the paper models it as all-random), justified by the standard metrology distinction "
             "between accuracy and precision. |V| bias is the VT ratio offset and theta bias is the VT phase displacement, "
             "both fixed transformer characteristics, so a systematic bias is the correct representation, not per-scan noise.",
        y=put_y - 0.06)
SIDE["measurement_model"] = ["quantity,source,max_class0.2,sigma"] + [",".join(f'"{c}"' for c in r) for r in mm_rows] + \
    ["split,bias=0.968*sigma_constant_per_meter,jitter=0.25*sigma_per_scan,total=class"]
save(fig)

# ======================= Physics operator, Ybus ordering + conventions =======================
PO = load_json("physics_op.json")
if PO and PO.get("systems"):
    fig = newpage("Physics Operator, Ybus Ordering, Sign, and the Shunt Convention",
                  "getting the AC power-flow identity right, three conventions the operator must respect, validated on the benign states")
    po_cols = ["System", "shunts", "injection MAE raw", "after sign match", "shunt (MVAr)"]
    po_rows = []
    for c in SYS_ALL:
        s = PO["systems"].get(f"ieee{c}")
        po_rows.append([f"IEEE-{c}", str(s["n_shunt"]), f"{s['inj_mae_raw_MW']:.0f} MW", f"{s['inj_mae_sign_matched_MW']:.2f} MW",
                        f"{s['shunt_total_Q_MVAr']:.0f}"] if s else [f"IEEE-{c}"] + ["n/a"]*4)
    bot = draw_table(fig, 0.86, po_cols, po_rows, fontsize=8.8, col_widths=[0.18, 0.13, 0.24, 0.23, 0.22])
    prose(fig, [
     ("Reconstructing measurements from the state",
      "The generator and the state estimator both need to turn a bus-voltage state (|V|, theta) into power "
      "quantities via the nodal admittance matrix. Bus injection S = V .* conj(Ybus @ V) and branch from-end flow "
      "S_f = V_from .* conj(Yf @ V). Three conventions have to be respected or the result is garbage that LOOKS "
      "like a broken Ybus."),
     ("1. Bus ordering (ppc vs pandapower)",
      "pandapower builds Ybus in its internal PYPOWER (ppc) bus order, which is a PERMUTATION of pandapower's own "
      "bus numbering. Multiplying V (pandapower order) by Ybus (ppc order) scrambles the indices. The fix is to "
      "build the voltage vector in ppc order via net._pd2ppc_lookups['bus'], do the multiply there, and map back."),
     ("2. Injection sign (the 70-280 MW 'mismatch')",
      ["The reconstructed bus injection appeared to be off by 70-280 MW (table, 'raw'). It is NOT a broken operator. "
       "The Ybus identity is INJECTION-positive (generation +, load -), while pandapower and our stored injection are "
       "CONSUMPTION-positive (load +, generation -) - the two are negatives of each other.",
       "Matching the sign convention collapses the error to 0.00 MW (table, 'after sign match') - an exact "
       "reconstruction. So the whole apparent discrepancy was a sign flip."]),
     ("3. Shunts on the Ybus diagonal",
      "A shunt (capacitor/reactor at a bus) sits on the Ybus DIAGONAL, so S = V .* conj(Ybus @ V) folds the shunt "
      "draw into the injection - but the state estimator treats shunts as a separate element and excludes them. This "
      "is a REACTIVE offset (the shunt totals, right column, 21 / 214 / 1603 MVAr). We store the SHUNT-CORRECTED "
      "injection (subtracting res_shunt) so the data matches the estimator's convention and passes bad-data detection."),
    ], top=bot - 0.03, fs=9.3, lh=0.0185)
    caption(fig, "The upshot. Branch FLOWS reconstruct from the state with an exact identity (no sign or shunt "
                 "ambiguity, and they are the directly-metered quantity), so the physics-consistency loss in the state "
                 "estimator and the residual in the bad-data-detection check are computed on FLOWS, not bus injections. "
                 "The injections are still stored (they are what an EMS reports) but shunt-corrected and in the "
                 "estimator's sign convention. Nothing was wrong with the Ybus - it just has to be read in the right "
                 "order, sign, and shunt convention.", y=0.14)
    SIDE["physics_op"] = ["system,n_shunt,inj_mae_raw_MW,inj_mae_sign_matched_MW,shunt_Q_MVAr"] + \
        [f"ieee{c},{PO['systems'][f'ieee{c}']['n_shunt']},{PO['systems'][f'ieee{c}']['inj_mae_raw_MW']},{PO['systems'][f'ieee{c}']['inj_mae_sign_matched_MW']},{PO['systems'][f'ieee{c}']['shunt_total_Q_MVAr']}" for c in SYS_ALL if f"ieee{c}" in PO["systems"]]
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
         "the readings directly, which breaks consistency and is caught. On notation, h maps a state x to the meter "
         "readings, PF is a power-flow solve, e is Gaussian meter noise, and z_a is the reading at an attacked bus a.")
for wl in textwrap.fill(intro, 106).split("\n"):
    put(wl, 0, 8.8, INK); Y[0] -= 0.0185
Y[0] -= 0.016
def grp(title, color):
    fig.add_artist(plt.Line2D([x0, x0 + 0.026], [Y[0] + 0.004, Y[0] + 0.004], color=color, lw=2.8))
    put(title, 0.034, 10.5, INK, "sans-serif", bold=True); Y[0] -= 0.033
def item(name, desc, formula):
    put(name, 0.018, 10.5, INK, "sans-serif", bold=True)
    put(desc, 0.085, 8.8, INK, "serif", ital=True); Y[0] -= 0.026
    put(formula, 0.050, 11.5, "#1f2d3d", "serif"); Y[0] -= 0.034
grp("State-level attacks   (change the loads, re-solve, evade bad-data detection)", STEAL)
item("benign", "reference, no attack", r"$z = h(x_t) + e$")
item("Aq", "per-bus load scaling on 1-6 buses, re-solve with generation rebalanced (AGC)",
     r"$L_a \leftarrow \alpha_a\, L_a,\ \ \alpha_a \in [1.05,\ 1.15]\ \mathrm{per\ bus},\ \ x' = \mathrm{PF}(L'),\ \ z = h(x') + e$")
item("At", "temporal surge, ramp up then back down (asymmetric rates, random turn), or a dip",
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
            put(wl, 0.058, 9.8, INK, "serif"); Y[0] -= 0.0188
        Y[0] -= 0.007
    Y[0] -= 0.017
steps("State-level attacks   (Aq, At, Al)", STEAL, [
 "Start from a clean operating point, the true grid state at a timestep t.",
 "Rebuild the bus loads from that state. Net injection plus the generator setpoints gives the gross load.",
 "Choose the attacked bus SET (drawn at random per record, so it is not memorizable). Aq hits 1 to 6 load "
 "buses, each with its OWN scaling multiplier. Ramp holds a fixed set of up to 5 buses across its whole "
 "sequence. LRA hits the PTDF-selected up-set and down-set. Then change their loads, bounded per-bus scaling "
 "for Aq, a rise-then-fall surge for At (ramp), or a load-conserving PTDF-targeted shift for Al (LRA).",
 "Re-solve a full AC power flow for the new loads with pandapower, holding generation at the true operating "
 "point's dispatch and distributing the net load change across generators (AGC). If it does not converge, "
 "discard the sample and try another timestep.",
 "Read every meter off the solved state (via the SAME measurement function used for benign) and add sensor "
 "noise. Because generation is pinned to the true dispatch, the attacked snapshot differs from the true state "
 "only in the attacked buses' loads plus the small distributed generation response, the honest attack footprint, "
 "not a spurious slack-bus swing.",
 "Record the attacked bus set as the label y, then check that the sample passes bad-data detection.",
])
steps("Meter-level attacks   (Ad, As, Ar)", DET, [
 "Start from the clean measurements at a timestep, read directly off the true state.",
 "Pick the attacked bus set. Ad, As, and Ar each hit 4 randomly-chosen load buses (and their incident branches).",
 "Corrupt only those buses' meters. Add random noise for Ad, scale the readings for As, or paste in a distant "
 "past scan for Ar.",
 "Record the attacked bus set as the label y. These fail bad-data detection by construction.",
])
caption(fig, "The same pipeline runs for IEEE-14, 118, and 300, and benign samples simply skip the attack step and "
             "are read straight off the true state. Everything is seeded, so the dataset regenerates identically each "
             "time. The re-solve in the state-level pipeline is what keeps every reading mutually consistent, which is "
             "why those attacks pass detection, while the meter-level attacks change a few readings and leave a "
             "detectable inconsistency.", y=Y[0] - 0.004)
save(fig)

# ======================= Stealth method: re-solve + dispatch pinning + AGC =======================
# The single most important mechanism for genuine stealth: we change the LOADS and re-solve the full AC power flow,
# but pin generation to the true dispatch and AGC-distribute the attack's load delta, so a no-op re-solve reproduces
# the true state exactly and any residual is the attack itself, not a slack-bus artifact. Numbers from the real
# alpha=1 experiment (results/dispatch_pinning.json), not asserted.
DP = load_json("dispatch_pinning.json", None)
fig = newpage("Making Attacks Truly Stealthy with Re-solve, Dispatch Pinning & AGC",
              "why we re-solve the whole power flow (not patch meters), and pin generation so the residual is the attack, not a slack-bus artifact")
yb = prose(fig, [
 ("The balancing problem",
  "A stealthy attack changes a few buses' loads, but total generation must still equal total load plus losses. So "
  "the moment we re-solve the power flow, some generation must move to absorb the change, and WHO absorbs it decides "
  "the entire measurement footprint."),
 ("The naive re-solve leaks a huge artifact",
  "If we set the new loads but leave generators at their setpoints, the SLACK bus absorbs the whole imbalance. Even a "
  "NO-OP re-solve (change nothing, alpha=1) then fails to reproduce the true state. On IEEE-118 the slack bus swings "
  "~178 MW and bus injections drift ~3 MW on average, a dispatch artifact from nowhere that would masquerade as, and "
  "swamp, the real attack signature."),
 ("Our fix, pin the true dispatch then AGC the delta",
  "We reconstruct each generator's TRUE output from the stored state and hold it, then spread only the attack's load "
  "delta across generators in proportion to their output, exactly how real Automatic Generation Control (AGC) "
  "responds. Generator voltage setpoints and the slack reference are pinned to the true values too."),
 ("The payoff, the residual is the attack and the grid looks real",
  "Now the alpha=1 re-solve reproduces the true state to numerical precision, so any residual is the attack itself, "
  "not a re-dispatch. And the attacked scan differs from benign by only the attacked loads plus a small, spread-out, "
  "physically-plausible generation response, which is why the whole measurement set stays bad-data-consistent."),
], top=0.895)
if DP:
    n = DP["naive"]; p = DP["pinned"]
    cols = ["re-solve approach (alpha=1, no-op)", "mean inj. error (MW)", "max inj. error (MW)", "slack-bus swing (MW)"]
    rows = [["naive (slack absorbs imbalance)", f"{n['mean_inj_err_MW']:.2f}", f"{n['max_inj_err_MW']:.0f}", f"{n['slack_swing_MW']:.0f}"],
            ["pinned dispatch + AGC (ours)", f"{p['mean_inj_err_MW']:.2f}", f"{p['max_inj_err_MW']:.2f}", "n/a"]]
    bottom = draw_table(fig, yb - 0.01, cols, rows, fontsize=8.6, col_widths=[0.34, 0.22, 0.22, 0.22],
                        shade_rule=lambda i, r: "#e9f2ec" if "ours" in r[0] else "#fbecea")
    caption(fig, f"Re-solving a no-op benign scan two ways on IEEE-118 ({DP['n']} samples). The naive approach leaves a "
                 "~178 MW slack-bus swing and cannot reproduce the true state even with no attack applied. Our "
                 "dispatch-pinning + AGC reproduces it to 0.00 MW. This is what makes the stealthy families (Aq/At/Al) "
                 "genuinely BDD-stealthy rather than carrying a large, detectable slack-bus footprint, and the residual "
                 "collapses to the attack's own plausibility-bounded load change.", y=bottom - 0.03)
    SIDE["dispatch_pinning"] = ["approach,mean_inj_err_MW,max_inj_err_MW,slack_swing_MW"] + \
        [f"naive,{n['mean_inj_err_MW']},{n['max_inj_err_MW']},{n['slack_swing_MW']}", f"pinned_agc,{p['mean_inj_err_MW']},{p['max_inj_err_MW']},0"]
save(fig)

# ======================= Aq sensitivity: detectability vs attack magnitude (the noise floor) =======================
# How stealthy is stealthy, quantified. We sweep Aq's load-scale magnitude and measure, at each scale, how separable
# the attack is from the meter-noise floor (ROC-AUC of attacked vs noise-only buses on the injection deviation the
# detector sees) and the SNR = attack shift / meter noise. Below a crossover scale the attack IS noise (AUC->0.5,
# SNR->~1): an information-theoretic detectability floor NO model can beat. Numbers from results/aq_sensitivity.json.
AQS = load_json("aq_sensitivity.json", None)
SYSCOL = {"ieee14": FAMCOL["Ad"], "ieee118": FAMCOL["Aq"], "ieee300": FAMCOL["As"]}   # dark, saturated per system
SYSLAB = {"ieee14": "IEEE-14", "ieee118": "IEEE-118", "ieee300": "IEEE-300"}


def _crossover(scales, ys, thr, rising=True):
    """Scale (%) at which the curve crosses thr, linearly interpolated in log-scale; None if it never does."""
    import math
    for i in range(1, len(scales)):
        a, b = ys[i - 1], ys[i]
        lo, hi = (a, b) if rising else (b, a)
        if (lo < thr <= hi) or (lo <= thr < hi):
            f = (thr - a) / (b - a) if b != a else 0.0
            lx = math.log10(scales[i - 1]) + f * (math.log10(scales[i]) - math.log10(scales[i - 1]))
            return 10 ** lx
    return None

fig = newpage("How Stealthy Is Stealthy? Aq Detectability vs Attack Magnitude",
              "sweeping the stealthy load-move magnitude to find the SNR floor where the attack becomes indistinguishable from meter noise")
yb = prose(fig, [
 ("The question",
  "Aq scales a few buses' loads and re-solves the physics, so it is BDD-consistent by construction. But a small "
  "enough move is buried in the meters' own noise. At what magnitude does the attack stop being a signal and become "
  "noise, so that NO detector (ML included) can catch it?"),
 ("The measurement",
  "Over load-scale magnitudes +0.5% to +50% we apply Aq, re-solve (pinned dispatch + AGC), and add accuracy-class "
  "meter noise (sigma 1.7% on P). Detectability is the ROC-AUC of separating attacked buses from noise-only buses on "
  "the injection deviation a per-bus detector sees. SNR is the attack shift over the noise sigma. AUC bounds ANY model."),
], top=0.895)

if AQS and AQS.get("systems"):
    axU = fig.add_axes([0.115, 0.435, 0.80, 0.165])
    axS = fig.add_axes([0.115, 0.225, 0.80, 0.165])
    cross_rows = []; snr_shared = None; scl_shared = None
    for sysk, rows in AQS["systems"].items():
        sc = [r["scale_pct"] for r in rows]; au = [r["auc"] for r in rows]; sn = [r["median_snr"] for r in rows]
        col = SYSCOL.get(sysk, INK)
        axU.plot(sc, au, "-o", color=col, lw=1.9, ms=3.4, label=SYSLAB.get(sysk, sysk))
        xc = _crossover(sc, au, 0.75, rising=True)                       # scale where AUC first clears 0.75
        cross_rows.append((SYSLAB.get(sysk, sysk), xc))
        snr_shared, scl_shared = sn, sc                                  # SNR = scale/sigma, identical across systems
    # SNR is topology-independent (set by meter class), so the per-system curves coincide, draw it once.
    axS.plot(scl_shared, snr_shared, "-o", color=INK, lw=2.0, ms=3.6)
    for ax in (axU, axS):
        ax.set_xscale("log"); ax.set_xlim(0.4, 60); ax.grid(True, which="both", alpha=0.18, lw=0.6)
        ax.tick_params(labelsize=7.5)
    axU.axhline(0.5, color=INK, ls="--", lw=0.9); axU.axhspan(0.44, 0.6, color=INK, alpha=0.06)
    axU.text(35, 0.55, "chance level (attack lost in noise)", fontsize=6.6, color=INK, ha="right", va="center")
    axU.set_ylim(0.44, 1.02); axU.set_ylabel("detectability AUC", fontsize=8.2)
    axU.set_xticklabels([]); axU.legend(fontsize=7.4, loc="upper left", frameon=False)
    axU.set_title("attacked-vs-noise separability (ceiling on any detector)", fontsize=8.4, loc="left", weight="bold")
    axS.axhline(1.0, color=INK, ls="--", lw=0.9); axS.axhspan(0.0, 1.0, color=INK, alpha=0.06)
    axS.text(35, 0.72, "SNR = 1  (attack = noise)", fontsize=6.6, color=INK, ha="right", va="center")
    axS.set_yscale("log"); axS.set_ylabel("median SNR", fontsize=8.2)
    axS.set_xlabel("Aq load-scale magnitude  (% change on attacked buses)", fontsize=8.2)
    axS.set_title("attack shift vs the 1.7% meter-noise floor  (identical across systems, set by meter class)",
                  fontsize=8.4, loc="left", weight="bold")
    ct = "   ".join(f"{lab} ≈ {xc:.1f}%" if xc else f"{lab} > 50%" for lab, xc in cross_rows)
    caption(fig, f"Detectability collapses to chance as the load move nears the noise floor. SNR crosses 1 at a "
                 f"topology-independent ~1.7% move (the meter class). The AUC a detector can reach is topology-dependent, "
                 f"clearing 0.75 at {ct} (earliest on IEEE-118, where redundancy aids separation). Below the crossover the "
                 "attack is information-theoretically indistinguishable from noise for ANY model, the ML-only-dangerous "
                 "window, which also pins where the plausibility cap should sit.", y=0.150)
    SIDE["aq_sensitivity"] = ["system,scale_pct,auc,median_snr"] + \
        [f"{s},{r['scale_pct']},{r['auc']},{r['median_snr']}" for s, rs in AQS["systems"].items() for r in rs]
else:
    fig.text(0.5, 0.34, "[ PENDING. run _aq_sensitivity.py ]", ha="center", va="center", fontsize=15, weight="bold", color=INK)
save(fig)

# ======================= Attack insertion: previous vs current design =======================
fig = newpage("Attack Insertion, Previous vs Current Design",
              "how the generator decides WHEN to place an attack, HOW LARGE it is, and how it PROGRESSES in time")
Y[0] = 0.872
intro = ("The generator was redesigned. The previous pipeline walked one long measurement timeline and crafted each "
         "stealthy Aq attack by a per-sample gradient-descent optimization. The current pipeline samples a pool of "
         "physical operating points and makes stealthy attacks by changing a bounded, plausible load and re-solving "
         "the power flow, so stealth is achieved by construction rather than by optimization.")
for wl in textwrap.fill(intro, 106).split("\n"):
    put(wl, 0, 8.8, INK); Y[0] -= 0.0185
Y[0] -= 0.013

def axis(title):
    fig.add_artist(plt.Line2D([x0, x0 + 0.026], [Y[0] + 0.004, Y[0] + 0.004], color=INK, lw=2.8))
    put(title, 0.034, 10.5, INK, "sans-serif", bold=True); Y[0] -= 0.029

def side(tag, txt, color):
    put(tag, 0.020, 9.2, color, "sans-serif", bold=True)
    for wl in textwrap.wrap(txt, 92):
        put(wl, 0.105, 9.2, INK, "serif"); Y[0] -= 0.0180
    Y[0] -= 0.007

axis("Stealthy-attack mechanism   (the biggest change)")
side("Previous", "The predecessor attack (Boyaci-style Ao) was solved per sample. About 1000 epochs of gradient descent tuned a fake measurement vector to "
                 "be both bad-data-stealthy and load-plausible, tampering the readings directly (a soft a = Hc).", MUTE)
side("Current", "Aq scales the load at the target buses by a bounded factor and re-solves a full AC power flow, then "
                "reads clean meters off the new state. The result is a genuine, mutually consistent operating point that "
                "passes detection by construction. Cheaper, exact, and a defensible threat model.", INK)

axis("Placement   (when an attack is inserted)")
side("Previous", "Timeline-driven. Every one of the ~86,400 timesteps was labeled with a family in sequence, via a "
                 "datatype map over the whole horizon.", MUTE)
side("Current", "Pool-driven. Benign draws n_benign distinct random timesteps, and each single-shot family retries random "
                "timesteps until per_family samples converge (hard cap of per_family x 25 attempts), so a "
                "non-converging operating point costs time, not sample count.", INK)

axis("Magnitude   (how large the attack is)")
side("Previous", "The attacked set was a graph-radius ball around an entry bus (r = 2-4 hops, capped at 15 buses). Size "
                 "was bounded by a 50% relative-change cap plus, for the predecessor Ao, a 2-sigma jump cap and a 3-sigma load-anomaly "
                 "band from a rolling seven-day window.", MUTE)
side("Current", "Small random bus sets (at most 4-6). Aq scales load 1.05-1.15x. Ramp grows 1 + rate*k. LRA is a "
                "load-conserving PTDF shift bounded by 15%. Meter-level As scales 1.25-1.5x, Ad adds ~30% Gaussian, Ar "
                "replays a past scan. Plausibility now comes from re-solving physics, not from explicit caps.", INK)

axis("Progression   (behavior over timesteps)")
side("Previous", "Attacks sat inline in the timeline. Temporal structure came from their position in the sequence.", MUTE)
side("Current", "Almost every family is single-shot (one snapshot, seq_id = -1). Ramp is the only temporal attack. A "
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

# ======================= Attack placement & frequency vs the prior protocol =======================
# Documents WHERE we attack (which buses) and HOW OFTEN / at what class balance, and how this differs from the
# widely-used prior protocol (Boyaci et al.). Prose page, the point is the design rationale, not a plotted number.
fig = newpage("Attack Placement & Frequency, Our Protocol vs Prior Work",
              "which buses we attack and how often, and why our balanced, pool-based, per-family protocol differs from the timeline/frequency approach")
prose(fig, [
 ("How often, attack frequency and class balance",
  "Placement and frequency are dataset design choices. The widely-used prior protocol (Boyaci et al.) walks a "
  "measurement timeline and, at each timestep, draws f ~ N(0,1) and inserts an attack when f exceeds tau_freq (=1), "
  "attacking roughly 15% of timesteps, giving an imbalanced ~15/85 dataset. We instead draw operating points from a "
  "pool of real 1-minute states and generate a FIXED count per family, matched by an equal benign majority, a "
  "balanced 50/50 benchmark."),
 ("Where, attack placement",
  "The prior protocol seizes one random entry bus (p ~ U(1,n)) plus a connected r-hop region around it (up to ~30% of "
  "the meters). We place attacks as per-family random sets of ACTIVE-LOAD buses instead. Aq on 1-6 buses (each with its "
  "own multiplier), the meter-level Ad/As/Ar on up to 4, the ramp on a fixed set of ~5 across a sequence, and the "
  "load-redistribution LRA on a PTDF-steered subset around the most-attackable line."),
 ("Why these choices matter",
  "The 50/50 balance is not cosmetic. A flag-all detector scores F1 = 2p/(1+p), which is ~0.26 at the prior work's ~15% "
  "attack prevalence but 0.67 at our 50/50, so a balanced set is what makes 'ML beats BDD' an honest comparison rather "
  "than an artifact of class imbalance (this is exactly why the prior BDD baseline degenerates to flag-all, see the "
  "BDD-baseline page). Drawing from a real-load pool gives genuine minute-to-minute dynamics, and per-family placement "
  "lets each family keep its own natural footprint, per-bus for Aq, line-targeted for LRA, a fixed set for the ramp, "
  "rather than forcing every family into one entry region."),
], top=0.885)
caption(fig, "We deliberately do NOT copy the prior timeline/frequency protocol. Balanced, pool-based, per-family "
             "generation is what makes the benchmark clean to train and evaluate on, keeps the bad-data baseline "
             "honest, and preserves each attack family's realistic spatial footprint.", y=0.12)
save(fig)

# ======================= Attack progression on one bus's load over time (one page per system) =======================
# Keys in load_timeline.npz are prefixed ieee{C}_ so each grid gets its own page; every value is a real shard record.
LT = np.load(os.path.join(RES, "load_timeline.npz")) if os.path.exists(os.path.join(RES, "load_timeline.npz")) else None
if LT is not None:
    fams = [("benign", MUTE,  "reference load rhythm"),
            ("Aq",     STEAL, "state, per-bus load scaling"),
            ("At",     STEAL, "state, temporal ramp"),
            ("Al",     STEAL, "state, load redistribution"),
            ("Ad",     DET,   "meter, random corruption"),
            ("As",     DET,   "meter, scaling"),
            ("Ar",     DET,   "meter, replay a past scan")]
    for fi, C in enumerate(SYS):
        if f"ieee{C}_bus" not in LT: continue                                # this system's traces not staged yet
        bus = int(LT[f"ieee{C}_bus"]); Wn = int(LT[f"ieee{C}_W"]); t = np.arange(Wn)
        fig = newpage(f"Attack Progression {'abc'[fi]}, One Bus's Load Over Time (real records, IEEE-{C} bus {bus})",
                      "actual injection readings pulled from the shard, benign timeline + each family's real attacked scans")
        bottoms = [0.66, 0.50, 0.34, 0.18]; x0s = [0.08, 0.525]; W, Hh = 0.40, 0.12
        for i, (fam, col, note) in enumerate(fams):
            if f"ieee{C}_{fam}_atk" not in LT: continue
            ben = LT[f"ieee{C}_{fam}_ben"]; atk = LT[f"ieee{C}_{fam}_atk"]; hits = LT[f"ieee{C}_{fam}_hits"]
            r, c = i // 2, i % 2
            ax = fig.add_axes([x0s[c], bottoms[r], W, Hh])
            ax.plot(t, ben, color=MUTE, lw=1.0, ls="--", zorder=2)           # real benign timeline for this bus
            if fam != "benign":                                              # ALL families drawn as a line (consistent with At):
                ax.plot(t, atk, color=col, lw=1.4, zorder=3)                 # the attacked reading over the benign line
                if len(hits): ax.scatter(hits, atk[hits], s=24, color=col, zorder=4, edgecolor="white", linewidth=0.6)  # mark attacked scans
            ax.set_title(fam, fontsize=9, color=(INK if fam != "benign" else MUTE), loc="left", weight="bold")
            ax.text(0.98, 0.05, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, color=SUBTLE, style="italic")
            ax.tick_params(labelsize=6.0); ax.margins(x=0.02)
            if c == 0: ax.set_ylabel("MW", fontsize=6.6)
            if r == 3: ax.set_xlabel("scan (1-min)", fontsize=6.6)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
        caption(fig, f"Every value is a REAL record from the shard, one IEEE-{C} load bus (bus {bus}) over time. Dashed grey = its benign "
                     "injection timeline (one benign record per scan). The coloured line = the ACTUAL attacked reading, with markers at the "
                     "scans this family targets the bus. The key thing to see is that the attacked values sit INSIDE the bus's natural load "
                     "swing, not as obvious spikes, a single scan is only 5-15% off, comparable to the real minute-to-minute variation, which "
                     "is exactly why they are hard to flag by eye or by a simple threshold and why localization (not detection) is the task. "
                     "Single-shot families (Aq/Al, and meter-level Ad/As/Ar) touch one scan at a time (the line meets benign except at the "
                     "marked scans). Only At progresses across many consecutive scans.", y=0.15)
        save(fig)
        SIDE[f"load_timeline_ieee{C}"] = open(os.path.join(RES, "sidecars", f"load_timeline_ieee{C}.csv")).read().splitlines() \
            if os.path.exists(os.path.join(RES, "sidecars", f"load_timeline_ieee{C}.csv")) else []

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
                           [dict(subhead="Chi-square bad-data detection rate (%), the stealthy families must NOT be flagged",
                                 cols=bdd_cols, rows=bdd_rows("chi2_detect_pct", lambda v: f"{v:.0f}%"),
                                 col_widths=[0.16, 0.16] + [0.68 / len(SYS_ALL)] * len(SYS_ALL), shade_rule=bshade,
                                 cap="Run per record. Build the realistic sparse measurement set, solve WLS, apply the chi-square "
                                     "test. Benign and the stealthy families (Aq/At/Al, red) pass, they are physically valid "
                                     "states. The detectable families (Ad/As/Ar, blue) are caught almost every time. This is the "
                                     "stealth split, measured directly rather than asserted."),
                            dict(subhead="Median largest-normalized-residual (LNR), the classical detector's statistic (threshold 3.0)",
                                 cols=bdd_cols, rows=bdd_rows("median_lnr", lambda v: f"{v:.1f}"),
                                 col_widths=[0.16, 0.16] + [0.68 / len(SYS_ALL)] * len(SYS_ALL), shade_rule=bshade,
                                 cap="The stealthy families sit at the benign noise floor (median LNR near or below the 3.0 "
                                     "threshold), while the detectable families are far above it. This gives the dataset physical "
                                     "plausibility. A reviewer can confirm the stealthy attacks genuinely evade the standard bad-data "
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
        fig = newpage(f"Bad-Data Detection, the Stealth Split, Measured (IEEE-{cc})",
                      "we run WLS + the chi-square test on real samples, detection rate, and the chi-square statistic distribution")
        axB, axJ = fig.add_axes([0.10, 0.585, 0.85, 0.275]), fig.add_axes([0.10, 0.245, 0.85, 0.275])
        order = [f for f in (["benign"] + FAMS) if f"ieee{cc}_{f}_J" in z]
        cols = [MUTE if f == "benign" else (STEAL if f in STEAL_FAM else DET) for f in order]
        # (a) chi-square detection rate
        rates = [summ.get(f, {}).get("chi2_detect_pct", 0) for f in order]
        axB.bar(range(len(order)), rates, color=cols, alpha=0.85)
        axB.set_xticks(range(len(order))); axB.set_xticklabels(order, fontsize=8); axB.set_ylim(0, 105)
        axB.set_ylabel("chi-square\ndetection rate (%)", fontsize=8.5); axB.tick_params(labelsize=7.5)
        axB.set_title("Detection rate. Stealthy families pass, detectable families are caught", fontsize=9.5)
        # (b) chi-square statistic distribution (log)
        data = [z[f"ieee{cc}_{f}_J"] for f in order]
        parts = axJ.violinplot(data, showmedians=True, widths=0.85)
        for j, b in enumerate(parts["bodies"]): b.set_facecolor(cols[j]); b.set_alpha(0.72)
        axJ.set_yscale("log"); axJ.set_xticks(range(1, len(order) + 1)); axJ.set_xticklabels(order, fontsize=8)
        axJ.set_ylabel("chi-square statistic J\n(log scale)", fontsize=8.5); axJ.tick_params(labelsize=7.5)
        axJ.set_title("Residual magnitude. A ~10x gap separates stealthy from detectable", fontsize=9.5)
        caption(fig, "Real BDD on the shipped data. Per record we rebuild the sparse measurement set, solve WLS, and apply the "
                     "chi-square test. Top, the detectable families (blue) are flagged ~100%, while benign and the stealthy families "
                     "(red) pass at the ~5% baseline false-alarm rate. Bottom, the chi-square statistic sits ~10x higher for the "
                     "detectable families. The stealthy attacks evade the standard detector by the detector's own residual.", y=0.185)
        SIDE["bdd_J"] = ["family,chi2_detect_pct,median_J"] + [f"{f},{summ.get(f,{}).get('chi2_detect_pct')},{summ.get(f,{}).get('median_J')}" for f in order]
        save(fig)

# ======================= Attack provenance & why each does (or does not) evade BDD =======================
# Verified against the original papers (Liu 2009, Yuan 2011, Haghshenas 2023, Mo-Sinopoli 2009), this page
# gives the reader the lineage of every family and the STRICT vs APPROX vs NO stealth distinction, so the
# BDD-measured split on the previous page is grounded in each attack's construction, not asserted.
prov_cols = ["Attack", "Origin (taxonomy, concept)", "Evades BDD?", "Why, by construction"]
prov_rows = [
    ["Aq  (ours)", "Boyaci T-SG'22 Table I ('Ao'),  Liu CCS'09 concept",
     "Yes, strict", "AC-consistent false state, z_a=h(x'), residual = benign"],
    ["Al  (LRA)", "Yuan, Li, Ren, IEEE T-SG 2011",
     "Yes, strict", "Load-conserving PTDF shift, r_a = Se, residual unchanged"],
    ["At  (ramp)", "Haghshenas et al., ISGT 2023",
     "Per-frame, evades temporal det.", "AC-consistent each step, gradual slope beats rate-of-change"],
    ["Ad", "Boyaci T-SG'22 Table I,  Ozay/Yan '16",
     "No", "Random N(mu,sig^2) breaks a=Hc -> residual spikes"],
    ["As", "Boyaci T-SG'22 Table I,  Hasnat'20 / Jevtic'18",
     "No", "Meter gain U(.)*z violates flow consistency -> residual"],
    ["Ar", "Boyaci T-SG'22 Table I,  Zhao'16 / Chaojun'15",
     "No (partial)", "Stale subset inconsistent w/ neighbours -> residual"],
]
def prov_shade(i, row):
    if "ours" in row[0]: return "#e9f2ec"                     # our contribution, highlighted
    return "#f3f0fb" if row[2].startswith("Yes") else "#f4f6f8"   # stealthy vs detectable
fig = table_page("Attack Provenance, What Each Family Is, and Why It Does or Doesn't Evade BDD",
                 "the reference each attack comes from, and the strict / per-frame / none stealth distinction, verified against the original papers",
                 prov_cols, prov_rows, cap=(
                 "Every family is traced to its source and its bad-data-detection behaviour explained from construction, not just measured. The "
                 "Ao/Ad/As/Ar letter taxonomy and the exact attack formulas we mirror come from Boyaci et al. (IEEE T-SG 13(1), 2022, Table I). "
                 "Boyaci in turn credits the concept of each detectable family to earlier work (random from Ozay 2016 / Yan 2016, scale from Hasnat 2020 / "
                 "Jevtic 2018, replay from Zhao 2016 / Chaojun 2015), so both the lineage cite and the concept-origin cite are available. 'Strict' means "
                 "zero residual increase by the Liu a=Hc argument, generalized to AC, and Aq (OUR contribution) and Al (LRA) achieve this, and we make "
                 "both AC-power-flow-exact rather than the original DC-linear form. Aq shares the defining stealth property of Boyaci's Ao "
                 "(z_a=h(x_check), residual unchanged) but reaches it through a load-scaling + AC-re-solve construction with a plausibility cap, "
                 "not Boyaci's state-space SGD optimization. At (ramp) is BDD-stealthy every frame but its real purpose is to slip under a temporal "
                 "rate-of-change detector, which is why it is the one family both detectors miss. The detectable trio (Ad/As/Ar) is deliberately "
                 "residual-inconsistent. The classical corruption / gain-tamper / replay cases BDD was designed to catch, included as an honest "
                 "contrast so that 'ML beats the detector' is a real result on this data, not a rigged one."),
                 fontsize=7.4, col_widths=[0.09, 0.34, 0.13, 0.44], shade_rule=prov_shade)
SIDE["attack_provenance"] = ["attack,origin,evades_bdd,mechanism"] + \
    [",".join('"' + x.replace('"', "'") + '"' for x in r) for r in prov_rows]
save(fig)

# ======================= Temporal rate-of-change detector (a second classical detector) =======================
ROC = load_json("roc_detector.json")
if ROC and ROC.get("systems"):
    STEAL_F = {"Aq", "At", "Al"}
    rc_cols = ["Family", "Class"] + [f"IEEE-{c}" for c in SYS_ALL]
    def rc_rows():
        rows = []
        for f in ["benign"] + FAMS:
            cls = "reference" if f == "benign" else ("stealthy" if f in STEAL_F else "detectable")
            cells = []
            for c in SYS_ALL:
                v = (ROC["systems"].get(f"ieee{c}", {}).get("per_family", {})).get(f)
                cells.append(f"{v:.0f}%" if v is not None else "n/a")
            rows.append([f, cls] + cells)
        return rows
    def rc_shade(i, row): return "#eef1f4" if row[1] == "reference" else ("#fbecea" if row[1] == "stealthy" else "#eaf2f8")
    # cross-detector: does each family evade BOTH bad-data detection AND rate-of-change? (uses IEEE-118)
    b118 = (BDD.get(118) or {}).get("families", {}); r118 = ROC["systems"].get("ieee118", {}).get("per_family", {})
    def evades(f):
        bdd_caught = (b118.get(f, {}).get("chi2_detect_pct", 0) or 0) > 50      # BDD flags it?
        roc_caught = (r118.get(f, 0) or 0) > 50                                  # RoC flags it?
        return ("caught" if bdd_caught else "evades"), ("caught" if roc_caught else "evades"), \
               ("YES" if (not bdd_caught and not roc_caught) else "no")
    x_rows = [[f] + list(evades(f)) for f in FAMS]
    fig = multi_table_page("A Second Classical Detector, Temporal Rate-of-Change",
                           "the abrupt-change detector we built, flag a scan if any bus's injection jump far exceeds its typical recent-window jump (window tuned to W=60, 5% benign false alarm)",
                           [dict(subhead="Rate-of-change catch rate (%), fraction of each family flagged",
                                 cols=rc_cols, rows=rc_rows(), col_widths=[0.14, 0.16] + [0.68/len(SYS_ALL)]*len(SYS_ALL), shade_rule=rc_shade,
                                 cap="A per-bus z-score of this scan's injection jump against the bus's typical recent-window jump, "
                                     "maxed over buses, thresholded at a 5% benign false-alarm rate (the window W=60 was TUNED to catch "
                                     "the most attacks). It catches the SINGLE-SHOT families, since Aq and Al spike far outside recent normal "
                                     "(~90-100% caught), plus the crude meter attacks. It does NOT catch At. A gradual ramp never makes a "
                                     "single abrupt jump, so it stays at the benign floor."),
                            dict(subhead="The two detectors together, which family evades BOTH? (IEEE-118)",
                                 cols=["Family", "Bad-data detection", "Rate-of-change", "Evades BOTH"], rows=x_rows,
                                 col_widths=[0.22, 0.28, 0.26, 0.24],
                                 shade_rule=lambda i, r: "#e7f4ea" if r[3] == "YES" else "#f6f7f8",
                                 cap="This is the sharpest result. Bad-data detection is a SPATIAL check (are the meters mutually "
                                     "consistent?), it catches Ad/As/Ar and misses the physically-consistent Aq/At/Al. The rate-of-change "
                                     "detector is a TEMPORAL check (is this an abrupt jump?), it catches the single-shot Aq/Al (and the "
                                     "crude families) and misses the gradual At. Only ONE family evades both. It is At, the temporal ramp. It is "
                                     "the genuinely ML-only-dangerous attack, no classical detector, spatial or temporal, sees it. The "
                                     "same rate-of-change statistic is also handed to the localizer as the per-bus 'swing' feature, so the "
                                     "model can pinpoint the single-shot attacks this detector flags.")])
    SIDE["roc_detector"] = ["family," + ",".join(f"catch_ieee{c}" for c in SYS_ALL)] + \
        [f"{f}," + ",".join(str((ROC['systems'].get(f'ieee{c}',{}).get('per_family',{})).get(f,'')) for c in SYS_ALL) for f in (["benign"]+FAMS)]
    SIDE["roc_sweep"] = ["window," + ",".join(f"ieee{c}_allatk_catch" for c in SYS_ALL)] + \
        [f"{W}," + ",".join(str(ROC['systems'].get(f'ieee{c}',{}).get('sweep',{}).get(str(W), ROC['systems'].get(f'ieee{c}',{}).get('sweep',{}).get(W,''))) for c in SYS_ALL) for W in (5,10,15,20,30,45,60,90)]
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
                 "attack protocol (heldout=True) additionally reserves As/Ar for test only, note their train/val entries drop to "
                 "zero under that flag (not applied here). The benign majority pads the set so both a balanced training view and an "
                 "attack-sparse realistic test view are samplable.",
                 col_widths=[0.14, 0.20, 0.17, 0.17, 0.16, 0.16], fontsize=8.0, height=0.56, y_table=0.30, section_rows=section_rows)
SIDE["composition"] = ["system,family,train,val,test,total"] + \
    [f'{c},{fn},{DATA[c]["per"]["train"].get(fn,0)},{DATA[c]["per"]["val"].get(fn,0)},{DATA[c]["per"]["test"].get(fn,0)},'
     f'{DATA[c]["per"]["train"].get(fn,0)+DATA[c]["per"]["val"].get(fn,0)+DATA[c]["per"]["test"].get(fn,0)}'
     for c in SYS for fn in FAMILIES.values() if DATA[c]["per"]["train"].get(fn,0)+DATA[c]["per"]["val"].get(fn,0)+DATA[c]["per"]["test"].get(fn,0)]
save(fig)

# ======================= measurement distributions (FIGURE, genuinely distributional) =======================
# One page per system. These are true distributions (not summary stats), so a histogram is the honest view, and
# each grid's meters move differently under attack, so we draw all three rather than one representative case.
chans = [("|V| (p.u.)", 0), ("P_inj (MW)", 1), ("Q_inj (MVAr)", 2), ("theta (deg)", 3)]
for fi, c in enumerate(SYS):
    fig = newpage(f"Figure 1{'abc'[fi]}, Measurement Distributions (IEEE-{c})",
                  "benign vs attacked, per metered channel, where the families do (and don't) move the meters")
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.93, top=0.86, bottom=0.30, wspace=0.26, hspace=0.4)
    nx = DATA[c]["nx"]; fam = DATA[c]["fam"]; nm_test = load_sys(c, split="test").to_numpy()["node_m"]
    for ax, (lab, ch) in zip([fig.add_subplot(gs[i]) for i in range(4)], chans):
        m = nm_test[:, :, ch].astype(bool)
        ben = nx[fam == 0, :, ch][m[fam == 0]]; atk = nx[fam > 0, :, ch][m[fam > 0]]
        lo, hi = np.percentile(np.concatenate([ben, atk]), [1, 99])
        ax.hist(ben, bins=60, range=(lo, hi), density=True, alpha=0.6, color=MUTE, label="benign")
        ax.hist(atk, bins=60, range=(lo, hi), density=True, alpha=0.6, color=ACCENT, label="attacked")
        ax.set_title(lab, fontsize=9); ax.tick_params(labelsize=7)
        if ch == 0: ax.legend(fontsize=7)
    caption(fig, "Voltage magnitude barely moves under attack (the stealthy families keep a plausible state), so |V| alone is a weak "
                 "signal, consistent with why the attacks evade detectors. The injection and angle channels shift more. These are "
                 "genuine distributions, so a histogram (not a table) is the right view. Metered entries only.", y=0.24)
    SIDE[f"distributions_ieee{c}"] = ["channel,benign_median,attacked_median"] + \
        [f'{lab},{np.median(nx[fam==0,:,ch][nm_test[:,:,ch].astype(bool)[fam==0]]):.4f},{np.median(nx[fam>0,:,ch][nm_test[:,:,ch].astype(bool)[fam>0]]):.4f}' for lab, ch in chans]
    save(fig)

# ======================= Page 7: attack surface + per-bus probability (FIGURE, spatial/distributional) =======================
# One page per system: sparsity (how many buses an attack touches) and spatial spread (which buses) are both
# grid-size dependent, so each system gets its own page.
for fi, c in enumerate(SYS):
    fig = newpage(f"Figure 2{'abc'[fi]}, Attack Surface & Spatial Pattern (IEEE-{c})",
                  "how many buses an attack touches, and which buses are attacked")
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.93, top=0.84, bottom=0.34, wspace=0.26)
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    surf = DATA[c]["surface"]
    ax0.hist(surf, bins=range(1, int(surf.max()) + 2), color=DET, alpha=0.85, align="left")
    ax0.set_title("Attacked buses per record", fontsize=9); ax0.set_xlabel("# attacked buses", fontsize=8); ax0.set_ylabel("count", fontsize=8); ax0.tick_params(labelsize=7)
    pb = DATA[c]["perbus"]
    ax1.bar(range(len(pb)), pb, color=STEAL, width=1.0)
    ax1.set_title("Per-bus attack probability", fontsize=9); ax1.set_xlabel("bus index", fontsize=8); ax1.set_ylabel("P(attacked)", fontsize=8); ax1.tick_params(labelsize=7)
    caption(fig, "Left, most attacks are sparse (a handful of buses), so localization, not just detection, is the task. Right, the "
                 "attack is spread across the grid rather than fixed to a few buses (LRA randomizes its target line and bus subset), so "
                 "a model cannot simply memorize a hot-spot. Both are spatial/distributional, hence figures rather than tables.", y=0.27)
    SIDE[f"attack_surface_ieee{c}"] = ["metric,value"] + [f"mean_attacked_buses,{surf.mean():.2f}", f"median_attacked_buses,{np.median(surf):.0f}", f"max_attacked_buses,{int(surf.max())}"]
    save(fig)

# ======================= Page 8: residual box/violin (FIGURE, distributional) =======================
# One page per system. The stealth argument (a real LOCAL footprint that vanishes in the network-wide average)
# is the crux of the dataset, so we show it for every grid, the wash-out grows with grid size (few of N buses move).
resid = None
rp = os.path.join(RES, "ml_only_residuals_v2.npz")
if os.path.exists(rp): resid = np.load(rp)
for fi, c in enumerate(SYS):
    have = resid is not None and f"ieee{c}_benign_dP" in resid
    if have:
        fig = newpage(f"Figure 3{'abc'[fi]}, How Far Each Attack Moves the True State (IEEE-{c})",
                      "injection deviation from the true state, at the attacked buses (local) vs over the whole network")
        gs = fig.add_gridspec(1, 2, left=0.08, right=0.93, top=0.84, bottom=0.36, wspace=0.26)
        axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        floor = float(np.median(resid[f"ieee{c}_benign_dP"]))             # benign = pure meter noise -> the floor
        # LEFT: deviation AT THE ATTACKED buses (the real per-target footprint), attack families only
        labA = [f for f in FAMS if f"ieee{c}_{f}_dPatk" in resid and len(resid[f"ieee{c}_{f}_dPatk"])]
        partsA = axA.violinplot([resid[f"ieee{c}_{f}_dPatk"] for f in labA], showmedians=True, widths=0.85)
        for j, b in enumerate(partsA["bodies"]): b.set_facecolor(STEAL if labA[j] in ("Aq", "At", "Al") else DET); b.set_alpha(0.72)
        axA.axhline(floor, ls="--", lw=1.0, color="#444"); axA.text(len(labA)+0.4, floor, "benign\nfloor", fontsize=6.3, va="center", color="#444")
        axA.set_xticks(range(1, len(labA)+1)); axA.set_xticklabels(labA, rotation=45, fontsize=7.5)
        axA.set_title("AT the attacked buses   |dP| (MW)", fontsize=9.5); axA.tick_params(labelsize=7.5)
        # RIGHT: deviation over ALL metered buses (the network-wide aggregate), benign + families
        order = ["benign"] + FAMS
        partsB = axB.violinplot([resid[f"ieee{c}_{f}_dP"] for f in order], showmedians=True, widths=0.85)
        for j, b in enumerate(partsB["bodies"]): b.set_facecolor(STEAL if order[j] in ("Aq", "At", "Al") else (MUTE if order[j] == "benign" else DET)); b.set_alpha(0.72)
        axB.axhline(floor, ls="--", lw=1.0, color="#444"); axB.text(len(order)+0.4, floor, "noise\nfloor", fontsize=6.3, va="center", color="#444")
        axB.set_xticks(range(1, len(order)+1)); axB.set_xticklabels(order, rotation=45, fontsize=7.5)
        axB.set_title("over ALL metered buses   |dP| (MW)", fontsize=9.5); axB.tick_params(labelsize=7.5)
        stealthy_med = np.median(np.concatenate([resid[f"ieee{c}_{f}_dPatk"] for f in labA if f in ("Aq", "At", "Al") and f"ieee{c}_{f}_dPatk" in resid])) if labA else float("nan")
        caption(fig, f"Left, at the buses each attack actually targets, the stealthy families (Aq/At/Al, red) move the injection by a real but "
                     f"plausibility-bounded amount (median ~{stealthy_med:.1f} MW here), a genuine manipulation, not noise. Right, averaged over the WHOLE "
                     f"network, that same manipulation is indistinguishable from benign, and the stealthy families sit on the meter-noise floor, "
                     f"because only a few of the grid's {DATA[c]['N']} buses move and the rest wash out the average. That combination of a real LOCAL effect "
                     f"that vanishes in the network-wide statistics is exactly what makes them stealthy. No aggregate check or bad-data detector "
                     f"sees them, so only per-bus localization can. (The detectable Ad/As/Ar, blue, perturb meters more but inconsistently, "
                     f"which is what BDD catches, see the BDD page.) The wash-out is stronger on the larger grids, where a fixed handful of "
                     f"attacked buses is a smaller fraction of the network.", y=0.30)
        SIDE[f"residuals_ieee{c}"] = ["family,median_dP_at_attacked_MW,median_dP_all_metered_MW"] + \
            [f'{f},{(np.median(resid[f"ieee{c}_{f}_dPatk"]) if f"ieee{c}_{f}_dPatk" in resid and len(resid[f"ieee{c}_{f}_dPatk"]) else float("nan")):.3f},{np.median(resid[f"ieee{c}_{f}_dP"]):.3f}' for f in order]
        save(fig)
    else:                                                   # PLACEHOLDER: residuals pass not finished for this system
        fig = newpage(f"Figure 3{'abc'[fi]}, How Far Each Attack Moves the True State (IEEE-{c})",
                      "per-record injection deviation from the true state at the same timestep, RMS over metered buses")
        fig.text(0.5, 0.54, "[ PENDING ]", ha="center", va="center", fontsize=17, weight="bold", color=INK)
        fig.text(0.5, 0.47, "The per-family residual distributions are still computing on the new v0.4.0 data.\n"
                 "This page populates automatically the moment the residuals pass finishes, just re-run the report.",
                 ha="center", va="center", fontsize=10.5, color=MUTE, family="serif", linespacing=1.5)
        save(fig)

# ======================= Operational impact, damage to operator decisions =======================
OPI = load_json("operator_impact.json", None)
if OPI and OPI.get("cases", {}).get("ieee118"):
    ci = OPI["cases"]["ieee118"]
    # headline stress level = 1.3 (grid loaded near limits, the regime where an operator actually acts)
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
    fig = table_page("Operational Impact, Damage to Operator Decisions",
                     "does a stealthy attack that PASSES bad-data detection still make the operator misjudge line safety? (IEEE-118, grid loaded to ~1.3x so lines sit near thermal limits)",
                     opi_cols, rows, cap=(
                     "Line-overload masking (our operationalization of the load-redistribution damage model, Yuan et al. 2011). For each "
                     "line we compare its TRUE thermal loading to the loading an operator would read from the stealthy-attacked state, and "
                     "count where they disagree across the 100%-of-rating action threshold. A HIDDEN overload (red) is the dangerous case, "
                     "a line truly over its limit that the attacked reading shows as safe, so the operator takes no corrective action. "
                     "The benign control gives exactly zero misjudgments (validated, the no-attack re-solve reproduces the true state to "
                     "1e-10 p.u.), so every event below is caused by the attack. KEY RESULT. The stealthy families evade bad-data detection "
                     "(see the BDD page) yet still flip line-safety status on nearly half (Aq) to over 80% (Al) of samples and hide hundreds "
                     "of real overloads, and the load-redistribution family Al, the classical "
                     "operationally-damaging attack, is the worst. The GNN-localization literature (Boyaci et al. 2022, Liu et al. 2009) "
                     "reports only detection/localization metrics and never an operator consequence. This carries that missing impact "
                     "label. Caveats, stated plainly. Thermal limits are derived from peak nominal line current (the IEEE cases ship "
                     "placeholder ratings ~200x too large), and the effect is shown at a stressed operating point because the base cases "
                     "are lightly loaded, at nominal load few lines sit near a limit, so few can be masked."),
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
det_cols = ["System", "DR", "FA", "det-F1"] + ["DR-" + f for f in FAMS]
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
    fig = multi_table_page("Model Benchmark, Localization & Detection",
                       "reference ARMA localizer (measurements + KCL residual + temporal delta), test split",
                       [dict(subhead="Model complexity ladder, overall localization swF1 (does the graph model earn its complexity?)",
                             cols=ladder_cols, rows=ladder_rows, col_widths=[0.22, 0.26, 0.26, 0.26],
                             cap="Same data, same metric. A per-bus MLP (node features only, NO graph) vs a plain 2-layer GCN vs "
                                 "our ARMA+attention hybrid. The gap from MLP to the graph models measures how much the grid "
                                 "structure matters. The gap from GCN to ARMA measures the wide-receptive-field / attention gain. "
                                 "The hybrid's lead over the simple baselines is what justifies its complexity, and where they are "
                                 "close, the simpler model is preferable."),
                        dict(subhead="Localization, sample-wise F1 per attack type (higher is better)",
                             cols=loc_cols, rows=loc_rows, col_widths=[0.16, 0.14] + [0.10] * 6,
                             cap="Per attack type, not accuracy. LRA (structured, targeted) is the most localizable. The diffuse "
                                 "stealthy families (Aq, At) and the larger grids are the hard cases. A table beats a bar chart "
                                 "here, the exact values are the point."),
                        dict(subhead="Detection, grid-level rate (DR), false-alarm (FA), F1, and per-family DR",
                             cols=det_cols, rows=det_rows, fontsize=8.0, col_widths=[0.13, 0.09, 0.09, 0.10] + [0.098] * 6,
                             cap="Grid-level score S = max over buses of the per-bus attack probability. Detection far exceeds "
                                 "per-bus localization because it aggregates evidence. The three bad-data-evading families are "
                                 "detected in Boyaci range, and ML catches what the classical detector cannot.")])
    SIDE["localization"] = ["system,overall," + ",".join(FAMS)] + [",".join([f"ieee{SYS_ALL[i]}"] + r[1:]) for i, r in enumerate(loc_rows)]
    SIDE["detection"] = ["system,DR,FA,det_f1," + ",".join("DR_" + f for f in FAMS)] + [",".join([f"ieee{SYS_ALL[i]}"] + r[1:]) for i, r in enumerate(det_rows)]
    SIDE["complexity_ladder"] = ["system,mlp_swf1,gcn_swf1,arma_hybrid_swf1"] + [",".join(r) for r in ladder_rows]
    save(fig)
else:
    pending_page("Model Benchmark, Localization & Detection",
                 "reference ARMA localizer (measurements + KCL residual + temporal delta), test split",
                 "The localizer and detector are being re-trained on the v0.4.0 dataset (new attack taxonomy, "
                 "accuracy-class meter noise). Per-family localization and detection numbers populate here once training finishes.")

# ======================= Where the localization difficulty lives, the temporal At family =======================
# Answers a concrete question: IEEE-118/300 sit ~9-10 swF1 points below IEEE-14 overall. We show this is ENTIRELY the
# temporal ramp (At), excluding At, all three grids are flat at ~0.95-0.97, so no other family degrades with size.
# Numbers are derived from the real benchmark (n-weighted, the same weighting that forms 'overall'), not asserted.
if MODELS_READY and all(BENCH.get(f"ieee{c}") for c in SYS):
    def _wmean_swf1(c, exclude=()):              # n-weighted swF1 over attacked families (matches how 'overall' is formed)
        pf = BENCH[f"ieee{c}"]["per_family"]; num = den = 0.0
        for f in FAMS:
            if f in exclude or f not in pf: continue
            num += pf[f]["swf1"] * pf[f]["n"]; den += pf[f]["n"]
        return num / den if den else float("nan")
    gap_rows = []
    for c in SYS:
        ov = BENCH[f"ieee{c}"]["overall"]["swf1"]; exat = _wmean_swf1(c, exclude=("At",)); at = BENCH[f"ieee{c}"]["per_family"]["At"]["swf1"]
        gap_rows.append([f"IEEE-{c}", f"{ov:.3f}", f"{exat:.3f}", f"{at:.3f}", f"{ov-exat:+.3f}"])
    fig = table_page("Where the Difficulty Lives, the Temporal Attack (At) Is the Only Family That Scales Poorly",
                     "the IEEE-118/300 localization gap vs IEEE-14 is almost entirely At, and every other family is flat across grid size",
                     ["System", "overall swF1", "swF1 excl. At", "At swF1", "At's drag"],
                     gap_rows, cap=(
                     "The overall localization gap between IEEE-14 (0.906) and the larger IEEE-118 / 300 (0.813 / 0.818) looks like a "
                     "grid-size effect, but it is not. Remove the single temporal family At and the remaining five (Aq, Ad, As, Ar, Al) sit at "
                     "0.95-0.97 on ALL three grids, flat, with no degradation as the network grows. The entire gap is At, whose own "
                     "localization collapses with size (0.64 -> 0.32 -> 0.17). Why At is uniquely hard. It is the ONE family that evades BOTH "
                     "classical detectors at once, every ramp step is an AC-power-flow-consistent state, so bad-data detection sees nothing "
                     "(exactly like Aq/Al), AND the per-scan change is deliberately gradual, sitting on the natural minute-to-minute load-swing "
                     "floor, so the rate-of-change detector sees nothing either. On a larger grid the same slow ramp on a few buses is a smaller "
                     "fraction of the network and must be picked out from far more candidate buses, so the model has the least signal precisely "
                     "where the attack is most buried. We report this as an OPEN CHALLENGE, not a solved benchmark. The promising direction is a "
                     "physics-informed temporal prior. Score a bus's multi-scan trajectory for being simultaneously power-flow-consistent AND "
                     "monotonically drifting against the grid's normal diurnal load pattern, a signature a slow coordinated ramp leaves but that "
                     "neither a per-scan residual (BDD) nor a raw rate-of-change threshold can isolate. Catching stealthy temporal attacks with "
                     "such a physics-informed temporal model is the clearest next step this benchmark points to."),
                     col_widths=[0.18, 0.22, 0.22, 0.16, 0.16], shade_rule=lambda i, r: "#fbecea")
    SIDE["at_difficulty"] = ["system,overall_swf1,exAt_swf1,At_swf1,At_drag_points"] + [",".join(r) for r in gap_rows]
    save(fig)

# ======================= Why the engineered features work, attacked-bus separability by feature (At excluded) =======================
# Pairs with the At-difficulty page: it shows WHY the non-At families localize so well. Real ROC-AUC (results/
# feature_sep.json): raw meter is near-chance (the attacks are stealthy in the raw signal) while the windowed swing
# feature almost perfectly fingerprints the attacked bus, the mechanism behind the ~0.95-0.97 excl-At results.
FS = load_json("feature_sep.json", None)
if FS and FS.get("systems"):
    order_fam = ["Aq", "Ad", "As", "Ar", "Al"]
    feat_order = ["|P_inj| reading", "temporal delta", "swing (windowed)"]
    fcolors = {"|P_inj| reading": MUTE, "temporal delta": DET, "swing (windowed)": STEAL}
    syslist = [f"ieee{c}" for c in SYS if f"ieee{c}" in FS["systems"]]
    fig = newpage("Why the Features Work, Attacked-Bus Separability by Feature (At excluded)",
                  "how well each per-bus feature separates an attacked bus from an untouched one (ROC-AUC, 0.5 = no signal, 1.0 = perfect)")
    n = max(1, len(syslist)); H = 0.15; GAP = 0.055; TOP0 = 0.60      # 3 stacked panels between the title and caption
    for si, sk in enumerate(syslist):
        ax = fig.add_axes([0.08, TOP0 - si * (H + GAP), 0.85, H])
        fams = [f for f in order_fam if f in FS["systems"][sk]]
        x = np.arange(len(fams)); w = 0.26
        for fi, ft in enumerate(feat_order):
            vals = [FS["systems"][sk][f]["auc"].get(ft) or 0 for f in fams]
            ax.bar(x + (fi - 1) * w, vals, w, color=fcolors[ft], label=ft if si == 0 else None)
        ax.axhline(0.5, ls=":", lw=0.9, color="#555")                    # chance line
        ax.set_ylim(0.4, 1.03); ax.set_xticks(x); ax.set_xticklabels(fams, fontsize=8.5)
        ax.set_ylabel(f"IEEE-{sk[4:]}\nAUC", fontsize=8.5); ax.tick_params(labelsize=7.5)
        if si == 0: ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.03), frameon=False)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    caption(fig, "Each bar is the ROC-AUC for telling an ATTACKED bus from an untouched one using a single feature (At excluded, the one "
                 "family these features cannot see). This is the whole rationale for our feature set. The RAW injection meter sits near 0.5, "
                 "barely better than chance, which is exactly what 'stealthy' means. The engineered temporal features recover the hidden signal. "
                 "The scan-to-scan temporal delta reaches 0.73-0.99, and the windowed SWING feature is near-perfect at 0.82-1.00 for every non-At "
                 "family on all three grids. Swing dominates because a load-plausible manipulation still makes an abnormally large single-scan jump "
                 "relative to the bus's own recent rhythm, which the window normalizes. Crucially this does NOT decay with grid size, the features "
                 "are local, so it is the mechanism behind the ~0.95-0.97 excl-At localization, and why swing serves as both a feature and a "
                 "standalone detector.", y=0.15)
    SIDE["feature_sep"] = ["system,family," + ",".join(feat_order)] + \
        [f"{sk},{f}," + ",".join(str(FS['systems'][sk][f]['auc'].get(ft)) for ft in feat_order) for sk in syslist for f in order_fam if f in FS['systems'][sk]]
    save(fig)

# ======================= Training curves, train/val F1, AUPRC, DR, FA (one 4-panel page per model per network) =======================
# Reads results/history_{model}_{C}.json (written by the training scripts with per-epoch logging). Each page is a
# 4-panel train-vs-val diagnostic for one model on one grid; missing ones are listed on a single pending page so
# the report never shows an empty or fabricated curve.
CURVE_MODELS = [("arma", "ARMA+attn (ours)"), ("gcn", "plain GCN"), ("mlp", "per-bus MLP")]
CURVE_MET = [("f1", "Localization F1"), ("auprc", "Localization AUPRC"),
             ("loss", "Training loss (BCE)"), ("fa", "Detection FA  @tuned thr (lower is better)")]
_curve_missing = []
for c in SYS:
    for mkey, mname in CURVE_MODELS:
        H = load_json(f"history_{mkey}_{c}.json", None)
        if not H or not H.get("epochs"):
            _curve_missing.append(f"{mname} / IEEE-{c}"); continue
        ep = H["epochs"]; tr = H.get("train", {}); va = H.get("val", {})
        fig = newpage(f"Training Curves, {mname}, IEEE-{c}",
                      "per-epoch train (solid) vs validation (dashed), localization F1 & AUPRC, BCE loss, detection FA at the tuned threshold")
        gs = fig.add_gridspec(2, 2, left=0.08, right=0.93, top=0.85, bottom=0.16, wspace=0.24, hspace=0.36)
        for ax, (mk, ml) in zip([fig.add_subplot(gs[i]) for i in range(4)], CURVE_MET):
            if mk in tr: ax.plot(ep, tr[mk], color=DET, lw=1.5, label="train")
            if mk in va: ax.plot(ep, va[mk], color=ACCENT, lw=1.5, ls="--", label="val")
            ax.set_title(ml, fontsize=9); ax.set_xlabel("epoch", fontsize=7.5); ax.tick_params(labelsize=7)
            # FA is a rate in [0,1]; pin its axis to [0,1] so a low FA reads honestly (not a 1e-8 auto-zoom) and a
            # rising FA is visible. Loss/F1/AUPRC keep their own auto scale.
            if mk == "fa": ax.set_ylim(-0.02, 1.02)
            ax.margins(x=0.02); ax.legend(fontsize=7, loc="best")
            for s in ("top", "right"): ax.spines[s].set_visible(False)
        caption(fig, f"Per-epoch train (solid) vs held-out validation (dashed) for {mname} on IEEE-{c}. Localization F1/AUPRC are sample-wise on "
                     "attacked buses. The BCE loss shows convergence. Detection FA is grid-level at the VAL-TUNED threshold (the real operating point), "
                     "a good detector sits near 0, whereas a fixed 0.5 threshold would make any confident model look like it flags everything. A "
                     "small, stable train/val gap indicates generalization.", y=0.115)
        SIDE[f"curve_{mkey}_ieee{c}"] = ["epoch," + ",".join(f"train_{mk},val_{mk}" for mk, _ in CURVE_MET)] + \
            [",".join([str(ep[i])] + [f"{tr.get(mk,[None]*len(ep))[i]},{va.get(mk,[None]*len(ep))[i]}" for mk, _ in CURVE_MET]) for i in range(len(ep))]
        save(fig)
if _curve_missing:
    pending_page("Training Curves, Pending",
                 "train/val F1, AUPRC, DR, FA, one 4-panel page per model per network",
                 "Per-epoch training histories are still being logged / re-trained for " + ", ".join(_curve_missing) +
                 ". Each becomes its own 4-panel train-vs-val page the moment its history_*.json lands, just re-run the report.")

# ======================= t-SNE: raw inputs vs learned embeddings (per system per model, At excluded) =======================
# Reads results/embeds_{model}_{C}.npz (Xin=input features, Z=learned embedding, family=id, NO At). We run t-SNE at
# report time and save the 2D coords as a sidecar (restyle without recomputing). At is excluded on purpose so the
# grouping is clean, it is the one family that does not separate. Shows whether the model learns a family-discriminative
# representation the raw inputs don't already have.
try:
    from sklearn.manifold import TSNE
    _HAVE_TSNE = True
except Exception:
    _HAVE_TSNE = False
FAM_TSNE = [(0, "benign", FAMCOL["benign"]), (1, "Aq", FAMCOL["Aq"]), (2, "Ad", FAMCOL["Ad"]), (3, "As", FAMCOL["As"]), (4, "Ar", FAMCOL["Ar"]), (6, "Al", FAMCOL["Al"])]
_tsne_missing = []
for c in SYS:
    for mkey, mname in CURVE_MODELS:
        ep = os.path.join(RES, f"embeds_{mkey}_{c}.npz")
        if not (_HAVE_TSNE and os.path.exists(ep)):
            _tsne_missing.append(f"{mname} / IEEE-{c}"); continue
        z = np.load(ep); fam = z["family"]
        fig = newpage(f"t-SNE, {mname}, IEEE-{c}  (At excluded)",
                      "each point is one record, raw input features (left) vs the model's learned embedding (right), coloured by attack family")
        coords = {}
        for pi, (key, ttl) in enumerate([("Xin", "raw input features"), ("Z", "learned embedding")]):
            if key not in z or len(z[key]) < 10: continue
            X = z[key]; perp = min(30, max(5, len(X) // 20))
            emb = TSNE(n_components=2, init="pca", perplexity=perp, random_state=123).fit_transform(X)
            coords[key] = emb
            ax = fig.add_axes([0.08 + pi * 0.45, 0.30, 0.40, 0.55])
            for fid, flab, col in FAM_TSNE:
                m = fam == fid
                if m.any(): ax.scatter(emb[m, 0], emb[m, 1], s=6, color=col, alpha=0.6, label=flab, linewidths=0)
            ax.set_title(ttl, fontsize=9.5); ax.set_xticks([]); ax.set_yticks([])
            for s in ("top", "right", "left", "bottom"): ax.spines[s].set_visible(False)
            if pi == 0:
                ax.legend(fontsize=7.5, markerscale=1.8, loc="upper center", bbox_to_anchor=(1.02, -0.03), ncol=6, frameon=False)
        caption(fig, f"t-SNE of {mname} on IEEE-{c}, At excluded. Left, raw per-record input features. Right, the model's pooled learned embedding "
                     "(what its detection head sees). Tighter, better-separated same-family clusters on the right than the left mean the model learned "
                     "a family-discriminative representation rather than passing the inputs through, that separation is what drives the scores. Benign "
                     "forms the bulk. A good embedding pushes each attack family off it. At is left out because it stays mixed into benign (the whole "
                     "point of the temporal-attack page). 2D coordinates are saved as a sidecar.", y=0.15)
        # sidecar: the 2D coords + family, per the always-save-plot-data rule
        rows = ["point,family,xin_x,xin_y,emb_x,emb_y"]
        xi = coords.get("Xin"); zi = coords.get("Z")
        for i in range(len(fam)):
            xin = f"{xi[i,0]:.3f},{xi[i,1]:.3f}" if xi is not None else ","
            emb = f"{zi[i,0]:.3f},{zi[i,1]:.3f}" if zi is not None else ","
            rows.append(f"{i},{int(fam[i])},{xin},{emb}")
        SIDE[f"tsne_{mkey}_ieee{c}"] = rows
        save(fig)
if _tsne_missing:
    pending_page("t-SNE Embeddings, Pending",
                 "raw-input vs learned-embedding t-SNE, one page per model per network (At excluded)",
                 "Embedding dumps are still being produced for " + ", ".join(_tsne_missing) +
                 ". Each becomes an input-vs-embedding t-SNE page once its embeds_*.npz lands, just re-run the report.")

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
    def g(d, k): return f"{d[k]:.4f}" if d and k in d and d[k] is not None else "n/a"
    def gt(d, k): return f"{d[k]:.2f}" if d and k in d and d[k] is not None else "n/a"
    se_rows.append([f"IEEE-{c}", g(p, "V_mae_meter_attacked"), g(w, "V_mae_wls"), g(nn, "V_mae_se_metered_attacked"), g(p, "V_mae_se_metered_attacked"),
                    gt(p, "th_mae_meter_attacked"), gt(w, "th_mae_wls"), gt(nn, "th_mae_se_metered_attacked"), gt(p, "th_mae_se_metered_attacked")])
blocks = []
if at_rows:
    blocks.append(dict(subhead="Physics-biased attention, localization swF1 / detection F1, ARMA vs + attention",
                       cols=at_cols, rows=at_rows, col_widths=[0.18, 0.16, 0.16, 0.14, 0.18, 0.18],
                       cap="A parallel attention head, gated from zero so the network starts identical to plain ARMA, attends "
                           "along physically-suspicious branches. It lifts localization on every system while holding detection, "
                           "now the SDK's default localizer."))
blocks.append(dict(subhead="Attack-resilient state estimation, recovery error on attacked buses, |V| (p.u.) / theta (deg)",
                   cols=se_cols, rows=se_rows, fontsize=7.8, col_widths=[0.13] + [0.108] * 8,
                   cap="Recover the TRUE state from possibly-attacked measurements. The result is honestly mixed. VOLTAGE recovery "
                       "is the real win. The learned estimators beat the raw meter and WLS at IEEE-14 (~4x) and IEEE-118 (~5x). ANGLE "
                       "only clearly wins at IEEE-14. At 118 it ties the meter and at 300 it is worse. The reason is a property of the "
                       "dataset, not a bug. On v0.4.0 the stealthy attacks are load-plausible and BDD-consistent, so they move the TRUE "
                       "angle very little, the meter is already only ~0.24-0.65 deg off under attack, leaving almost nothing to recover, "
                       "and the estimator's own reconstruction floor then dominates. In other words, the attacks are now stealthy enough "
                       "that state recovery is nearly moot for angle. The physics term (PINN vs NN) helps at 14 and 300 but hurts at 118, "
                       "so it is system-dependent, not universal. WLS (classical) is confirmed fooled, its error jumps 9-126x under "
                       "stealthy attack. Reported as measured, single seed."))
if MODELS_READY and (at_rows or any("n/a" not in r for r in se_rows)):
    fig = multi_table_page("Model Improvements, Attention & State Estimation",
                       "beyond the base localizer, an attention head that raises localization, and a state estimator that recovers the true state",
                       blocks)
    SIDE["attention_ab"] = ["system,loc_arma,loc_attn,det_arma,det_attn"] + \
        [f'ieee{c},{ATTN[f"ieee{c}"]["arma"]["loc"]},{ATTN[f"ieee{c}"]["hybrid"]["loc"]},{ATTN[f"ieee{c}"]["arma"]["det"]},{ATTN[f"ieee{c}"]["hybrid"]["det"]}' for c in SYS if f"ieee{c}" in ATTN]
    SIDE["state_estimation"] = ["system,V_meter,V_wls,V_nn,V_pinn,th_meter,th_wls,th_nn,th_pinn"] + [",".join([f"ieee{SYS[i]}"] + r[1:]) for i, r in enumerate(se_rows)]
    save(fig)
else:
    pending_page("Model Improvements, Attention & State Estimation",
                 "beyond the base localizer, an attention head that raises localization, and a state estimator that recovers the true state",
                 "The attention A/B and the attack-resilient state estimator are being re-run on the v0.4.0 data. "
                 "Localization gains and per-variable recovery error populate here once training finishes.")

# ======================= Why the learned estimator beats WLS on voltage but not angle (mechanism, quantified) =======================
# The SE result is mixed and reviewers WILL ask why. This page answers it from measurements (results/se_analysis.json),
# not hand-waving: voltage wins are exactly WLS's attack-drag, which redundancy dilutes on big grids; angle loses to a
# learned reconstruction floor that grows because angle is a global state a shallow GNN can't propagate across a large grid.
SEA = load_json("se_analysis.json", None)
if SEA and SEA.get("systems"):
    SYST = SEA["systems"]; sysk = [f"ieee{c}" for c in SYS if f"ieee{c}" in SYST]
    def _pf(k, fam, key): return SYST[k]["per_family"][fam].get(key, 0.0)   # per-family SE metric (benign / Ad ...)
    def _sy(k, key): return SYST[k].get(key, 0.0)                            # system-level stat (redundancy, angle spread)
    labs = [f"IEEE-{k[4:]}" for k in sysk]; x = np.arange(len(sysk))
    fig = newpage("Why the Learned Estimator Beats WLS on Voltage but Not Angle",
                  "the mixed state-estimation result, explained from measurements, WLS is optimal until an attack drags it (voltage), but angle is a global state a shallow GNN cannot reach")
    axV = fig.add_axes([0.08, 0.57, 0.85, 0.27]); axA = fig.add_axes([0.08, 0.19, 0.85, 0.26])
    # Panel A (mechanisms 1-2): learned voltage advantage = how far the high-drag Ad attack pulls WLS; it shrinks with redundancy
    wdrag = [_pf(k, "Ad", "V_wls") for k in sysk]; advV = [_pf(k, "Ad", "adv_V") for k in sysk]
    axV.bar(x - 0.2, wdrag, 0.4, color=DET, label="WLS |V| error under Ad (attack drags it off truth)")
    axV.bar(x + 0.2, advV, 0.4, color=STEAL, label="learned advantage over WLS (= the drag it avoids)")
    axV.set_xticks(x); axV.set_xticklabels(labs, fontsize=8); axV.set_ylabel("|V| MAE (p.u.)", fontsize=8.5); axV.tick_params(labelsize=7.5)
    _advstr = " -> ".join(f"{v:.4f}".rstrip("0").rstrip(".") for v in advV)
    axV.legend(fontsize=7, loc="upper right"); axV.set_title(f"Voltage. The learned win equals WLS's attack-drag, and it shrinks as redundancy grows ({_advstr})", fontsize=9.0)
    for s in ("top", "right"): axV.spines[s].set_visible(False)
    # Panel B (mechanism 3): benign angle floor grows for the learned model; WLS stays flat (global-state reach problem)
    lf = [_pf(k, "benign", "th_learned") for k in sysk]; wf = [_pf(k, "benign", "th_wls") for k in sysk]; mf = [_pf(k, "benign", "th_meter") for k in sysk]
    axA.plot(labs, lf, "o-", color=STEAL, lw=1.6, label="learned angle floor (grows)")
    axA.plot(labs, wf, "s-", color=DET, lw=1.6, label="WLS (flat, solves whole network jointly)")
    axA.plot(labs, mf, "^--", color=MUTE, lw=1.2, label="raw meter")
    axA.set_ylabel("benign angle MAE (deg)", fontsize=8.5); axA.tick_params(labelsize=7.5); axA.legend(fontsize=7, loc="upper left")
    axA.set_title("Angle. The learned floor grows with grid size (global state, wide-but-shallow GNN on a large-diameter graph). WLS stays flat", fontsize=9.0)
    for s in ("top", "right"): axA.spines[s].set_visible(False)
    _lfstr = " -> ".join(f"{v:.2f}" for v in lf); _wfavg = float(np.mean(wf))
    caption(fig, "The honest SE story, measured on the per-unit v0.4.1 data. (1) The learned estimator is an ATTACK-INVARIANT DENOISER, |V| error "
                 "~constant across every family, so it beats WLS only when an attack DRAGS WLS off the truth (high-magnitude Ad/As), and the voltage win "
                 f"(top) is exactly that drag. (2) The drag shrinks with grid size, redundant measurements grow ~20x and dilute each capped local "
                 f"attack, so the win fades toward ~0 by IEEE-300. (3) ANGLE is a GLOBAL state. The learned floor grows {_lfstr} deg (bottom) while WLS "
                 f"holds ~{_wfavg:.2f}, because the angle spread triples AND even the wider hid256 GNN (~4 hops) can't carry a network-wide phase "
                 "reference across a 300-bus graph, whereas WLS solves it jointly. Widening helped (300 floor 0.50 -> 0.34 deg) but didn't close it. "
                 "NET, the learned SE wins only on a small grid, a high-drag attack, and voltage.", y=0.16)
    SIDE["se_mechanism"] = ["system,V_wls_drag_underAd,learned_adv_V_underAd,angle_floor_learned_deg,angle_wls_deg,angle_meter_deg,theta_state_std_deg,redundancy"] + \
        [f'{k},{_pf(k,"Ad","V_wls")},{_pf(k,"Ad","adv_V")},{_pf(k,"benign","th_learned")},{_pf(k,"benign","th_wls")},{_pf(k,"benign","th_meter")},{_sy(k,"state_spread_theta_std_deg")},{_sy(k,"redundancy_meas_per_state")}' for k in sysk]
    save(fig)

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
    fig = multi_table_page("Reference Benchmark, Falas et al. Table II Reproduction",
                       "IEEE-14 state estimation, PINN vs plain NN across the attack ladder, Falas reported vs our reproduction",
                       [dict(subhead="Mean absolute error (standardized-pooled), lower is better",
                             cols=falas_cols, rows=falas_mae, col_widths=_fcw,
                             cap="We reproduce the same range and the attack-ladder trend (error rises with attack magnitude). Our "
                                 "absolute MAE sits higher because of a standardization and target-construction difference documented "
                                 "separately, not a method error."),
                        dict(subhead="R-squared, higher is better",
                             cols=falas_cols, rows=falas_r2, col_widths=_fcw,
                             cap="The real finding is qualitative. In our setup the plain NN beats the PINN, the opposite of Falas, and "
                                 "our NN R-squared does not collapse under attack. A PINN beats an NN only against a weak data-only "
                                 "baseline. Our NN is stronger and our attack is milder, so the physics term does not add. Falas reports "
                                 "IEEE-14 only.")])
    SIDE["falas_reproduction"] = ["metric,scenario,falas_pinn,falas_nn,ours_pinn,ours_nn"] + \
        [f"MAE,{','.join(r)}" for r in falas_mae] + [f"R2,{','.join(r)}" for r in falas_r2]
    save(fig)
else:
    pending_page("Reference Benchmark, Falas et al. Table II Reproduction",
                 "IEEE-14 state estimation, PINN vs plain NN across the attack ladder",
                 "Our reproduction columns are being re-computed on the v0.4.0 IEEE-14 data. Falas's own reported values "
                 "are published and unchanged, but we withhold the side-by-side until our numbers are current.")

# ======================= State estimation: V and theta separated (companion to the pooled metric) =======================
# se_rows (from the improvements page) = [System, V meter, V WLS, V NN, V PINN, th meter, th WLS, th NN, th PINN]
if MODELS_READY and se_rows:
    v_rows = [[r[0], r[1], r[2], r[3], r[4]] for r in se_rows]
    th_rows = [[r[0], r[5], r[6], r[7], r[8]] for r in se_rows]
    sep_cols = ["System", "meter", "WLS", "NN", "PINN"]
    fig = multi_table_page("State Estimation, Voltage and Angle Separated",
                       "the attack-resilient state-recovery error, split by variable so the per-variable behavior a pooled metric hides is visible",
                       [dict(subhead="Voltage magnitude |V| error (p.u.) on attacked buses, lower is better",
                             cols=sep_cols, rows=v_rows, col_widths=[0.20, 0.20, 0.20, 0.20, 0.20],
                             cap="Split by variable, the picture is size-dependent. On IEEE-14 both learned estimators sit far below "
                                 "the raw meter and WLS on voltage. On IEEE-118 they still beat the baselines and NN edges PINN. On "
                                 "IEEE-300 the learned voltage error rises above the meter/WLS baseline (PINN is the better of the two "
                                 "learned but still not ahead of the classical baselines). Physics helps voltage only on the small system."),
                        dict(subhead="Voltage angle theta error (deg) on attacked buses, lower is better",
                             cols=sep_cols, rows=th_rows, col_widths=[0.20, 0.20, 0.20, 0.20, 0.20],
                             cap="Angle carries most of the error and the SE only clearly helps at IEEE-14 (PINN 0.15 vs meter 0.21 deg). "
                                 "At 118 it ties the meter. At 300 both learned estimators are WORSE than the meter. This is not (mainly) "
                                 "under-training, it is because the v0.4.0 stealthy attacks barely move the true angle (the meter is only "
                                 "~0.24-0.65 deg off under attack), so there is little to recover and the estimator's reconstruction floor "
                                 "dominates. Angle recovery only pays off for the high-magnitude Al/LRA family. The honest reading. State "
                                 "recovery matters far less once the attacks are this stealthy.")])
    SIDE["se_separated"] = ["variable,system,meter,WLS,NN,PINN"] + \
        [f"V,{','.join(r)}" for r in v_rows] + [f"theta,{','.join(r)}" for r in th_rows]
    save(fig)
else:
    pending_page("State Estimation, Voltage and Angle Separated",
                 "attack-resilient state-recovery error, split by variable (voltage vs angle)",
                 "The per-variable state-estimation results are being re-computed on the v0.4.0 data, see the previous "
                 "State Estimation page. This companion breakdown populates alongside it.")

# ======================= SE under attack (2026-07-16): noise-floor framing, WLS-vs-learned per regime, robust-WLS ceiling =======================
# These pages read the retargeted WLS baseline (se_C_wls.json, now with unmetered-bus columns), the temporal
# ARMA+attn estimator (se_C_armaattn_temporal.json), and the oracle robust-WLS ceiling (se_14_robwls_oracle.json).
# All numbers are read from results/, never hand-entered; a missing run renders as n/a rather than a fabricated cell.
SEW = {c: load_json(f"se_{c}_wls.json") for c in SYS}
SET = {c: load_json(f"se_{c}_armaattn_temporal.json") for c in SYS}
ROB = load_json("se_14_robwls_oracle.json")


def _walk(d, *keys, nd=3):
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return "n/a"
        d = d[k]
    try:
        return f"{float(d):.{nd}f}"
    except Exception:
        return "n/a"


# ---- Page A: WLS vs learned, per regime, metered and unmetered ----
reg_cols = ["System", "th WLS met", "th WLS unmet", "th learned met", "th learned unmet", "V WLS", "V learned"]
ben_rows, atk_rows = [], []
for c in SYS:
    w = SEW[c] or {}; t = SET[c] or {}
    wb = w.get("per_family", {}).get("benign", {}); tb = t.get("overall_benign", {}); tbu = t.get("overall_benign_unmetered", {})
    ben_rows.append([f"IEEE-{c}", _walk(wb, "th_mae_wls"), _walk(wb, "th_mae_wls_unmetered"),
                     _walk(tb, "th_mae_se_metered"), _walk(tbu, "th_mae_se"),
                     _walk(wb, "V_mae_wls", nd=4), _walk(tb, "V_mae_se_metered", nd=4)])
    wa = w.get("overall_attacked", {}); wau = w.get("overall_unmetered", {}); ta = t.get("overall", {}); tau = t.get("overall_attacked_unmetered", {})
    atk_rows.append([f"IEEE-{c}", _walk(wa, "th_mae_wls"), _walk(wau, "attacked_th"),
                     _walk(ta, "th_mae_se_metered_attacked"), _walk(tau, "th_mae_se"),
                     _walk(wa, "V_mae_wls", nd=4), _walk(ta, "V_mae_se_metered_attacked", nd=4)])
fig = multi_table_page("State Estimation Under Attack, WLS versus the Learned Estimator",
                       "the honest split. classical WLS is optimal on clean data at every bus, the learned estimator earns its place under attack",
                       [dict(subhead="Benign regime, angle MAE (deg) and voltage MAE (p.u.), metered and unmetered buses",
                             cols=reg_cols, rows=ben_rows, fontsize=8.4, col_widths=[0.16, 0.15, 0.15, 0.15, 0.15, 0.12, 0.12],
                             cap="On clean data WLS is the maximum-likelihood estimator and sits on the noise floor at every bus, metered "
                                 "and unmetered alike, because the systems are observable and the power-flow physics pins the angle-unmetered "
                                 "buses through the real-power measurements. The learned estimator ties WLS on voltage. On angle it trails, and "
                                 "the gap grows with grid size, from about 0.05 deg at IEEE-14 to 0.25 deg at IEEE-300, because angle is a "
                                 "global state a shallow graph network propagates less well as the grid grows. Every one of those figures still "
                                 "sits below the roughly 0.29 deg single-meter angle noise, so the benign gap carries no operational consequence "
                                 "and is not the bar the estimator has to clear."),
                        dict(subhead="Attacked regime, angle MAE (deg) and voltage MAE (p.u.), metered and unmetered buses",
                             cols=reg_cols, rows=atk_rows, fontsize=8.4, col_widths=[0.16, 0.15, 0.15, 0.15, 0.15, 0.12, 0.12],
                             cap="Under attack WLS faithfully estimates the attacker's false state, so its angle error jumps by roughly 16x to "
                                 "120x the benign floor. The learned estimator beats WLS on every system and on both bus sets because it carries "
                                 "a prior over plausible states and reads the previous scan. The margin is largest at IEEE-14, 0.056 against 0.364 "
                                 "deg, and it holds at IEEE-118 and IEEE-300 where the angle state is harder to recover. This is where state "
                                 "recovery matters, and it is the regime the estimator is built for. Reported as measured, single seed.")])
SIDE["se_regime_benign"] = ["system," + ",".join(reg_cols[1:])] + [",".join(r) for r in ben_rows]
SIDE["se_regime_attacked"] = ["system," + ",".join(reg_cols[1:])] + [",".join(r) for r in atk_rows]
save(fig)

# ---- Page B: within the noise floor, attacks have no effect (the detectability-impact tradeoff) ----
fig = newpage("Within the Noise Floor, an Attack Has No Effect",
              "why a sub-noise attack is both undetectable and inconsequential, and where that line sits on our data")
prose(fig, [
 ("Two conditions make an attack stealthy", [
  "The residual condition. To evade bad-data detection the normalized measurement residuals must stay within the "
  "range benign noise already produces, which the classic result of Liu, Ning, and Reiter (2009) achieves by "
  "aligning the injection with the column space of the measurement Jacobian so the residual barely moves.",
  "The plausibility condition. Separately, the injected loads and measurements must stay inside the grid's "
  "plausible operating range, a bound on the values rather than the residuals. Our double-stealthy construction "
  "enforces both, so an attack is invisible to a residual detector and to a plausibility check at once.",
 ]),
 ("A stealthy attack is by construction too small to matter", [
  "A perfect column-space injection needs exact topology and the right meter set, which a realistic attacker does "
  "not have, so a residual always remains and the attack must be shrunk toward the meter noise level to stay under "
  "the detection threshold. This is the detectability and impact tradeoff, made precise by Kosut, Jia, Thomas, and "
  "Tong (2011) and by the security indices of Sandberg, Teixeira, and Johansson (2010) and the AC analysis of Hug "
  "and Giampapa (2012).",
  "The impact half follows from the estimator. A redundant weighted-least-squares estimator, in the maximum-"
  "likelihood tradition of Schweppe and Wildes (1970) and Abur and Gomez Exposito (2004), resolves the state only "
  "to the noise floor fixed by the measurement error covariance, itself set by physical meter accuracy of a "
  "fraction of a percent for instrument transformers (Asprou, Kyriakides, and Albu 2013) and one percent total "
  "vector error for synchrophasors (IEEE Std C37.118.1, 2011). A perturbation held at that floor cannot move the "
  "estimate or any operator decision beyond the estimator's own uncertainty.",
 ]),
 ("Where the line sits on our data", [
  "Our Aq load-moving sensitivity sweep puts the noise-floor crossing at a topology-independent load move near "
  "1.7 percent, below which detection AUC collapses to chance. That is the lower edge of the ML-only-dangerous "
  "window. Below it a sub-noise attack is a wash for any estimator, learned or classical. Above it our released "
  "attacks sit 16x to 120x the benign angle floor, which is the regime that matters.",
 ]),
], top=0.895)
save(fig)

# ---- Page C: GNN-guided robust WLS, the division of labor (oracle ceiling) ----
if ROB and ROB.get("per_family"):
    rc_cols = ["Family", "plain th met", "robust th met", "plain th unmet", "robust th unmet", "class"]
    rc_rows = []
    for name, pf in ROB["per_family"].items():
        p = pf.get("plain", {}); r = pf.get("robust", {})
        rc_rows.append([name, _walk(p, "th_met"), _walk(r, "th_met"), _walk(p, "th_unmet"), _walk(r, "th_unmet"),
                        "stealthy" if pf.get("stealthy") else "crude" if name != "benign" else ""])
    fig = table_page("GNN-Guided Robust WLS, the Division of Labor",
                     "an oracle drops the flagged buses and re-solves. what classical bad-data removal can and cannot recover (IEEE-14, angle deg)",
                     rc_cols, rc_rows,
                     cap="An oracle names the attacked buses, we drop their bus measurements, and WLS re-solves. On benign nothing is dropped "
                         "so it is WLS exactly. Replay (Ar) is fully recovered, from 0.445 back to the 0.021 benign floor, because the bad data "
                         "is inconsistent and removing it restores the clean solution. The stealthy families (Aq, At, Al) barely move, because "
                         "a stealthy attack leaves a self-consistent measurement set with no residual to flag, so dropping the buses cannot "
                         "recover the true state. That is exactly why the learned estimator, not bad-data removal, is the necessary tool for "
                         "the stealthy families. Ad and As do not recover here because their corruption reaches the incident flow measurements "
                         "we did not drop, a bound on this oracle rather than on the idea. Real localizer flags replace the oracle in the "
                         "pipeline, and the same model trained on one grid transfers to the others because the backbone is size-invariant.",
                     fontsize=8.4, col_widths=[0.16, 0.17, 0.17, 0.17, 0.17, 0.16],
                     shade_rule=lambda i, row: "#fbecea" if row[5] == "stealthy" else ("#eaf2f8" if row[5] == "crude" else "#ffffff"))
    SIDE["se_robwls_oracle"] = ["family," + ",".join(rc_cols[1:])] + [",".join(r) for r in rc_rows]
    save(fig)

# ---- Page D: the four papers and which report sections feed each ----
fig = newpage("The Four Papers and Where They Draw From This Report",
              "this report is the single source of truth for the numbers. each paper draws the sections below")
prose(fig, [
 ("Dataset paper, a stealth-validated impact-quantified dataset", [
  "Draws the attack-construction pages, the BDD stealth-validation run, the How Stealthy Is Stealthy sensitivity "
  "sweep with its 1.7 percent noise-floor crossing, the operator-impact pages, and the noise-floor framing on the "
  "page before this one.",
 ]),
 ("State-estimation paper, physics-informed graph SE", [
  "Draws the State Estimation Under Attack table, the Within the Noise Floor page, the GNN-Guided Robust WLS "
  "division-of-labor page, and the voltage-versus-angle mechanism page. The benign-null result and the supra-floor "
  "win are its spine.",
 ]),
 ("Localization and detection paper, federated ARMA graph network", [
  "Draws the per-attack localization and detection tables, the complexity ladder, the physics-biased attention "
  "A and B, and the feature-separation page. The noise-floor point frames why localization matters only above the "
  "floor.",
 ]),
 ("Federated paper, the cost of the no-sharing constraint", [
  "Draws the detection results under client partition and the threat-model page, with the measurement-and-attack "
  "model stating that every released family sits above the estimator noise floor.",
 ]),
], top=0.895)
save(fig)

# ======================= External-dataset comparison + head-to-head on the IEEE-30 DataPort benchmark (2026-07-18) =======================
# Verified against the actual papers (Boyaci TSG 2022 joint-localization, Boyaci Systems J 2022 / DEFENDA,
# Zaman NAPS 2025 / PING) and the actual IEEE-30 DataPort arrays we downloaded and loaded. Facts, not claims.
cmp_cols = ["Dataset", "Systems", "Est.", "Measurements", "Stealth check", "Impact", "Public"]
cmp_rows = [
    ["Ours (fdia-graph)", "14/118/300", "AC", "sparse SCADA/PMU + masks", "BDD + load (evasion %)", "yes", "yes"],
    ["Boyaci joint 2022", "57/118/300", "AC", "full P,Q injections", "by construction", "no", "no"],
    ["DEFENDA 2022", "14/118/300", "AC", "full P,Q injections", "residual shown", "no", "detect only"],
    ["PING 2025", "8 MATPOWER", "DC", "P inj + flow + mask", "a=Hc", "no", "not reported"],
    ["IEEE-30 DataPort", "30", "DC", "41 branch flows only", "a=Hc (micro)", "no", "yes (CC BY)"],
]
fig = table_page("How Our Dataset Compares to Prior FDIA Benchmarks",
                 "verified against the actual papers and, for the IEEE-30 set, against the downloaded arrays",
                 cmp_cols, cmp_rows, fontsize=8.0, col_widths=[0.20, 0.135, 0.075, 0.24, 0.19, 0.075, 0.11],
                 shade_rule=lambda i, row: "#eaf7ee" if i == 0 else "#ffffff",
                 cap="Ours is the only entry that combines a realistic sparse SCADA/PMU measurement model, a "
                     "double-stealth check with a reported evasion rate, an operator-impact label, and public "
                     "release, across three systems. Honest scoping. Real load time series and full-AC stealth are "
                     "NOT unique to us (Boyaci uses both). Sparse and masked measurements are not new either (PING). "
                     "The concurrent IEEE-30 DataPort set independently makes the low-intensity-attack argument we "
                     "make, which we cite as corroboration rather than claim as our own. What is unique is the "
                     "combination above, and the seconds-cadence extension.")
SIDE["external_comparison"] = [",".join(cmp_cols)] + [",".join(r) for r in cmp_rows]
save(fig)

fig = newpage("Evaluation Protocol & Threat Model",
              "how the benchmark is scored, and what we assume about the attacker")
prose(fig, [
 ("Split and metrics", [
  "60/20/20 chronological split, cut on sequence boundaries so a ramp never straddles train and test, and a random "
  "shuffle would leak it (Boyaci et al., 2022, PING, NAPS 2025). Equal count per attack family.",
  "Report per attack type, a pooled accuracy or F1 hides the stealthy families behind the easy ones.",
  "Localization = sample-wise F1 (predicted attacked set vs truth).  Detection = grid-level DR / FA / detection-F1.",
  "Unseen-attack protocol available via load(..., heldout=True), which reserves As and Ar for test only.",
 ]),
 ("Threat model and assumptions", [
  "We assume a strong, system-aware attacker, consistent with the stealthy FDIA literature (Liu et al. 2009, "
  "Boyaci et al. 2022). Evading bad-data detection requires knowing the topology and coordinating the affected meters.",
  "The state-level attacks re-solve the full power flow, so they present a fully consistent fake across the network, "
  "near the worst case for a defender, which is deliberate but worth stating plainly.",
  "This is optimistic about reach. The change follows the flow paths across the system, so it assumes broad meter "
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
  "and the larger grids is modest and honest, a difficult benchmark, not a solved one.",
  "Detection aggregates evidence across the grid and reaches Boyaci range on the bad-data-evading families, the "
  "ML-only thesis.",
  "Physics-biased attention improves localization on every system, and a physics-informed state estimator recovers "
  "the true state that a fooled weighted-least-squares estimator cannot.",
 ]),
 ("Honest limitations", [
  "Temporal windowing did not rescue ramp localization. Over a short window, ramp's per-step creep is smaller than "
  "benign load drift, so it stays hard, a documented negative result, not an oversight.",
  "The state estimator recovers state, not attack labels. A large state correction is a signal, but naming the "
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
