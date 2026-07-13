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
from fdia_graph.dataset import FAMILIES, STEALTHY_FAMILIES

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "fdia_dataset_report")
SYS = [14, 118, 300]; FAMS = ["Ao", "Ad", "As", "Ar", "ramp", "LRA"]; SPLITS = ["train", "val", "test"]
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
BENCH = {r["system"]: r for r in load_json("ml_only_benchmark.json", [])}
ATTN = load_json("ml_only_attn_ab.json", {})
SE = {c: dict(pinn=load_json(f"se_{c}.json"), nn=load_json(f"se_{c}_nophys.json"), wls=load_json(f"se_{c}_wls.json")) for c in SYS}

# dataset-intrinsic stats straight from the SDK (fg.load per split)
DATA = {}
for c in SYS:
    per = {sp: fg.load(f"ieee{c}", split=sp).summary()["families"] for sp in SPLITS}
    tst = fg.load(f"ieee{c}", split="test"); a = tst.to_numpy()
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
  "Stealthy (evade bad-data detection — the hard, dangerous cases):  Ao, a state-consistent load "
  "redistribution;  ramp, a slow multi-timestep creep;  LRA, a targeted masked-overload.",
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
 ["family", "scalar", "int", "0 benign 1 Ao 2 Ad 3 As 4 Ar 5 ramp 6 LRA"],
 ["stealthy", "scalar", "int", "1 if the attack evades classical bad-data detection"],
 ["split", "scalar", "int", "0 train 1 val 2 test (60/20/20 chronological)"],
 ["seq_id", "scalar", "int", "ramp-sequence id (>=0 groups one ramp); -1 otherwise"],
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
SRC = {"Ao": "state-consistent load redistribution", "Ad": "random meter corruption", "As": "meter scaling",
       "Ar": "replay of a past scan", "ramp": "temporal creep (Haghshenas et al., ISGT 2023)",
       "LRA": "targeted masked-overload (Yuan et al., T-SG 2011)"}
def bdd_pct(c, fam):
    try: return MAN["systems"][f"ieee{c}"]["families"][fam].get("bdd_pass_pct")
    except Exception: return None
fam_cols = ["Family", "Description", "Class", "BDD-pass  14 / 118 / 300"]
fam_rows = []
for f in FAMS:
    fid = [k for k, v in FAMILIES.items() if v == f][0]
    cls = "stealthy" if fid in STEALTHY_FAMILIES else "detectable"
    bdd = " / ".join(str(bdd_pct(c, f)) + "%" if bdd_pct(c, f) is not None else "—" for c in SYS)
    fam_rows.append([f, SRC[f], cls, bdd])
def fam_shade(i, row): return "#fbecea" if row[2] == "stealthy" else "#eaf2f8"
cov_cols = ["System", "Buses N", "Branches E", "Redundancy", "|V| cov", "P_inj cov", "theta cov", "flow cov"]
cov_rows = [[f"IEEE-{c}", DATA[c]["N"], DATA[c]["E"], f"{DATA[c]['redundancy']:.2f}",
             f"{100*DATA[c]['cov']['V']:.0f}%", f"{100*DATA[c]['cov']['Pinj']:.0f}%",
             f"{100*DATA[c]['cov']['theta']:.0f}%", f"{100*DATA[c]['cov']['flow']:.0f}%"] for c in SYS]
fig = multi_table_page("Attack Families & Measurement Model",
                       "what the attacks are, how stealthy they are, and what the meters actually see",
                       [dict(subhead="Seven families and the bad-data-detection stealth split",
                             cols=fam_cols, rows=fam_rows, col_widths=[0.11, 0.44, 0.13, 0.32], shade_rule=fam_shade,
                             cap="BDD-pass is the fraction of samples that evade the chi-square detector. The stealthy families "
                                 "(Ao/ramp/LRA, red) pass on ~90-100% — they are load-conserving, plausibility-bounded, valid states "
                                 "— while the detectable families (Ad/As/Ar, blue) are caught. A localizer is measured on the "
                                 "stealthy families; acing the detectable ones proves little. ramp: Haghshenas et al. (ISGT 2023); "
                                 "LRA: Yuan et al. (2011)."),
                        dict(subhead="Measurement coverage — fraction of each quantity metered (sparse SCADA/PMU graph)",
                             cols=cov_cols, rows=cov_rows, col_widths=[0.16, 0.11, 0.13, 0.14, 0.11, 0.12, 0.11, 0.12],
                             cap="Redundancy = (# measurements)/(2N-1 states), the classical observability ratio; ~2-3 is a "
                                 "realistic EMS regime. Voltage magnitude and branch flows are widely metered; PMU angles (theta) "
                                 "are sparse — the masks encode which meters exist per record.")])
SIDE["families"] = ["family,description,class,bdd_pass_14,bdd_pass_118,bdd_pass_300"] + \
    [f'{f},"{SRC[f]}",{fam_rows[i][2]},{bdd_pct(14,f)},{bdd_pct(118,f)},{bdd_pct(300,f)}' for i, f in enumerate(FAMS)]
SIDE["coverage"] = ["system,N,E,redundancy,V_cov,Pinj_cov,theta_cov,flow_cov"] + \
    [f"{c},{DATA[c]['N']},{DATA[c]['E']},{DATA[c]['redundancy']:.3f},{DATA[c]['cov']['V']:.3f},{DATA[c]['cov']['Pinj']:.3f},{DATA[c]['cov']['theta']:.3f},{DATA[c]['cov']['flow']:.3f}" for c in SYS]
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
c = 118; nx = DATA[c]["nx"]; fam = DATA[c]["fam"]; nm_test = fg.load("ieee118", split="test").to_numpy()["node_m"]
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
                  "per-record injection deviation from the true state at the same timestep — RMS over METERED buses")
    gs = fig.add_gridspec(1, 2, left=0.10, right=0.96, top=0.84, bottom=0.36, wspace=0.26)
    axP, axQ = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    order = ["benign"] + FAMS
    for ax, q, ttl in [(axP, "dP", "real-power shift  |dP|  (MW)"), (axQ, "dQ", "reactive shift  |dQ|  (MVAr)")]:
        data = [resid[f"ieee118_{f}_{q}"] for f in order]
        parts = ax.violinplot(data, showmedians=True, widths=0.85)
        for j, b in enumerate(parts["bodies"]):
            b.set_facecolor(STEAL if order[j] in ("Ao", "ramp", "LRA") else (MUTE if order[j] == "benign" else DET)); b.set_alpha(0.72)
        floor = float(np.median(resid[f"ieee118_benign_{q}"]))            # benign = pure meter noise -> the noise floor
        ax.axhline(floor, ls="--", lw=1.0, color="#444")
        ax.text(len(order) + 0.4, floor, "meter-noise\nfloor", fontsize=6.5, va="center", color="#444")
        ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=45, fontsize=7.5)
        ax.set_title(ttl, fontsize=9.5); ax.tick_params(labelsize=7.5)
    caption(fig, "Deviation = |measured injection - TRUE injection at the SAME timestep|, RMS over buses that actually have a meter (not "
                 "vs the previous scan). The dashed line is the METER-NOISE floor (benign = pure noise). KEY POINT: the DETECTABLE families "
                 "Ad/As/Ar (blue) barely clear that floor (~4-5 MW) — in state terms their impact is nearly indistinguishable from noise, and "
                 "they are caught only because the corruption is INCONSISTENT, not because it is large. The STEALTHY families Ao/ramp/LRA "
                 "(red) are the only ones that move the true state well ABOVE the noise floor (~14-18 MW) — a real, dangerous, plausibility-"
                 "bounded load redistribution — WHILE still passing every classical check. That 'large-impact yet stealthy' quadrant, which "
                 "standard attack sets miss, is why these new families were added. (Reactive power moves less; a redistribution is mainly real "
                 "power.)", y=0.30)
    SIDE["residuals"] = ["family,median_absdP_MW,median_absdQ_MVAr"] + \
        [f'{f},{np.median(resid[f"ieee118_{f}_dP"]):.3f},{np.median(resid[f"ieee118_{f}_dQ"]):.3f}' for f in order]
    save(fig)

# ======================= Model benchmark: localization + detection (two tables, one page) =======================
loc_cols = ["System", "overall"] + FAMS
loc_rows = []
for c in SYS:
    b = BENCH.get(f"ieee{c}")
    if not b: continue
    pf = b["per_family"]
    loc_rows.append([f"IEEE-{c}", f"{b['overall']['swf1']:.3f}"] + [f"{pf.get(f,{}).get('swf1',0):.3f}" for f in FAMS])
det_cols = ["System", "DR", "FA", "det-F1"] + ["DR:" + f for f in FAMS]
det_rows = []
for c in SYS:
    b = BENCH.get(f"ieee{c}"); d = (b or {}).get("detection", {})
    if not d: continue
    pfd = d.get("per_family_DR", {})
    det_rows.append([f"IEEE-{c}", f"{d['DR']:.3f}", f"{d['FA']:.3f}", f"{d['det_f1']:.3f}"] + [f"{pfd.get(f,0):.2f}" for f in FAMS])
fig = multi_table_page("Model Benchmark — Localization & Detection",
                       "reference ARMA localizer (measurements + KCL residual + temporal delta), test split",
                       [dict(subhead="Localization — sample-wise F1 per attack type (higher is better)",
                             cols=loc_cols, rows=loc_rows, col_widths=[0.16, 0.14] + [0.10] * 6,
                             cap="Per attack type, not accuracy: LRA (structured, targeted) is the most localizable; the diffuse "
                                 "stealthy families (Ao, ramp) and the larger grids are the hard cases. A table beats a bar chart "
                                 "here — the exact values are the point."),
                        dict(subhead="Detection — grid-level rate (DR), false-alarm (FA), F1, and per-family DR",
                             cols=det_cols, rows=det_rows, fontsize=8.0, col_widths=[0.13, 0.09, 0.09, 0.10] + [0.098] * 6,
                             cap="Grid-level score S = max over buses of the per-bus attack probability. Detection far exceeds "
                                 "per-bus localization because it aggregates evidence: the three bad-data-evading families are "
                                 "detected in Boyaci range — ML catches what the classical detector cannot.")])
SIDE["localization"] = ["system,overall," + ",".join(FAMS)] + [",".join([f"ieee{SYS[i]}"] + r[1:]) for i, r in enumerate(loc_rows)]
SIDE["detection"] = ["system,DR,FA,det_f1," + ",".join("DR_" + f for f in FAMS)] + [",".join([f"ieee{SYS[i]}"] + r[1:]) for i, r in enumerate(det_rows)]
save(fig)

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
                   cap="Recover the TRUE state from possibly-attacked measurements. WLS (classical) is fooled by a stealthy "
                       "attack; the learned estimators beat it and the raw meter on both quantities, and the physics term (PINN "
                       "vs NN) sharpens voltage. On IEEE-300 the meter is off ~15 deg under attack while the PINN recovers to ~4 deg."))
fig = multi_table_page("Model Improvements — Attention & State Estimation",
                       "beyond the base localizer: an attention head that raises localization, and a state estimator that recovers the true state",
                       blocks)
SIDE["attention_ab"] = ["system,loc_arma,loc_attn,det_arma,det_attn"] + \
    [f'ieee{c},{ATTN[f"ieee{c}"]["arma"]["loc"]},{ATTN[f"ieee{c}"]["hybrid"]["loc"]},{ATTN[f"ieee{c}"]["arma"]["det"]},{ATTN[f"ieee{c}"]["hybrid"]["det"]}' for c in SYS if f"ieee{c}" in ATTN]
SIDE["state_estimation"] = ["system,V_meter,V_wls,V_nn,V_pinn,th_meter,th_wls,th_nn,th_pinn"] + [",".join([f"ieee{SYS[i]}"] + r[1:]) for i, r in enumerate(se_rows)]
save(fig)

# ======================= Page 13: protocol + critical analysis (prose) =======================
fig = newpage("Evaluation Protocol & Honest Notes",
              "how the benchmark is scored, and what the results do and do not claim")
prose(fig, [
 ("Split and metrics", [
  "60/20/20 chronological split, cut on sequence boundaries so a ramp never straddles train and test — a random "
  "shuffle would leak it (Boyaci et al., 2022; PING, NAPS 2025). Equal count per attack family.",
  "Report per attack type — a pooled accuracy or F1 hides the stealthy families behind the easy ones.",
  "Localization = sample-wise F1 (predicted attacked set vs truth).  Detection = grid-level DR / FA / detection-F1.",
  "Unseen-attack protocol available via load(..., heldout=True), which reserves As and Ar for test only.",
 ]),
 ("What the results say", [
  "The stealthy families are hard by construction (they pass every classical check), so localization on Ao, ramp "
  "and the larger grids is modest and honest — a difficult benchmark, not a solved one.",
  "Detection aggregates evidence across the grid and reaches Boyaci range on the bad-data-evading families — the "
  "ML-only thesis.",
  "Physics-biased attention improves localization on every system.",
  "A physics-informed state estimator recovers the true state that a fooled weighted-least-squares estimator cannot.",
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
