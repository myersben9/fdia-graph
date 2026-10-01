"""The repository's results queries: what each results block in the docs renders from the store.

`tools/results_docs.py` imports this file, which registers every query below with
`fdia_graph.results.query`. A block names a query and its arguments:

    <!-- results: se.comparison angle_mae_deg -->          a whole table
    <!-- results: v se ieee14 prior+huber geo angle_mae_deg -->   one number
    <!-- results: red se ieee14 geo angle_mae_deg wls prior+huber -->   a percent error reduction

`v` and `red` name the experiment by a short alias (ALIASES); a "-" stands for an empty key.
"""

from __future__ import annotations

from typing import Optional

from fdia_graph.results import METRICS, Record, Store, cell, query, table
from fdia_graph.results.render import number

ALIASES = {
    "se": "se.estimators",
    "loc": "localization.common",
    "zs": "localization.zero_shot",
    "fed": "federated.localization",
    "trust": "trust.selection",
    "sloc": "trust.secured_localization",
    "sest": "trust.secured_estimation",
    "wu": "wu26.reproduction",
    "bench": "bench",
}
SYSTEMS = ("ieee14", "ieee118", "ieee300")


def _short(system: str) -> str:
    return system.replace("ieee", "")


def _key(value: str) -> str:
    return "" if value == "-" else value


def _get(store: Store, experiment: str, **keys: object) -> Optional[Record]:
    found = store.latest(experiment, **keys)
    return found[0] if len(found) == 1 else None


def _systems(store: Store, experiment: str) -> list[str]:
    have = {r.system for r in store.latest(experiment)}
    return [s for s in SYSTEMS if s in have]


# ---- one number
@query("v")
def value(
    store: Store,
    alias: str,
    system: str,
    method: str,
    family: str,
    metric: str,
    fmt: str = "",
    scale: str = "1",
    **tags: str,
) -> str:
    """One cell: experiment alias, system, method, family, metric; tags by name (`sd=1` adds the spread)."""
    sd = tags.pop("sd", "") == "1"
    rec = store.one(
        ALIASES[alias], system=_key(system), method=_key(method), family=_key(family), metric=metric, **tags
    )
    return cell(rec, fmt or None, float(scale), sd=sd)


@query("red")
def reduction(
    store: Store,
    alias: str,
    system: str,
    family: str,
    metric: str,
    base: str,
    new: str,
    fmt: str = ".0f",
    **tags: str,
) -> str:
    """The percent error reduction of method `new` over `base` (negative when worse)."""
    keys = dict(system=_key(system), family=_key(family), metric=metric, **tags)
    b = store.one(ALIASES[alias], method=_key(base), **keys).value
    n = store.one(ALIASES[alias], method=_key(new), **keys).value
    return format(100.0 * (1.0 - n / b), fmt)


# ---- state estimation (docs/se)
SE_ROWS = [
    ("wls", "WLS baseline"),
    ("removal", "Residual removal"),
    ("huber", "Adaptive weighting"),
    ("prior+huber", "**Prior + Huber (proposed)**"),
    ("jacobian", "Jacobian weighting"),
    ("prior+huber+gate", "**Prior + Huber + CNN gate**"),
    ("prior+huber+oracle", "Prior + Huber + oracle gate (ceiling)"),
]
SE_FAMILIES = [
    ("benign", "Benign"),
    ("Ad", "Bias (Ad)"),
    ("As", "Scaling (As)"),
    ("Ar", "Replay (Ar)"),
    ("Aq", "Stealthy re-solve (Aq)"),
    ("At", "Slow ramp (At)"),
    ("Al", "Load redistribution (Al)"),
    ("Am", "Multi-snapshot (Am)"),
]
_SE_UNITS = {
    "angle_mae_deg": ("Angle MAE (deg)", 1.0, ".3f"),
    "voltage_mae_pu": ("Voltage MAE (10^-3 pu)", 1e3, ".3f"),
}


def _red(
    store: Store, exp: str, system: str, family: str, metric: str, base: str, new: str, **tags: str
) -> Optional[float]:
    b = _get(store, exp, system=system, family=family, metric=metric, method=base, **tags)
    n = _get(store, exp, system=system, family=family, metric=metric, method=new, **tags)
    return None if b is None or n is None else 100.0 * (1.0 - n.value / b.value)


@query("se.comparison")
def se_comparison(store: Store, metric: str) -> str:
    """Estimator x system, the geometric-mean cell, closed by the proposed estimator's reduction."""
    exp, systems = "se.estimators", _systems(store, "se.estimators")
    label, scale, fmt = _SE_UNITS[metric]
    rows = [[f"*{label}*", *("" for _ in systems)]]
    for m, lab in SE_ROWS:
        cells = [
            cell(_get(store, exp, system=s, method=m, family="geo", metric=metric), fmt, scale)
            for s in systems
        ]
        if lab.startswith("**"):
            cells = [f"**{c}**" if c else c for c in cells]
        rows.append([lab, *cells])
    reds = [_red(store, exp, s, "geo", metric, "wls", "prior+huber") for s in systems]
    rows.append(["WLS error reduction", *("" if r is None else f"{r:.0f}%" for r in reds)])
    return table(["Estimator", *(f"IEEE {_short(s)}" for s in systems)], rows)


@query("se.families")
def se_families(store: Store) -> str:
    """Per family: the WLS error and the proposed estimator's percent reduction, every system."""
    exp, systems = "se.estimators", _systems(store, "se.estimators")
    blocks = [
        ("Base angle (deg)", "angle_mae_deg", 1.0, ".3f"),
        ("Base volt (10^-3)", "voltage_mae_pu", 1e3, ".2f"),
        ("Angle red. (%)", "angle_mae_deg", None, ".0f"),
        ("Volt red. (%)", "voltage_mae_pu", None, ".0f"),
    ]
    rows = []
    for fam, lab in SE_FAMILIES:
        cells = []
        for _, metric, scale, fmt in blocks:
            for s in systems:
                base = _get(store, exp, system=s, method="wls", family=fam, metric=metric)
                if base is None:
                    cells.append("n/a")
                elif scale is None:
                    cells.append(format(_red(store, exp, s, fam, metric, "wls", "prior+huber"), fmt))
                else:
                    cells.append(cell(base, fmt, scale))
        rows.append([lab, *cells])
    return table(["Family", *(f"{b} {_short(s)}" for b, *_ in blocks for s in systems)], rows)


@query("se.gated")
def se_gated(store: Store) -> str:
    """Angle MAE of the proposed estimator alone, with the CNN gate and with the oracle gate."""
    exp, systems = "se.estimators", _systems(store, "se.estimators")
    arms = ("prior+huber", "prior+huber+gate", "prior+huber+oracle")
    groups = [
        ("Aq stealthy re-solve", ("Aq",)),
        ("Ad / As / Ar in place", ("Ad", "As", "Ar")),
        ("At slow ramp", ("At",)),
        ("Al redistribution", ("Al",)),
        ("Am multi-snapshot", ("Am",)),
        ("geometric mean", ("geo",)),
    ]
    rows = []
    for lab, fams in groups:
        cells = []
        for s in systems:
            fmt = ".3f" if s == "ieee14" else ".4f"
            for a in arms:
                vals = [
                    cell(_get(store, exp, system=s, method=a, family=f, metric="angle_mae_deg"), fmt)
                    for f in fams
                ]
                cells.append(" / ".join(vals))
        rows.append([lab, *cells])
    head = [
        "angle MAE (deg)",
        *(f"{a} {_short(s)}" for s in systems for a in ("proposed", "+ CNN gate", "+ oracle")),
    ]
    return table(head, rows)


# ---- localization (docs/localization)
LOC_COMMON = [
    ("swing", "Swing threshold"),
    ("delta", "Delta threshold"),
    ("residual", "Residual (LNR)"),
    ("mlp", "Per-bus MLP"),
    ("cnn", "**1D CNN**"),
    ("cnn+jac", "1D CNN + Jacobian (C)"),
    ("cnn+prev", "1D CNN + previous swing"),
    ("cnn+prev+jac", "1D CNN + previous swing + Jacobian"),
]
LOC_ZERO_SHOT = [
    ("mlp", "Per-bus MLP"),
    ("cnn", "**1D CNN**"),
    ("cnn+prev", "1D CNN + previous swing"),
    ("swing", "Swing threshold"),
]
LOC_ABLATION = [
    ("cnn_meas", "A: measurements only"),
    ("cnn", "B: measurements + temporal (the papers' 14)"),
    ("cnn+jac", "C: B + Jacobian features"),
    ("cnn_jac", "D: Jacobian features only"),
    ("cnn+prev", "E: B + the previous frame's swing"),
    ("cnn+prev+jac", "F: E + Jacobian features"),
]


def _f1_dr_fr(
    store: Store,
    exp: str,
    rows: list[tuple[str, str]],
    fmts: tuple[str, str, str],
    first: str,
    bold_best: bool = False,
) -> str:
    """F1 / DR / FR per system; `bold_best` marks each system's best F1."""
    systems = _systems(store, exp)
    best = {
        s: max(
            (
                r.value
                for m, _ in rows
                for r in store.latest(exp, system=s, method=m, family="all", metric="macro_f1")
            ),
            default=None,
        )
        for s in systems
    }
    body = []
    for m, lab in rows:
        cells = []
        for s in systems:
            for metric, fmt in zip(("macro_f1", "macro_dr", "macro_fr"), fmts):
                rec = _get(store, exp, system=s, method=m, family="all", metric=metric)
                text = cell(rec, fmt)
                if bold_best and rec is not None and metric == "macro_f1" and rec.value == best[s]:
                    text = f"**{text}**"
                cells.append(text)
        body.append([lab, *cells])
    head = [first, *(f"{k} {_short(s)}" for s in systems for k in ("F1", "DR", "FR"))]
    return table(head, body)


@query("loc.table")
def loc_table(store: Store, protocol: str) -> str:
    """F1 / DR / FR per system for one protocol: common, zero_shot or ablation."""
    if protocol == "common":
        return _f1_dr_fr(store, "localization.common", LOC_COMMON, (".4f", ".4f", ".4f"), "Method")
    if protocol == "zero_shot":
        return _f1_dr_fr(store, "localization.zero_shot", LOC_ZERO_SHOT, (".4f", ".4f", ".4f"), "Method")
    return _f1_dr_fr(
        store,
        "localization.zero_shot",
        LOC_ABLATION,
        (".3f", ".3f", ".4f"),
        "Model (1D CNN, zero-shot)",
        True,
    )


# ---- federated localization (docs/federated)
FED_ROWS = [
    (m, k, f"{lab}, K = {k}")
    for m, lab in (("cnn", "1D CNN"), ("mlp", "Per-bus MLP"))
    for k in ("1", "2", "3")
]


@query("fed.table")
def fed_table(store: Store, kind: str) -> str:
    """The federated tables: `table4` (F1 / DR / FR over every record), `benign` (FR over benign
    records) or `families` (per-family node F1), mean ± sd over seeds."""
    exp = "federated.localization"
    systems = _systems(store, exp)
    if kind == "families":
        fams = ("Aq", "Ad", "As", "Ar")
        body = [
            [
                lab,
                *(
                    cell(
                        _get(
                            store, exp, system=s, method=m, family=f, metric="macro_f1", clients=k, pool="all"
                        ),
                        ".3f",
                    )
                    for s in systems
                    for f in fams
                ),
            ]
            for m, k, lab in FED_ROWS
        ]
        return table(["Model", *(f"{f} {_short(s)}" for s in systems for f in fams)], body)
    cols = [("macro_f1", "F1", ".3f"), ("macro_dr", "DR", ".3f"), ("macro_fr", "FR", ".4f")]
    pool = "all"
    if kind == "benign":
        cols, pool = [("macro_fr", "FR", ".5f")], "benign"
    body = [
        [
            lab,
            *(
                cell(
                    _get(store, exp, system=s, method=m, family="all", metric=metric, clients=k, pool=pool),
                    fmt,
                    sd=True,
                )
                for s in systems
                for metric, _, fmt in cols
            ),
        ]
        for m, k, lab in FED_ROWS
    ]
    return table(["Model", *(f"{name} {_short(s)}" for s in systems for _, name, _ in cols)], body)


# ---- trusted meters (docs/trust)
_STEALTHY = ("Aq", "At", "Al", "Am")
_SELECTORS = (("greedy", "greedy"), ("dqn", "DQN"))


@query("trust.detection")
def trust_detection(store: Store) -> str:
    """Residual detection rate per family, before → after securing the selection."""
    exp = "trust.selection"
    body = []
    for s in _systems(store, exp):
        for sel, lab in _SELECTORS:

            def rate(fam: str, which: str, s: str = s, sel: str = sel) -> str:
                return cell(_get(store, exp, system=s, method=sel, family=fam, metric=which), ".2f")

            cells = [f"{rate(f, 'detected_before')} → {rate(f, 'detected_after')}" for f in _STEALTHY]
            inplace = [
                " / ".join(rate(f, w) for f in ("Ad", "As", "Ar"))
                for w in ("detected_before", "detected_after")
            ]
            body.append([s, lab, *cells, " → ".join(inplace)])
    return table(["system", "selector", *_STEALTHY, "Ad / As / Ar"], body, numeric_from=2)


_COPIES = (("plain", "plain"), ("greedy", "greedy"), ("dqn", "DQN"))


@query("trust.secured_loc")
def trust_secured_loc(store: Store, system: str) -> str:
    """Node F1 (detection rate) per stealthy family on the plain and secured copies."""
    exp = "trust.secured_localization"
    body = []
    for m, mlab in (("residual", "residual"), ("cnn+jac", "1D CNN + Jacobian")):
        for copy, clab in _COPIES:
            keys = dict(system=system, method=m, secured=copy)
            cells = [
                f"{cell(_get(store, exp, family=f, metric='node_f1', **keys), '.2f')} "
                f"({cell(_get(store, exp, family=f, metric='detection_rate', **keys), '.2f')})"
                for f in _STEALTHY
            ]
            tail = f"{cell(_get(store, exp, family='all', metric='macro_f1', **keys), '.3f')} / " + cell(
                _get(store, exp, family="all", metric="macro_fr", **keys), ".4f"
            )
            body.append([mlab, clab, *cells, tail])
    return table(["localizer", "copy", *_STEALTHY, "macro-F1 / benign FA"], body, numeric_from=2)


_SECURED_EST = [
    ("prior+huber", "prior + Huber"),
    ("prior+huber+cnn", "prior + Huber + CNN gate"),
    ("prior+huber+cnn+trust", "prior + Huber + CNN gate, secured meters exempt"),
    ("prior+huber+residual+trust", "prior + Huber + residual gate, secured meters exempt"),
    ("prior+huber+oracle", "prior + Huber + oracle gate"),
    ("prior+huber+oracle+trust", "prior + Huber + oracle gate, secured meters exempt"),
]


@query("trust.secured_est")
def trust_secured_est(store: Store, system: str, fmt: str = ".3f") -> str:
    """Geometric-mean angle MAE per estimator on the plain and secured copies."""
    exp = "trust.secured_estimation"
    body = [
        [
            lab,
            *(
                cell(
                    _get(
                        store, exp, system=system, method=m, family="geo", metric="angle_mae_deg", secured=c
                    ),
                    fmt,
                )
                for c, _ in _COPIES
            ),
        ]
        for m, lab in _SECURED_EST
    ]
    return table(["estimator", *(lab for _, lab in _COPIES)], body)


# ---- benchmarks (docs/reference/BENCHMARKS.md)
_BENCH = ("generate", "wls", "huber", "prior+huber")


@query("bench.table")
def bench_table(store: Store) -> str:
    """One line per stored `bench` run: its date, the ms per record of each timed step, the SDK version,
    the machine and the fixture recipe's tag."""
    body = []
    for prov in sorted((p for p in store.runs() if p.experiment == "bench"), key=lambda p: p.timestamp):
        recs = {r.method: r for r in store.query("bench", run_id=prov.run_id)}
        recipe = next((r.tag("recipe") for r in recs.values()), "")
        body.append(
            [
                prov.timestamp[:10],
                *(cell(recs.get(m), ".3f") for m in _BENCH),
                prov.sdk_version,
                prov.note,
                recipe,
            ]
        )
    head = ["date", *(f"{m} ms/record" for m in _BENCH), "version", "machine", "recipe"]
    return table(head, body)


# ---- [WU26] defense plan (docs/plans/WU_DEFENSE_PLAN.md)
def _count_pair(store: Store, exp: str, **keys: str) -> Optional[tuple[int, int]]:
    d, c = (_get(store, exp, metric=m, **keys) for m in ("devices", "channels"))
    return None if d is None or c is None else (int(d.value), int(c.value))


def _pair_text(pair: Optional[tuple[int, int]]) -> str:
    if pair is None:
        return ""
    return "none converged" if pair[0] < 0 else f"{pair[0]} / {pair[1]}"


def _rise(before: Optional[tuple[int, int]], after: Optional[tuple[int, int]]) -> str:
    if before is None or after is None or min(*before, *after) < 0:
        return ""
    return " / ".join(f"{100.0 * (a / b - 1.0):+.1f}%" for a, b in zip(after, before))


@query("wu.prototype")
def wu_prototype(store: Store) -> str:
    """The plan's IEEE-14 prototype: undefended, Wu's schedule (own bus), the rise, Table II beside it."""
    exp = "wu26.prototype"
    paper = {("1", "1.1"): "25.6% / 23.9%", ("2", "1.1"): "35.2% / 27.0%"}  # [WU26] Table II, as published
    body = []
    for s, k, cap in (("1", "1.1", "0.5"), ("2", "1.1", "0.5"), ("1", "1.2", "0.5"), ("2", "1.2", "0.5"), ("1", "1.2", "off"), ("2", "1.2", "off")):
        keys = dict(scenario=s, k=k, load_cap=cap)
        before = _count_pair(store, exp, trust="none", **keys)
        after = _count_pair(store, exp, trust="wu", **keys)
        label = f"S{s}{' (3-4, 6-11)' if s == '1' else ' (1-2, 4-5)'}, {k}" + (", no load cap" if cap == "off" else "")
        body.append([label, _pair_text(before), _pair_text(after), _rise(before, after), paper.get((s, k), "")])
    head = ["scenario, k", "undefended: devices / channels", "Wu's schedule, own bus: devices / channels", "increase: devices / channels", "Table II (Sol 1 / Sol 2)"]
    return table(head, body, numeric_from=5)


@query("wu.methods")
def wu_methods(store: Store) -> str:
    """The search against [WU26]'s row reduction, undefended and under the paper's schedule."""
    exp = "wu26.method_compare"
    rows = [
        ("ieee14", "1", "1.1", "Lines 3-4 and 6-11, 1.1", "7 devices", "+25.6% (Sol 1), +23.9% (Sol 2)", "1, 4, 6, 13"),
        ("ieee14", "2", "1.1", "Lines 1-2 and 4-5, 1.1", "9 devices", "+35.2% (Sol 1), +27.0% (Sol 2)", "4, 6, 1, 13"),
        ("ieee14", "1", "1.2", "Lines 3-4 and 6-11, 1.2", "", "", "the paper's"),
        ("ieee14", "2", "1.2", "Lines 1-2 and 4-5, 1.2", "", "", "the paper's"),
        ("ieee118", "1", "1.2", "IEEE-118 lines 84-85 and 99-100, 1.2", "", "+16.4% mean, 3-5 extra devices", "the 11 PMUs, one per snapshot"),
    ]  # the paper's column is [WU26]'s published numbers (Figs. 4, 12, Table II)
    body = []
    for system, s, k, label, paper_none, paper_wu, schedule in rows:
        cells = {}
        for method in ("search", "rref"):
            base = _count_pair(store, exp, system=system, method=method, scenario=s, k=k, trust="none")
            wu = _count_pair(store, exp, system=system, method=method, scenario=s, k=k, trust="wu")
            cells[method] = (base, wu)
        body.append([label, "none", *(_pair_text(cells[m][0]) for m in ("search", "rref")), paper_none])
        wu_cells = []
        for m in ("search", "rref"):
            base, wu = cells[m]
            text = "no support reaches both ratings" if wu is not None and wu[0] < 0 else _pair_text(wu)
            rise = _rise(base, wu)
            wu_cells.append(f"{text} ({rise})" if rise else text)
        body.append([label, schedule, *wu_cells, paper_wu])
    return table(["Scenario, k", "Trusted PMUs", "Search: devices / channels", "RREF: devices / channels", "Paper"], body, numeric_from=5)


# ---- spans over many records
@query("span")
def span(store: Store, experiment: str, metric: str, fmt: str = "", **keys: str) -> str:
    """"lo to hi" of `metric` over every newest record the keys select (one number when they agree)."""
    vals = sorted(r.value for r in store.latest(experiment, metric=metric, **keys))
    spec = fmt or METRICS[metric].fmt
    lo, hi = number(vals[0], spec), number(vals[-1], spec)
    return lo if lo == hi else f"{lo} to {hi}"


# ---- the certifier (docs/plans/RELAX_CERTIFIER_PLAN.md)
_CUTS = (("socp", "second-order cone"), ("bounds", "+ bounds"), ("bounds+qc", "+ qc"), ("bounds+qc+cycle", "+ cycle"))


@query("certify.ablation")
def certify_ablation(store: Store) -> str:
    """Per cut family: certificates on the paper scenarios, Am and At, Am gaps, mismatch and time."""
    import statistics

    exp = "certify.ablation"
    body = []
    for cut, label in _CUTS:
        def rows(metric: str, family: str, cut: str = cut) -> list[Record]:
            return store.latest(exp, method=cut, family=family, metric=metric)

        paper_gaps = ", ".join(str(int(r.value)) for r in sorted(rows("gap", ""), key=lambda r: r.tag("episode")))
        paper = f"{int(sum(r.value for r in rows('certified', '')))} of {len(rows('certified', ''))} certified (gaps {paper_gaps})"
        am = f"{int(sum(r.value for r in rows('certified', 'Am')))} of {len(rows('certified', 'Am'))} certified"
        at = f"{int(sum(r.value for r in rows('certified', 'At')))} certified, {int(sum(r.value for r in rows('uncertain', 'At')))} uncertain"
        gaps = sorted(r.value for r in rows("gap", "Am"))
        mism = statistics.median(r.value for r in rows("mismatch_mw", "Am"))
        secs = sorted(r.value for r in store.latest(exp, method=cut, metric="seconds", stage="relax"))
        tight = sorted(r.value for r in store.latest(exp, method=cut, metric="seconds", stage="tightening"))
        time = (f"{tight[0]:.0f} to {tight[-1]:.0f} tightening, " if tight else "") + f"{secs[0]:.0f} to {secs[-1]:.0f} solve"
        body.append([label, paper, am, at, f"{gaps[0]:.0f} to {gaps[-1]:.0f}, median {statistics.median(gaps):.0f}", f"{mism:.0f}", time])
    head = ["family", "paper", "`Am`", "`At`", "`Am` gaps", "median mismatch, MW (`Am`)", "seconds"]
    return table(head, body, numeric_from=7)
