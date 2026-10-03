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

from fdia_graph.errors import NoSuchResult
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
    """The newest record the keys select, None when there is none (an n/a cell); several is a
    selection that left a key open and raises, so a table never shows an arbitrary one."""
    found = store.latest(experiment, **keys)
    if len(found) > 1:
        raise NoSuchResult(f"{experiment} {keys}: {len(found)} records match, expected at most one")
    return found[0] if found else None


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


def _rate(store: Store, system: str, sel: str, fam: str, which: str) -> str:
    return cell(_get(store, "trust.selection", system=system, method=sel, family=fam, metric=which), ".2f")


def _detection_row(store: Store, system: str, sel: str, label: str) -> list[str]:
    cells = [
        f"{_rate(store, system, sel, f, 'detected_before')} → {_rate(store, system, sel, f, 'detected_after')}"
        for f in _STEALTHY
    ]
    inplace = [
        " / ".join(_rate(store, system, sel, f, w) for f in ("Ad", "As", "Ar"))
        for w in ("detected_before", "detected_after")
    ]
    return [system, label, *cells, " → ".join(inplace)]


@query("trust.detection")
def trust_detection(store: Store) -> str:
    """Residual detection rate per family, before → after securing the selection."""
    body = [
        _detection_row(store, s, sel, lab)
        for s in _systems(store, "trust.selection")
        for sel, lab in _SELECTORS
    ]
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
    for s, k, cap in (
        ("1", "1.1", "0.5"),
        ("2", "1.1", "0.5"),
        ("1", "1.2", "0.5"),
        ("2", "1.2", "0.5"),
        ("1", "1.2", "off"),
        ("2", "1.2", "off"),
    ):
        keys = dict(scenario=s, k=k, load_cap=cap)
        before = _count_pair(store, exp, trust="none", **keys)
        after = _count_pair(store, exp, trust="wu", **keys)
        label = f"S{s}{' (3-4, 6-11)' if s == '1' else ' (1-2, 4-5)'}, {k}" + (
            ", no load cap" if cap == "off" else ""
        )
        body.append(
            [label, _pair_text(before), _pair_text(after), _rise(before, after), paper.get((s, k), "")]
        )
    head = [
        "scenario, k",
        "undefended: devices / channels",
        "Wu's schedule, own bus: devices / channels",
        "increase: devices / channels",
        "Table II (Sol 1 / Sol 2)",
    ]
    return table(head, body, numeric_from=5)


@query("wu.methods")
def wu_methods(store: Store) -> str:
    """The search against [WU26]'s row reduction, undefended and under the paper's schedule."""
    exp = "wu26.method_compare"
    rows = [
        (
            "ieee14",
            "1",
            "1.1",
            "Lines 3-4 and 6-11, 1.1",
            "7 devices",
            "+25.6% (Sol 1), +23.9% (Sol 2)",
            "1, 4, 6, 13",
        ),
        (
            "ieee14",
            "2",
            "1.1",
            "Lines 1-2 and 4-5, 1.1",
            "9 devices",
            "+35.2% (Sol 1), +27.0% (Sol 2)",
            "4, 6, 1, 13",
        ),
        ("ieee14", "1", "1.2", "Lines 3-4 and 6-11, 1.2", "", "", "the paper's"),
        ("ieee14", "2", "1.2", "Lines 1-2 and 4-5, 1.2", "", "", "the paper's"),
        (
            "ieee118",
            "1",
            "1.2",
            "IEEE-118 lines 84-85 and 99-100, 1.2",
            "",
            "+16.4% mean, 3-5 extra devices",
            "the 11 PMUs, one per snapshot",
        ),
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
    return table(
        ["Scenario, k", "Trusted PMUs", "Search: devices / channels", "RREF: devices / channels", "Paper"],
        body,
        numeric_from=5,
    )


# ---- spans over many records
@query("span")
def span(store: Store, experiment: str, metric: str, fmt: str = "", **keys: str) -> str:
    """ "lo to hi" of `metric` over every newest record the keys select (one number when they agree)."""
    vals = sorted(r.value for r in store.latest(experiment, metric=metric, **keys))
    spec = fmt or METRICS[metric].fmt
    lo, hi = number(vals[0], spec), number(vals[-1], spec)
    return lo if lo == hi else f"{lo} to {hi}"


# ---- generation (docs/guides/generation.md)
_AM_RUNS = (
    ("1", "none", "one line, before the D16 bounds"),
    ("1", "d16", "one line, D16 bounds"),
    ("2", "d16", "two lines, D16 bounds (default)"),
)


@query("gen.am")
def gen_am(store: Store) -> str:
    """The overload episodes generated per system and recipe: built and fallen back, mean devices and
    channels, the largest change on a channel (median and worst episode), time (these runs predate the removal of
    the search's proof bookkeeping; its proven share is no longer shown)."""
    exp = "generation.am_overload"
    body = []
    for system in ("ieee14", "ieee30", "ieee118"):
        for lines, bounds, label in _AM_RUNS:
            keys = dict(system=system, lines=lines, edge_bounds=bounds)
            frames = _get(store, exp, metric="frames", **keys)
            if frames is None:
                continue
            body.append(
                [
                    f"IEEE-{_short(system)}",
                    label,
                    cell(frames, "d"),
                    cell(_get(store, exp, metric="episodes", stage="built", **keys), "d"),
                    cell(_get(store, exp, metric="episodes", stage="fallen_back", **keys), "d"),
                    cell(_get(store, exp, metric="devices", **keys), ".1f"),
                    cell(_get(store, exp, metric="channels", **keys), ".1f"),
                    f"{cell(_get(store, exp, metric='max_change_pu', stat='median', **keys))} / {cell(_get(store, exp, metric='max_change_pu', stat='max', **keys))}",
                    cell(_get(store, exp, metric="seconds_per_episode", **keys), ".0f"),
                ]
            )
    head = [
        "system",
        "recipe",
        "frames",
        "episodes",
        "fallen back",
        "devices (mean)",
        "channels (mean)",
        "largest change, pu (median / max episode)",
        "s per episode",
    ]
    return table(head, body, numeric_from=2)


@query("wu.smax")
def wu_smax(store: Store) -> str:
    """[WU26]'s IEEE-14 scenarios at ratings k times the window's peak flow: devices, channels and
    the largest change, or no attack."""
    exp = "wu26.smax_sensitivity"
    ks = sorted(
        {r.tag("k") for r in store.latest(exp)},
        key=lambda k: (k == "pglib", float(k) if k != "pglib" else 0.0),
    )
    body = []
    for s, label in (("1", "lines 3-4 and 6-11"), ("2", "lines 1-2 and 4-5")):
        row = [label]
        for k in ks:
            d = _get(store, exp, scenario=s, k=k, metric="devices")
            c = _get(store, exp, scenario=s, k=k, metric="channels")
            m = _get(store, exp, scenario=s, k=k, metric="max_change_pu")
            row.append(
                "no attack"
                if d is None or d.value < 0
                else f"{int(d.value)} / {int(c.value) if c else ''} / {cell(m)}"
            )
        body.append(row)
    return table(["scenario (devices / channels / largest change, pu)", *(f"k = {k}" for k in ks)], body)


def _per_family(store: Store, method: str, split: str, metric: str, stage: str = "") -> str:
    extra = {"stage": stage} if stage else {}
    fams = method.split("+")
    return " / ".join(
        cell(
            _get(
                store, "generation.split_first", method=method, split=split, family=f, metric=metric, **extra
            ),
            "d",
        )
        for f in fams
    )


def _split_line(store: Store, method: str, split: str) -> list[str]:
    exp = "generation.split_first"
    return [
        method,
        split,
        cell(_get(store, exp, method=method, split=split, metric="frames"), "d"),
        cell(_get(store, exp, method=method, split=split, metric="attacked_frac", family=""), ".3f"),
        _per_family(store, method, split, "episodes", "requested"),
        _per_family(store, method, split, "episodes", "built"),
        _per_family(store, method, split, "redraws"),
        _per_family(store, method, split, "shortfall"),
    ]


@query("gen.split_first")
def gen_split_first(store: Store) -> str:
    """The full IEEE-14 split-first builds: per dataset and split, frames, the attacked fraction, the
    episodes requested and built per family, redraws and shortfall, and the build time."""
    body = []
    for method in ("At", "Am", "At+Am"):
        body += [_split_line(store, method, split) for split in ("train", "val", "test")]
        minutes = cell(_get(store, "generation.split_first", method=method, metric="minutes"), ".0f")
        body.append([method, "build", "", "", "", "", "", f"{minutes} min"])
    head = ["dataset", "split", "frames", "attacked", "episodes asked", "built", "redraws", "shortfall"]
    return table(head, body, numeric_from=2)


# ---- a data release's statistics (docs/reference/EXAMPLES.md)
_LADDER = ("ieee14", "ieee30", "ieee57", "ieee89", "ieee118", "ieee145", "ieee200", "ieee300")
_FAMILY_LABELS = (
    ("benign", "benign (0)"),
    ("Aq", "`Aq` stealthy load-scale"),
    ("Ad", "`Ad` meter corruption"),
    ("As", "`As` meter scaling"),
    ("Ar", "`Ar` replay"),
    ("At", "`At` temporal ramp"),
    ("Al", "`Al` load redistribution"),
    ("Am", "`Am` multi-snapshot"),
)


def _int(rec: Optional[Record]) -> str:
    return "" if rec is None else f"{int(rec.value):,}"


@query("data.sizes")
def data_sizes(store: Store) -> str:
    """Per system: buses, branches, frames per split and episodes."""
    exp = "data.release_stats"
    body = []
    for s in _LADDER:
        frames = [
            _int(_get(store, exp, system=s, split=sp, family="", metric="frames"))
            for sp in ("all", "train", "val", "test")
        ]
        body.append(
            [
                s,
                _int(_get(store, exp, system=s, metric="buses")),
                _int(_get(store, exp, system=s, metric="branches")),
                *frames,
                _int(_get(store, exp, system=s, split="all", metric="episodes")),
            ]
        )
    return table(["system", "N buses", "E branches", "frames", "train", "val", "test", "episodes"], body)


@query("data.families")
def data_families(store: Store, system: str) -> str:
    """Frames of each family per split on one system."""
    exp = "data.release_stats"
    body = []
    for fam, label in _FAMILY_LABELS:
        cells = [
            _get(store, exp, system=system, split=sp, family=fam, metric="frames")
            for sp in ("train", "val", "test")
        ]
        total = sum(int(c.value) for c in cells if c is not None)
        body.append([label, *(_int(c) for c in cells), f"{total:,}"])
    return table(["family", "train", "val", "test", "total"], body)


def _quantiles(store: Store, system: str, quantity: str, which: tuple[str, ...], fmt: str) -> str:
    return " / ".join(
        cell(_get(store, "data.release_stats", system=system, metric="quantile", quantity=quantity, q=w), fmt)
        for w in which
    )


@query("data.states")
def data_states(store: Store) -> str:
    """Per system, |V| at p1 / median / p99 and theta min / median / max over the operating pool."""
    body = [
        [
            s,
            _quantiles(store, s, "V_pu", ("p1", "p50", "p99"), ".3f"),
            _quantiles(store, s, "theta_deg", ("min", "p50", "max"), ".0f").replace("-", "−"),
        ]
        for s in _LADDER
    ]
    return table(["system", "\\|V\\| p1 / med / p99 (pu)", "θ min / med / max (deg)"], body)


# ---- the fewest-tamper search's speed (docs/reference/BENCHMARKS.md)
def _secs(x: float) -> str:
    """Seconds to one decimal from 1 s up, two below."""
    return f"{x:.1f}" if x >= 1 else f"{x:.2f}"


@query("search.speed")
def search_speed(store: Store, pr: str) -> str:
    """Median seconds per episode, before to after one change, per system, family and thread setting."""
    import statistics

    exp = "search.speed"
    body = []
    for s in ("ieee14", "ieee30", "ieee118"):
        recs = store.latest(exp, system=s, pr=pr, metric="seconds")
        if not recs:
            continue
        counts = {f: len({r.tag("episode") for r in recs if r.family == f}) for f in ("Am", "At")}
        cells = []
        for threads in ("default", "one"):
            for fam in ("Am", "At"):
                med = {
                    st: statistics.median(
                        r.value
                        for r in recs
                        if r.family == fam and r.tag("threads") == threads and r.tag("stage") == st
                    )
                    for st in ("before", "after")
                }
                cells.append(f"{_secs(med['before'])} to {_secs(med['after'])}")
        body.append([f"IEEE-{_short(s)} ({counts['Am']} Am, {counts['At']} At)", *cells])
    head = ["system", "default threads, Am", "default threads, At", "one thread, Am", "one thread, At"]
    return table(head, body)


# ---- the parallel Am design stage (docs/guides/generation.md)
@query("gen.workers")
def gen_workers(store: Store) -> str:
    """Am-only IEEE-14 generation per worker count: wall time, the speed-up over one worker, the
    episodes built and whether the file equals the one-worker file byte for byte."""
    exp = "generation.parallel_design"
    rows = sorted({r.method for r in store.latest(exp)}, key=lambda m: int(m.split("=")[1]))
    one = _get(store, exp, method="workers=1", metric="seconds")
    body = []
    for method in rows:
        sec = _get(store, exp, method=method, metric="seconds")
        speed = number(one.value / sec.value, ".1f") + "x" if one and sec else "n/a"
        same = _get(store, exp, method=method, metric="identical")
        body.append(
            [
                method.split("=")[1],
                cell(sec, ".0f"),
                speed,
                cell(_get(store, exp, method=method, metric="episodes"), "d"),
                "yes" if same and same.value == 1.0 else "no",
            ]
        )
    frames = next(
        (
            t.split("=")[1]
            for r in store.latest(exp, metric="seconds")
            for t in r.tags
            if t.startswith("frames=")
        ),
        "",
    )
    return table([f"workers ({frames} frames)", "seconds", "speed-up", "Am episodes", "same file"], body)


# ---- the attack search and the [WU26] reproduction (docs/wu26/README.md)
@query("search.solvers")
def search_solvers(store: Store) -> str:
    """The pinned IEEE-14 searches with the least-norm Gauss-Newton solve against scipy's least squares:
    devices, channels and seconds per search."""
    exp = "search.solver_compare"
    cases = sorted({(r.family, r.tag("load_cap"), r.tag("onset"), r.tag("lines")) for r in store.latest(exp)})
    body = []
    for fam, cap, onset, lines in cases:
        keys = dict(family=fam, load_cap=cap, onset=onset, lines=lines)
        what = f"lines {lines}" if fam == "Am" else f"ramp draw {lines}"
        row = [f"`{fam}` onset {onset}, {what}, cap {cap}"]
        for method in ("gauss_newton", "least_squares"):
            d = _get(store, exp, method=method, metric="devices", **keys)
            c = _get(store, exp, method=method, metric="channels", **keys)
            s = _get(store, exp, method=method, metric="seconds", **keys)
            attack = "none" if d is None or d.value < 0 else f"{cell(d, 'd')} / {cell(c, 'd')}"
            row += [attack, cell(s, ".1f")]
        body.append(row)
    head = ["search", "Gauss-Newton (devices / channels)", "s", "least squares (devices / channels)", "s"]
    return table(head, body)


def _defense_cell(store: Store, exp: str, **keys: str) -> str:
    """ "undefended -> trusted" devices / channels of one case, "none" where no attack meets the goal."""
    out = []
    for defense in ("undefended", "trust"):
        feasible = _get(store, exp, metric="feasible", defense=defense, **keys)
        d = _get(store, exp, metric="devices", defense=defense, **keys)
        c = _get(store, exp, metric="channels", defense=defense, **keys)
        no = d is None or (feasible is not None and feasible.value == 0)
        out.append("none" if no else f"{cell(d, 'd')} / {cell(c, 'd')}")
    return " → ".join(out)


@query("minlp.rating_delta")
def minlp_rating_delta(store: Store) -> str:
    """[WU26]'s scenarios with ratings at the window's last flow plus delta: devices / channels without
    and with the paper's trust schedule (load cap on, the paper's area)."""
    exp = "minlp.rating_delta"
    deltas = sorted({r.tag("delta_pu") for r in store.latest(exp)}, key=float)
    cases = (
        ("ieee14", "0", "IEEE-14, lines 3-4 and 6-11"),
        ("ieee14", "1", "IEEE-14, lines 1-2 and 4-5"),
        ("ieee118", "0", "IEEE-118, lines 84-85 and 99-100"),
    )
    body = [
        [label, *(_defense_cell(store, exp, system=s, scenario=sc, delta_pu=d) for d in deltas)]
        for s, sc, label in cases
    ]
    return table(["case (undefended → trusted, devices / channels)", *(f"+{d} pu" for d in deltas)], body)


@query("minlp.cap_trust")
def minlp_cap_trust(store: Store) -> str:
    """The same scenarios without the load cap and with the neighbour reading of trust (a trusted PMU's
    branch currents secured too), against the defaults."""
    exp = "minlp.cap_trust_grid"
    cases = (
        ("ieee14", "0", "IEEE-14, lines 3-4 and 6-11"),
        ("ieee14", "1", "IEEE-14, lines 1-2 and 4-5"),
        ("ieee118", "0", "IEEE-118, lines 84-85 and 99-100"),
    )
    combos = sorted({(r.method, r.tag("load_cap")) for r in store.latest(exp)})
    deltas = sorted({r.tag("delta_pu") for r in store.latest(exp)}, key=float)
    body = []
    for s, sc, label in cases:
        for d in deltas:
            row = [f"{label}, +{d} pu"]
            for method, cap in combos:
                has = store.latest(exp, system=s, method=method, scenario=sc, delta_pu=d, load_cap=cap)
                row.append(
                    _defense_cell(store, exp, system=s, method=method, scenario=sc, delta_pu=d, load_cap=cap)
                    if has
                    else ""
                )
            body.append(row)
    head = ["case", *(f"{m} trust, cap {c}" for m, c in combos)]
    return table(head, body)


_WU_TABLE2 = {
    ("ieee14", "0"): "25.6% / 23.9%",
    ("ieee14", "1"): "35.2% / 27.0%",
    ("ieee118", "0"): "16.4% mean, mostly 10% to 20%",
}


@query("wu26.reproduction")
def wu26_reproduction(store: Store) -> str:
    """Per case, rating and method: windows run, windows whose attack survives the trust schedule, the
    median devices and channels undefended and defended, the median cost rise (channels, eq. 33) and
    extra devices over the surviving windows, beside the paper's figure."""
    import statistics

    exp = "wu26.reproduction"
    rows = store.latest(exp)
    groups = sorted({(r.system, r.tag("scenario"), r.tag("k"), r.method) for r in rows})
    body = []
    for system, scenario, k, method in groups:
        keys = dict(system=system, scenario=scenario, k=k, method=method)

        def values(metric: str, **more: str) -> list[float]:
            return [r.value for r in store.latest(exp, metric=metric, **keys, **more)]

        before, after = values("devices", stage="undefended"), values("devices", stage="defended")
        ch_before, ch_after = values("channels", stage="undefended"), values("channels", stage="defended")
        rise, extra = values("cost_increase_pct"), values("extra_devices")
        attacked = [b for b in before if b > 0]
        med = lambda v, spec: number(statistics.median(v), spec) if v else ""  # noqa: E731
        body.append(
            [
                f"IEEE-{_short(system)}, scenario {int(scenario) + 1}",
                k,
                method,
                f"{len(before)}",
                f"{len(attacked)} / {len(rise)}",
                f"{med([b for b in before if b > 0], '.0f')} / {med([c for c in ch_before if c > 0], '.0f')}",
                (
                    f"{med([a for a in after if a > 0], '.0f')} / {med([c for c in ch_after if c > 0], '.0f')}"
                    if any(a > 0 for a in after)
                    else "none survives"
                ),
                f"{med(rise, '.1f')}%" if rise else "",
                med(extra, ".0f"),
                _WU_TABLE2.get((system, scenario), ""),
            ]
        )
    head = [
        "case",
        "rating",
        "method",
        "windows",
        "attacked / survives",
        "undefended (devices / channels)",
        "defended",
        "rise (channels)",
        "extra devices",
        "[WU26]",
    ]
    return table(head, body, numeric_from=3)


# ---- replication against [WU26] (docs/wu26/README.md, "Replication against [WU26]")
def _vals(store: Store, exp: str, metric: str, **keys: str) -> list[float]:
    return [r.value for r in store.latest(exp, metric=metric, **keys)]


def _range(values: list[float], spec: str, unit: str = "") -> str:
    """ "a" or "a to b" over the values, "" when there are none."""
    if not values:
        return ""
    lo, hi = number(min(values), spec), number(max(values), spec)
    return f"{lo}{unit}" if lo == hi else f"{lo}{unit} to {hi}{unit}"


def _pair(store: Store, exp: str, method: str, before: dict, after: dict, **keys: str) -> str:
    """ "d0 / c0 → d1 / c1" (devices / channels) of one case, "no attack" for a missing or failed side."""
    out = []
    for side in (before, after):
        d = _vals(store, exp, "devices", method=method, **keys, **side)
        c = _vals(store, exp, "channels", method=method, **keys, **side)
        ok = bool(d) and bool(c) and d[0] >= 0
        out.append(f"{number(d[0], '.0f')} / {number(c[0], '.0f')}" if ok else "no attack")
    return " → ".join(out)


def _case(tags: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(t for t in tags if not t.startswith(("defense", "case"))))


def _found(store: Store, exp: str, method: str, **keys: str) -> str:
    """ "a of b": cases with an attack (devices >= 0, not marked infeasible) among the cases run."""
    recs = store.latest(exp, method=method, **keys)
    cases = {_case(r.tags) for r in recs}
    hit = {_case(r.tags) for r in recs if r.metric == "devices" and r.value >= 0}
    hit -= {_case(r.tags) for r in recs if r.metric == "feasible" and r.value == 0}
    return f"{len(hit)} of {len(cases)}"


def _best(store: Store, exp: str, metric: str, **keys: str) -> str:
    v = _vals(store, exp, metric, **keys)
    return number(max(v), ".2f") if v else ""


def _hypotheses(store: Store) -> list[list[str]]:
    rows: list[list[str]] = []

    def add(what: str, outcome: str, evidence: str, exp: str) -> None:
        rows.append([str(len(rows) + 1), what, outcome, evidence, exp])

    add(
        "Δx_t is a per-snapshot increment (a staged reading of trust)",
        "rejected",
        "eq. (26) defines Δx_t as the state deviation",
        "",
    )
    add(
        "A trusted PMU also pins its neighbours through its branch currents (p. 655)",
        "rejected, contradicts Table II",
        "trusted attacks found: "
        + _found(store, "minlp.cap_trust_grid", "neighbour", defense="trust")
        + " (neighbour reading), "
        + _found(store, "minlp.cap_trust_grid", "own", defense="trust")
        + " (own V and θ)",
        "minlp.cap_trust_grid",
    )
    add(
        "The cost increase counts devices (Table II lists devices)",
        "rejected",
        "eq. (33) counts measurements; Table II's percentages do not follow from its device counts",
        "",
    )
    cold = store.latest("minlp.prototype", method="minlp_scip_devices", cold="1", metric="devices")
    add(
        "Solve eq. (12) exactly as a mixed-integer program (SCIP)",
        "not viable at these sizes",
        f"cold-start attacks found: {sum(1 for r in cold if r.value >= 0)} of {len(cold)} runs",
        "minlp.prototype",
    )
    region = dict(system="ieee14", scenario="0", defended="0")
    add(
        'Every meter in the area counts ("all measurement data ... altered simultaneously", p. 654)',
        "rejected, Fig. 4 leaves sub-noise devices out (p. 659)",
        "IEEE-14 lines 3-4 and 6-11, undefended devices: threshold "
        + _range(_vals(store, "minlp.region_reading", "devices", method="threshold", **region), ".0f")
        + ", region "
        + _range(_vals(store, "minlp.region_reading", "devices", method="region", **region), ".0f"),
        "minlp.region_reading",
    )
    add(
        "The attack is anchored on both ends of each target line (Fig. 4)",
        "no effect on the rise",
        "IEEE-118: "
        + _pair(
            store,
            "minlp.anchoring",
            "both",
            {"defense": "undefended"},
            {"defense": "trust per_slot-False"},
            system="ieee118",
        ),
        "minlp.anchoring",
    )
    rise = _vals(store, "wu26.reproduction", "cost_increase_pct", system="ieee118", k="+0.10")
    add(
        "Ratings at the true flow plus 0.10 pu, so attacks reach the Fig. 4 scale (ours)",
        "magnitudes match; the IEEE-118 rise stays below Fig. 12",
        "IEEE-118 channel rise over the windows: " + _range(rise, ".1f", "%"),
        "wu26.reproduction",
    )
    add(
        "Our load cap (D16) or the reading of trust separates us from [WU26]",
        "no single setting reproduces all three cases",
        "the cap and trust grid above",
        "minlp.cap_trust_grid",
    )
    add(
        "[WU26]'s data recipe (p. 658) in place of our operating pool",
        "same loading levels, gap not closed",
        "IEEE-14 lines 3-4 and 6-11, trusted attack found: "
        + _found(store, "wu.data_recipe", "wu", system="ieee14", scenario="0", case="trust"),
        "wu.data_recipe",
    )
    add(
        "[WU26]'s IEEE-118 attack area (Fig. 9)",
        "adopted (the table existed but was unused); no effect on the gap",
        "",
        "",
    )
    add(
        "The row-reduction attacker (p. 655, ref. [25])",
        "matches Table II worse than the search",
        "IEEE-118: "
        + _pair(store, "wu26.method_compare", "rref", {"trust": "none"}, {"trust": "wu"}, system="ieee118"),
        "wu26.method_compare",
    )
    l1_14 = dict(system="ieee14", scenario="1", rating="+0.10", source="pool", weights="sigma")
    add(
        "ℓ1 relaxation, then a count beyond noise ([37] p. 1898), solved with IPOPT",
        "reproduces IEEE-14 lines 1-2 and 4-5, not the others",
        "IEEE-14 lines 1-2 and 4-5: "
        + _pair(store, "wu.l1_attack", "ipopt", {"trust": "0"}, {"trust": "1"}, **l1_14)
        + "; IEEE-118: "
        + _pair(
            store, "wu.l1_attack", "ipopt", {"trust": "0"}, {"trust": "1"}, system="ieee118", rating="+0.10"
        ),
        "wu.l1_attack",
    )
    t5 = dict(system="ieee118", phase="undefended")
    add(
        "Table V factorial: solver, goal form, load curve, PMU measurement vector (eqs. 1 to 3)",
        "Table V's set not reached",
        "best overlap with Table V: SCADA "
        + _best(store, "wu.table5_search", "jaccard_scada", **t5)
        + ", PMU "
        + _best(store, "wu.table5_search", "jaccard_pmu", **t5),
        "wu.table5_search",
    )
    add(
        "The attacker builds on its own estimate from forecast loads (pp. 654, 659; forecast error ours)",
        "set and residual unchanged",
        "IEEE-118 detection "
        + _range(_vals(store, "wu.attacker_estimate", "bdd_detection_pct", system="ieee118"), ".1f", "%")
        + ", false alarms "
        + _range(_vals(store, "wu.attacker_estimate", "bdd_false_alarm_pct", system="ieee118"), ".1f", "%"),
        "wu.attacker_estimate",
    )
    add(
        "SCADA held between 5-min scans, PMUs every snapshot (p. 658; [29] Sec. 2.2)",
        "benign false alarms far below Table V's",
        "IEEE-118 benign false alarms "
        + _range(_vals(store, "wu.multirate", "bdd_false_alarm_pct", system="ieee118"), ".1f", "%"),
        "wu.multirate",
    )
    add(
        "Every snapshot an independent sample of the p. 658 recipe, SCADA held (ours)",
        "false alarms reach Table V's range; the set and detection pattern do not",
        "IEEE-118 benign false alarms "
        + _range(
            _vals(store, "wu.independent_samples", "bdd_false_alarm_pct", scada="stale", se_mode="every"),
            ".1f",
            "%",
        )
        + " with held SCADA, "
        + _range(
            _vals(store, "wu.independent_samples", "bdd_false_alarm_pct", scada="fresh", se_mode="every"),
            ".1f",
            "%",
        )
        + " with fresh SCADA",
        "wu.independent_samples",
    )
    opf = dict(system="ieee118", rating="+0.10", method="search")
    add(
        'The attacker\'s base state is a local optimal power flow ("local optimal power flow", p. 655)',
        "closest PMU overlap, but a goal set on the estimate is not a true overload",
        "IEEE-118: "
        + _range(_vals(store, "wu.local_opf", "devices", **opf), ".0f")
        + " devices, overlap SCADA "
        + _range(_vals(store, "wu.local_opf", "jaccard_scada", **opf), ".2f")
        + ", PMU "
        + _range(_vals(store, "wu.local_opf", "jaccard_pmu", **opf), ".2f"),
        "wu.local_opf",
    )
    return rows


@query("wu.hypotheses")
def wu_hypotheses(store: Store) -> str:
    """Every replication hypothesis against [WU26], its outcome and the evidence from its experiment."""
    head = ["#", "hypothesis", "outcome", "evidence (ours)", "experiment"]
    return table(head, _hypotheses(store), numeric_from=5)


@query("wu.table5")
def wu_table5(store: Store) -> str:
    """IEEE-118 undefended attacks of the Table V factorial: the set of PMUs each attack tampers, and the
    overlap of its devices with [WU26] Table V's multi-snapshot attack."""
    exp = "wu.table5_search"
    recs = store.latest(exp, system="ieee118", phase="undefended", metric="devices")
    body = []
    for r in sorted(
        recs, key=lambda r: (r.tag("pmu"), r.tag("rating"), r.method, r.tag("goal"), r.tag("load"))
    ):
        keys = {t.split("=", 1)[0]: t.split("=", 1)[1] for t in r.tags}
        js = _get(store, exp, system="ieee118", method=r.method, metric="jaccard_scada", **keys)
        jp = _get(store, exp, system="ieee118", method=r.method, metric="jaccard_pmu", **keys)
        body.append(
            [
                r.method,
                r.tag("goal"),
                r.tag("load"),
                r.tag("pmu"),
                r.tag("rating"),
                cell(r, "d") if r.value >= 0 else "no solve",
                r.tag("pmu_set"),
                cell(js, ".2f"),
                cell(jp, ".2f"),
            ]
        )
    head = [
        "solver",
        "goal",
        "load",
        "PMU vector",
        "rating",
        "devices",
        "PMUs",
        "overlap SCADA",
        "overlap PMU",
    ]
    return table(head, body, numeric_from=5)


# ---- the faithful reproduction of [WU26] (docs/wu26/README.md, "Faithful reproduction")
_WU_FAITHFUL = {
    "118": "Fig. 12: 16.4% mean, mostly 10% to 20%; mostly 3 to 5 new devices; Table V: 15 devices",
    "14-1": "Table II: 25.6% / 23.9%; new SCADA 1, 9, 13; Fig. 4: 7 devices, 0.22 pu",
    "14-2": "Table II: 35.2% / 27.0%; new SCADA 7, 11-14; Fig. 4: 9 devices, 0.17 pu",
}


@query("wu26.faithful")
def wu26_faithful(store: Store) -> str:
    """Per scenario and trust order of the faithful reproduction (`tools/wu26_faithful.py`): the
    windows, the order (Solution 1's, derived, or the paper's), the median devices undefended and
    defended, the mean and median rise in the devices tampered at any snapshot (Fig. 12's cost), the
    share of windows in Fig. 12's 10% to 20% band, the mean new devices, the mean overlap of the
    undefended attack with the paper's tampered set (Table V or Fig. 4) and the median largest change,
    beside the paper's figures."""
    import statistics

    exp = "wu26.faithful"
    body = []
    for name in sorted({r.tag("scenario") for r in store.query(exp)}):
        run = store.newest_run(exp, scenario=name)  # one run whole: a rerun never mixes with older windows

        def vals(metric: str, **more: str) -> list[float]:
            return [r.value for r in store.query(exp, metric=metric, scenario=name, run_id=run, **more)]

        steps = store.query(exp, metric="trusted_pmu", scenario=name, run_id=run)
        derived = " ".join(f"{int(r.value)}" for r in sorted(steps, key=lambda r: int(r.tag("step"))))
        for method, order in (("solution1", derived), ("described", "the paper's")):
            rise = vals("cost_increase_pct", method=method)
            if not rise:
                continue
            band = sum(10.0 <= v <= 20.0 for v in rise) / len(rise)
            overlap = f"{number(statistics.mean(vals('jaccard_scada')), '.2f')} / {number(statistics.mean(vals('jaccard_pmu')), '.2f')}"
            body.append(
                [
                    name,
                    order,
                    f"{len(rise)}",
                    number(statistics.median(vals("devices", defended="0")), ".0f"),
                    number(statistics.median(vals("devices", defended="1", method=method)), ".0f"),
                    f"{number(statistics.mean(rise), '.1f')}% / {number(statistics.median(rise), '.1f')}%",
                    f"{number(100 * band, '.0f')}%",
                    number(statistics.mean(vals("extra_devices", method=method)), ".1f"),
                    overlap,
                    number(statistics.median(vals("max_change_pu")), ".2f"),
                    _WU_FAITHFUL.get(name, ""),
                ]
            )
    head = [
        "scenario",
        "trust order",
        "windows",
        "devices undefended",
        "defended",
        "rise mean / median",
        "windows in 10-20%",
        "new devices",
        "overlap SCADA / PMU",
        "largest change (pu)",
        "[WU26]",
    ]
    return table(head, body, numeric_from=2)
