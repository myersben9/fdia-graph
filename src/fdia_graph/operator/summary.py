"""The attack's consequence over a day, attacked run less benign run (`summarize`): the dispatch moved, the
regulation and its energy by utility, the inadvertent interchange, the true interface changes and margins, the
scans over a true limit and those whose power flow failed, the costs, and any load shed."""

from __future__ import annotations

import numpy as np

from fdia_graph.operator.run import Response
from fdia_graph.operator.sced import FRAME_HOURS


def _over(r: Response) -> dict:
    """Where the attack takes an interface past its true limit (N72): the scans on which the attacked run is further
    over a true limit than the benign twin, the longest run of them, that excess energy (MWh) and the interface most
    often over on them (-1 none). The least-cost dispatch holds a binding interface at its shift-factor limit, so
    the true flow can pass the limit in the benign twin too, and only the attack's excess counts. A scan whose power
    flow failed in either run has no margin and is not counted over; `pf_failed_scans` says how many such scans
    there are."""
    excess = np.clip(-np.nan_to_num(r.attacked.margin, nan=1.0), 0, None) - np.clip(
        -np.nan_to_num(r.benign.margin, nan=1.0), 0, None
    )
    over = excess > 0
    run = best = 0
    for o in over:
        run = run + 1 if o else 0
        best = max(best, run)
    at = r.attacked.margin_at[over]
    return {
        "over_scans": int(over.sum()),
        "over_run_scans": int(best),
        "over_mwh": float(excess[over].sum() * FRAME_HOURS),
        "over_iface": int(np.bincount(at).argmax()) if len(at) else -1,
    }


def summarize(r: Response, voll: float | None = None) -> dict:
    """The attack's consequence over the day, attacked less benign scan by scan, so what the benign twin
    already shows (a lagged reading over its limit, load the units cannot follow) is not counted: the
    most dispatch moved and regulation called (MW), the regulation energy (MWh, upward, in total and by the
    utility whose territory holds each unit), the most interface flow left unrelieved, load unmatched and
    imbalance uncovered (MW), the largest true interface drop and rise and N-1 rise (MW), the tightest true
    margin of the attacked and of the benign run (MW, negative over a limit) and the most margin the attack
    takes, and the production cost and LBMP settlement changes ($). Then the escalation (N59): the most load shed (MW),
    the energy shed (MWh, in total and by utility), what relieving the interfaces cost in production
    ($, the fake congestion's cost), the shortage cost of the flow left over the limit ($), and with
    `voll` ($/MWh) the shed energy at that value."""
    a, b = r.attacked, r.benign

    def peak(x: np.ndarray) -> float:
        return float(np.nanmax(x)) if np.isfinite(x).any() else 0.0

    reg_up = (np.clip(a.reg, 0, None) - np.clip(b.reg, 0, None)) * FRAME_HOURS  # [T, G] MWh, attack only
    out = {
        "delay": r.delay,
        "over_mw": r.over,
        "dispatch_moved_mw": peak(np.abs(a.dev).sum(axis=1) - np.abs(b.dev).sum(axis=1)),
        "regulation_mw": peak(np.abs(a.reg).sum(axis=1) - np.abs(b.reg).sum(axis=1)),
        "regulation_mwh": float(reg_up.sum()),
        "unrelieved_mw": peak(np.nan_to_num(a.unrelieved) - np.nan_to_num(b.unrelieved)),
        "unmatched_mw": peak(np.abs(np.nan_to_num(a.unmatched)) - np.abs(np.nan_to_num(b.unmatched))),
        "uncovered_mw": peak(np.abs(a.uncovered) - np.abs(b.uncovered)),
        "iface_down_mw": float(np.nanmin(a.iface_change[:, 0] - b.iface_change[:, 0])),
        "iface_up_mw": float(np.nanmax(a.iface_change[:, 0] - b.iface_change[:, 0])),
        "other_iface_up_mw": peak(a.iface_change[:, 1:] - b.iface_change[:, 1:])
        if a.iface_change.shape[1] > 1
        else 0.0,
        "inadvertent_mw": peak(np.abs(a.uncovered) - np.abs(b.uncovered)),
        "inadvertent_mwh": float(
            np.clip(np.abs(a.uncovered) - np.abs(b.uncovered), 0, None).sum() * FRAME_HOURS
        ),
        "tie_gap_mw": peak(np.abs(r.tie_attacked - r.tie_benign)),
        "alarm_scans": r.alarm,
        "mode": r.mode,
        "n1_up_mw": float(np.nanmax(a.n1_change - b.n1_change)),
        "margin_min_mw": float(np.nanmin(a.margin)) if np.isfinite(a.margin).any() else float("nan"),
        "benign_margin_min_mw": float(np.nanmin(b.margin)) if np.isfinite(b.margin).any() else float("nan"),
        "margin_lost_mw": peak(b.margin - a.margin),
        **_over(r),
        "pf_failed_scans": int(a.pf_failed.sum()),
        "pf_failed_benign_scans": int(b.pf_failed.sum()),
        "fuel_usd": float(a.fuel.sum() - b.fuel.sum()),
        "settled_usd": float(a.settled.sum() - b.settled.sum()),
        "relief_usd": float(a.relief.sum() - b.relief.sum()),
        "shortage_usd": float(a.shortage.sum() - b.shortage.sum()),
        "shed_mw": peak(a.shed.sum(axis=1) - b.shed.sum(axis=1)),
        "shed_mwh": float((a.shed - b.shed).sum() * FRAME_HOURS),
    }
    if voll is not None:
        out["shed_usd"] = out["shed_mwh"] * voll
    for c, name in enumerate(r.client_names):
        out[f"regulation_mwh_{name}"] = float(reg_up[:, r.unit_client == c].sum())
        out[f"shed_mwh_{name}"] = float((a.shed - b.shed)[:, r.bus_client == c].sum() * FRAME_HOURS)
    return out
