"""The meter error model: accuracy-class standard deviations split into a constant per-meter
bias and a per-scan jitter [ASP14], our split."""

from __future__ import annotations

import numpy as np

from ..models.grid import EDGE, NODE


def bias_jitter_split(
    sd: dict[str, float], jitter_frac: float = 0.25
) -> tuple[dict[str, float], dict[str, float]]:
    """Split each accuracy-class std into a per-scan jitter and a per-meter bias [ASP14], our split.

        jitter = jitter_frac * SD,   bias = sqrt(1 - jitter_frac^2) * SD,   so bias^2 + jitter^2 = SD^2

    The accuracy-class error is mostly systematic (a constant calibration offset drawn once per
    meter) with a small per-scan random part; treating all of SD as per-scan noise would
    over-jitter one-minute traces and drown the temporal channel. The root-sum-square keeps the
    class total.

    sd      : std per channel kind, e.g. {"pf": 0.017, "v": 0.0012, ...}
    returns : (jitter std per kind, bias std per kind)
    """
    jitter = {k: v * jitter_frac for k, v in sd.items()}
    bias = {k: v * (1.0 - jitter_frac * jitter_frac) ** 0.5 for k, v in sd.items()}
    return jitter, bias


def jitter_sigma(
    node_true: np.ndarray, flow_true: np.ndarray, jitter: dict[str, float], floor_mw: float
) -> tuple[np.ndarray, np.ndarray]:
    """The per-scan noise standard deviation of every channel, the one the emitter draws with
    (`MeasurementMixin.emit_from_state`) and the one the fewest-tamper objective counts against.

        |V|      : jitter["v"]                              (pu, absolute)
        angle    : degrees(jitter["va"])                    (deg, absolute)
        P, Q inj : |P_true| jitter["pi"] + floor_mw, |Q_true| jitter["qi"] + floor_mw   (MW, MVAr)
        P, Q flow: |P_true| jitter["pf"] + floor_mw, |Q_true| jitter["qf"] + floor_mw   (MW, MVAr)

    A new channel kind (the PMU branch-current phasors of [WU26, eqs. 19-20]) gets its rule here,
    beside the others, so the emitter and the objective cannot disagree on it.

    node_true : [N, 4] true |V|, P_inj, Q_inj, theta of the scan
    flow_true : [E, 2] true P_from, Q_from of the scan (MW, MVAr)
    jitter    : per-scan std per channel kind (`bias_jitter_split`'s first return)
    floor_mw  : the absolute floor on power channels
    returns   : (sigma per node channel [N, 4], sigma per flow channel [E, 2])
    """
    node_true = np.asarray(node_true, np.float64)
    flow_true = np.asarray(flow_true, np.float64)
    sig_node = np.empty(node_true.shape, np.float64)
    sig_node[:, NODE.v] = jitter["v"]
    sig_node[:, NODE.p_inj] = np.abs(node_true[:, NODE.p_inj]) * jitter["pi"] + floor_mw
    sig_node[:, NODE.q_inj] = np.abs(node_true[:, NODE.q_inj]) * jitter["qi"] + floor_mw
    sig_node[:, NODE.theta] = np.degrees(jitter["va"])
    sig_flow = np.empty(flow_true.shape, np.float64)
    sig_flow[:, EDGE.p_from] = np.abs(flow_true[:, EDGE.p_from]) * jitter["pf"] + floor_mw
    sig_flow[:, EDGE.q_from] = np.abs(flow_true[:, EDGE.q_from]) * jitter["qf"] + floor_mw
    return sig_node, sig_flow
