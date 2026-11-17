"""The meter error model: accuracy-class standard deviations split into a constant per-meter
bias and a per-scan jitter [ASP14], our split."""

from __future__ import annotations

from typing import Union

import numpy as np

from ..constants.channels import CurrentChannel, EdgeChannel, NodeChannel
from ..constants.meters import CURRENT_FLOOR_PU, JITTER_FRAC, WU26_NOISE


def accuracy_class_sigma(
    mean_abs: np.ndarray, cls: np.ndarray, relative: np.ndarray, floor: Union[float, np.ndarray]
) -> np.ndarray:
    """Each meter's error std from its accuracy class c_i and its mean absolute reading |z_i| [ASP14],
    in the reading's units. `floor` is one value or [m] per meter. Equipment data and measurements only:
    no state enters.

        sigma_i = c_i |z_i| + floor   (relative: power injections and flows)
        sigma_i = c_i                 (absolute: voltage magnitude and angle)
    """
    mean_abs, cls = np.asarray(mean_abs, np.float64), np.asarray(cls, np.float64)
    return np.where(np.asarray(relative, bool), cls * mean_abs + floor, cls)


def bias_jitter_split(
    sd: dict[str, float], jitter_frac: float = JITTER_FRAC
) -> tuple[dict[str, float], dict[str, float]]:
    """(jitter std, bias std) per channel kind from each accuracy-class std SD [ASP14], our split:
    jitter = jitter_frac * SD, bias = sqrt(1 - jitter_frac^2) * SD, so bias^2 + jitter^2 = SD^2.
    The class error is mostly a constant per-meter offset: treating all of SD as per-scan noise would
    over-jitter one-minute traces and drown the temporal channel."""
    jitter = {k: v * jitter_frac for k, v in sd.items()}
    bias = {k: v * (1.0 - jitter_frac * jitter_frac) ** 0.5 for k, v in sd.items()}
    return jitter, bias


def jitter_sigma(
    node_true: np.ndarray, flow_true: np.ndarray, jitter: dict[str, float], floor_mw: float
) -> tuple[np.ndarray, np.ndarray]:
    """The per-scan jitter std of every channel, (node [N, 4], flow [E, 2]), which the emitter draws
    its noise with. Emission only: a change is compared with the rated accuracy (`accuracy_sigma`).

        |V|      : jitter["v"]                              (pu, absolute)
        angle    : degrees(jitter["va"])                    (deg, absolute)
        P, Q inj : |P_true| jitter["pi"] + floor_mw, |Q_true| jitter["qi"] + floor_mw   (MW, MVAr)
        P, Q flow: |P_true| jitter["pf"] + floor_mw, |Q_true| jitter["qf"] + floor_mw   (MW, MVAr)
    """
    node_true = np.asarray(node_true, np.float64)
    flow_true = np.asarray(flow_true, np.float64)
    sig_node = np.empty(node_true.shape, np.float64)
    sig_node[:, NodeChannel.V] = jitter["v"]
    sig_node[:, NodeChannel.P_INJ] = np.abs(node_true[:, NodeChannel.P_INJ]) * jitter["pi"] + floor_mw
    sig_node[:, NodeChannel.Q_INJ] = np.abs(node_true[:, NodeChannel.Q_INJ]) * jitter["qi"] + floor_mw
    sig_node[:, NodeChannel.THETA] = np.degrees(jitter["va"])
    sig_flow = np.empty(flow_true.shape, np.float64)
    sig_flow[:, EdgeChannel.P_FROM] = np.abs(flow_true[:, EdgeChannel.P_FROM]) * jitter["pf"] + floor_mw
    sig_flow[:, EdgeChannel.Q_FROM] = np.abs(flow_true[:, EdgeChannel.Q_FROM]) * jitter["qf"] + floor_mw
    return sig_node, sig_flow


def accuracy_sigma(
    node_true: np.ndarray, flow_true: np.ndarray, cls: dict[str, float], floor_mw: float
) -> tuple[np.ndarray, np.ndarray]:
    """The rated-accuracy std of every channel of one scan [ASP14], (node [N, 4], flow [E, 2]), at the
    scan's true readings. A change must exceed it to count as tampered, and it is the most a channel
    may move between two snapshots of a stealthy attack window [WU26].

        |V|      : cls["v"]                                   (pu, absolute)
        angle    : degrees(cls["va"])                         (deg, absolute)
        P, Q inj : cls["pi"] |P_true| + floor_mw, cls["qi"] |Q_true| + floor_mw    (MW, MVAr)
        P, Q flow: cls["pf"] |P_true| + floor_mw, cls["qf"] |Q_true| + floor_mw    (MW, MVAr)
    """
    node_true = np.asarray(node_true, np.float64)
    flow_true = np.asarray(flow_true, np.float64)
    N, E = node_true.shape[0], flow_true.shape[0]

    sig_node = np.empty((N, 4), np.float64)
    sig_node[:, NodeChannel.V] = _class_rule(node_true[:, NodeChannel.V], cls["v"], False, floor_mw)
    sig_node[:, NodeChannel.P_INJ] = _class_rule(node_true[:, NodeChannel.P_INJ], cls["pi"], True, floor_mw)
    sig_node[:, NodeChannel.Q_INJ] = _class_rule(node_true[:, NodeChannel.Q_INJ], cls["qi"], True, floor_mw)
    theta_rad = _class_rule(node_true[:, NodeChannel.THETA], cls["va"], False, floor_mw)
    sig_node[:, NodeChannel.THETA] = np.degrees(theta_rad)
    sig_flow = np.empty((E, 2), np.float64)
    sig_flow[:, EdgeChannel.P_FROM] = _class_rule(flow_true[:, EdgeChannel.P_FROM], cls["pf"], True, floor_mw)
    sig_flow[:, EdgeChannel.Q_FROM] = _class_rule(flow_true[:, EdgeChannel.Q_FROM], cls["qf"], True, floor_mw)
    return sig_node, sig_flow


def _class_rule(reading: np.ndarray, c: float, relative: bool, floor: float) -> np.ndarray:
    """One channel kind's accuracy-class sigma at its readings (`accuracy_class_sigma`)."""
    n = len(reading)
    return accuracy_class_sigma(np.abs(reading), np.full(n, c), np.full(n, relative), floor)


def current_magnitude(current_true: np.ndarray) -> np.ndarray:
    """Each PMU branch-current channel's end-phasor magnitude [..., E, 4]: |I_from| on the two from
    columns, |I_to| on the two to columns, the scale of the C37.118 error model [C37118]."""
    i = np.asarray(current_true, np.float64)
    from_mag = np.hypot(i[..., CurrentChannel.RE_FROM], i[..., CurrentChannel.IM_FROM])
    to_mag = np.hypot(i[..., CurrentChannel.RE_TO], i[..., CurrentChannel.IM_TO])
    return np.stack([from_mag, from_mag, to_mag, to_mag], axis=-1)


def biased_current(current_true: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """The PMU branch-current readings with their systematic error, before jitter [C37118]:
    I_reading = I_true + b |I_end| per channel. The scale is the end-phasor magnitude, as in
    `current_sigma`, so a channel whose component is zero still carries the class error."""
    scale = current_magnitude(current_true)
    return np.asarray(current_true, np.float64) + np.asarray(bias, np.float64) * scale


def current_sigma(current_true: np.ndarray, rel: float, floor: float = CURRENT_FLOOR_PU) -> np.ndarray:
    """The std [..., E, 4] of every PMU branch-current channel [C37118], per unit on the base current:
    sigma_re = sigma_im = rel |I_end| + floor. The emitter passes the jitter part of the class as
    `rel`, the detection side the whole class (`PMU_CURRENT_CLASS`)."""
    return rel * current_magnitude(current_true) + floor


def paper_current_sigma(shape: tuple[int, ...]) -> np.ndarray:
    """[WU26]'s case-study noise on the PMU branch-current channels, the scale of the overload
    attack: 0.01 pu on every channel, per unit on the base current."""
    return np.full(shape, WU26_NOISE["pmu"], np.float64)


def paper_sigma(
    node_shape: tuple[int, ...], edge_shape: tuple[int, ...], pmu_bus: np.ndarray, base_mva: float
) -> tuple[np.ndarray, np.ndarray]:
    """The noise std of every channel as [WU26]'s case studies state it, (node [N, 4], flow [E, 2]) in
    the stored units: the scale of the overload attack's stealth bound and tamper count.

        |V|      : 0.01 pu at a PMU bus, 0.03 pu at a SCADA voltmeter
        angle    : degrees(0.01 rad) (angles are PMU channels)
        P, Q inj : 0.03 base_mva                              (MW, MVAr)
        P, Q flow: 0.03 base_mva                              (MW, MVAr)
    """
    scada, pmu = WU26_NOISE["scada"], WU26_NOISE["pmu"]
    sig_node = np.empty(node_shape, np.float64)
    sig_node[:, NodeChannel.V] = np.where(np.asarray(pmu_bus, bool), pmu, scada)
    sig_node[:, NodeChannel.P_INJ] = scada * base_mva
    sig_node[:, NodeChannel.Q_INJ] = scada * base_mva
    sig_node[:, NodeChannel.THETA] = np.degrees(pmu)
    sig_flow = np.full(edge_shape, scada * base_mva, np.float64)
    return sig_node, sig_flow
