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
    """Each meter's error std from its accuracy class [ASP14]: a relative class scales the meter's
    mean absolute reading, an absolute class is the std itself,

        sigma_i = c_i |z_i| + floor   (relative: power injections and flows)
        sigma_i = c_i                 (absolute: voltage magnitude and angle)

    Equipment data and measurements only; no state enters.

    mean_abs : [m] mean |z_i| over the calibration scans
    cls      : [m] accuracy class per meter
    relative : [m] bool, True where the class is relative
    floor    : absolute floor of a relative meter's std (its reading's units), one for all or [m] per meter
    returns  : [m], or the shape of `mean_abs` with the meters last
    """
    mean_abs, cls = np.asarray(mean_abs, np.float64), np.asarray(cls, np.float64)
    return np.where(np.asarray(relative, bool), cls * mean_abs + floor, cls)


def bias_jitter_split(
    sd: dict[str, float], jitter_frac: float = JITTER_FRAC
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
    """The per-scan jitter standard deviation of every channel, the one the emitter draws its noise
    with (`MeasurementMixin.emit_from_state`). Emission only: what counts as a change above a meter's
    noise, for detection and for the fewest-tamper search, is the meter's rated accuracy
    (`accuracy_sigma`).

        |V|      : jitter["v"]                              (pu, absolute)
        angle    : degrees(jitter["va"])                    (deg, absolute)
        P, Q inj : |P_true| jitter["pi"] + floor_mw, |Q_true| jitter["qi"] + floor_mw   (MW, MVAr)
        P, Q flow: |P_true| jitter["pf"] + floor_mw, |Q_true| jitter["qf"] + floor_mw   (MW, MVAr)

    A new channel kind (the PMU branch-current phasors of [WU26 eqs. 19-20]) gets its rule here and
    in `accuracy_sigma`, beside the others.

    node_true : [N, 4] true |V|, P_inj, Q_inj, theta of the scan
    flow_true : [E, 2] true P_from, Q_from of the scan (MW, MVAr)
    jitter    : per-scan std per channel kind (`bias_jitter_split`'s first return)
    floor_mw  : the absolute floor on power channels
    returns   : (sigma per node channel [N, 4], sigma per flow channel [E, 2])
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
    """The accuracy-class standard deviation of every channel of one scan [ASP14], in the scan's
    physical units: the meter's rated accuracy, the sigma the measured calibration of the estimators
    uses (`accuracy_class_sigma`, through `SEBase._class_sigma`), here at this scan's true readings.
    What a change must exceed to count as tampered, and the most a channel may move between two
    snapshots of a stealthy attack window [WU26].

        |V|      : cls["v"]                                   (pu, absolute)
        angle    : degrees(cls["va"])                         (deg, absolute)
        P, Q inj : cls["pi"] |P_true| + floor_mw, cls["qi"] |Q_true| + floor_mw    (MW, MVAr)
        P, Q flow: cls["pf"] |P_true| + floor_mw, cls["qf"] |Q_true| + floor_mw    (MW, MVAr)

    node_true : [N, 4] true |V|, P_inj, Q_inj, theta of the scan
    flow_true : [E, 2] true P_from, Q_from of the scan (MW, MVAr)
    cls       : accuracy class per channel kind (`ACCURACY_CLASS`)
    floor_mw  : the absolute floor on power channels
    returns   : (sigma per node channel [N, 4], sigma per flow channel [E, 2])
    """
    node_true = np.asarray(node_true, np.float64)
    flow_true = np.asarray(flow_true, np.float64)
    N, E = node_true.shape[0], flow_true.shape[0]

    sig_node = np.empty((N, 4), np.float64)
    sig_node[:, NodeChannel.V] = _class_rule(node_true[:, NodeChannel.V], cls["v"], False, floor_mw)
    sig_node[:, NodeChannel.P_INJ] = _class_rule(node_true[:, NodeChannel.P_INJ], cls["pi"], True, floor_mw)
    sig_node[:, NodeChannel.Q_INJ] = _class_rule(node_true[:, NodeChannel.Q_INJ], cls["qi"], True, floor_mw)
    sig_node[:, NodeChannel.THETA] = np.degrees(
        _class_rule(node_true[:, NodeChannel.THETA], cls["va"], False, floor_mw)
    )
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
    """The PMU branch-current readings with their systematic error before jitter [C37118]: each
    channel moved by its relative bias times its end-phasor magnitude, the same scale as
    `current_sigma`, so a channel whose component is zero still carries the class error.

        I_reading = I_true + b |I_end|            (per channel, before the per-scan jitter)
    """
    return np.asarray(current_true, np.float64) + np.asarray(bias, np.float64) * current_magnitude(
        current_true
    )


def current_sigma(current_true: np.ndarray, rel: float, floor: float = CURRENT_FLOOR_PU) -> np.ndarray:
    """The standard deviation of every PMU branch-current channel [C37118]: `rel` times the magnitude
    of that end's phasor, plus `floor`, on both its real and imaginary part. The one rule the
    emitter (with the per-scan jitter part of the class) and the detection side (with the whole
    class, `PMU_CURRENT_CLASS`) share.

        sigma_re = sigma_im = rel |I_end| + floor          (per unit on the base current)

    current_true : [..., E, 4] the true Re/Im of I_from and I_to (`CurrentChannel` columns)
    rel          : the relative std (the class, or its jitter part)
    floor        : the absolute floor, per unit
    returns      : [..., E, 4] sigma per channel
    """
    return rel * current_magnitude(current_true) + floor


def paper_current_sigma(shape: tuple[int, ...]) -> np.ndarray:
    """[WU26]'s case-study noise on the PMU branch-current channels (its D8 scale for the overload
    attack): 0.01 pu on every channel, per unit on the base current."""
    return np.full(shape, WU26_NOISE["pmu"], np.float64)


def paper_sigma(
    node_shape: tuple[int, ...], edge_shape: tuple[int, ...], pmu_bus: np.ndarray, base_mva: float
) -> tuple[np.ndarray, np.ndarray]:
    """The noise standard deviation of every channel as [WU26]'s case studies state it (0.03 pu for
    SCADA, 0.01 pu for PMU), in the stored units, the scale of the overload attack's
    stealth bound and of its tamper count, since the paper excludes from its l0 count the changes
    smaller than its own noise.

        |V|      : 0.01 pu at a PMU bus, 0.03 pu at a SCADA voltmeter
        angle    : degrees(0.01 rad) (angles are PMU channels)
        P, Q inj : 0.03 base_mva                              (MW, MVAr)
        P, Q flow: 0.03 base_mva                              (MW, MVAr)

    node_shape : (N, 4), the node channels
    edge_shape : (E, 2), the flow channels
    pmu_bus    : [N] booleans, the buses with a PMU
    base_mva   : the case's base power
    returns    : (sigma per node channel [N, 4], sigma per flow channel [E, 2])
    """
    scada, pmu = WU26_NOISE["scada"], WU26_NOISE["pmu"]
    sig_node = np.empty(node_shape, np.float64)
    sig_node[:, NodeChannel.V] = np.where(np.asarray(pmu_bus, bool), pmu, scada)
    sig_node[:, NodeChannel.P_INJ] = scada * base_mva
    sig_node[:, NodeChannel.Q_INJ] = scada * base_mva
    sig_node[:, NodeChannel.THETA] = np.degrees(pmu)
    sig_flow = np.full(edge_shape, scada * base_mva, np.float64)
    return sig_node, sig_flow
