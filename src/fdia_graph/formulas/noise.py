"""The meter error model: the standard deviation an accuracy class gives a reading [ASP14], and the
noise [WU26]'s case studies state. `engine.sensing.noise.NoiseModel` splits a class into a constant
per-meter bias and a per-scan jitter (ours) and draws both."""

from __future__ import annotations

from typing import Union

import numpy as np

from ..constants.channels import CurrentChannel, NodeChannel
from ..constants.meters import CURRENT_FLOOR_PU, WU26_NOISE


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


def current_magnitude(current_true: np.ndarray) -> np.ndarray:
    """Each PMU branch-current channel's end-phasor magnitude [..., E, 4]: |I_from| on the two from
    columns, |I_to| on the two to columns, the scale of the C37.118 error model [C37118]."""
    i = np.asarray(current_true, np.float64)
    from_mag = np.hypot(i[..., CurrentChannel.RE_FROM], i[..., CurrentChannel.IM_FROM])
    to_mag = np.hypot(i[..., CurrentChannel.RE_TO], i[..., CurrentChannel.IM_TO])
    return np.stack([from_mag, from_mag, to_mag, to_mag], axis=-1)


def current_sigma(current_true: np.ndarray, rel: float, floor: float = CURRENT_FLOOR_PU) -> np.ndarray:
    """The std [..., E, 4] of every PMU branch-current channel [C37118], per unit on the base current:
    sigma_re = sigma_im = rel |I_end| + floor, `rel` the class (`PMU_CURRENT_CLASS`). The rule of
    `engine.sensing.noise.NoiseModel` on a file's current layer."""
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
