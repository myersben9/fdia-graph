"""The meter error model: the standard deviation an accuracy class gives a reading [ASP14].
`sensing.noise.NoiseModel` splits a class into a constant per-meter bias and a per-scan jitter (ours)
and draws both."""

from __future__ import annotations

from typing import Union

import numpy as np

from ..constants.channels import CurrentChannel
from ..constants.meters import CURRENT_FLOOR_PU


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
    `sensing.noise.NoiseModel` on a file's current layer."""
    return rel * current_magnitude(current_true) + floor
