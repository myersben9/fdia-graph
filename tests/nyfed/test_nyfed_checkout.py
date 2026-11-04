"""The hourly checkout knows of a gap only once its hour ends, and only when the hour's mean gap passes the
threshold the scan-rate check uses."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "nyfed"))
from hourly_checkout import hourly_alarm  # noqa: E402

CALIB = (28.0, 14.0)  # benign gap mean and standard deviation (MW): the threshold is 56 MW off the mean
TIME = np.datetime64("2019-06-17T00:00") + np.arange(36) * np.timedelta64(5, "m")  # three clock hours


def test_a_gap_is_known_at_the_end_of_the_hour_whose_mean_passes_the_threshold():
    gap = np.full(36, 28.0)
    gap[14:] = 28.0 + 300.0  # the attack's gap opens at scan 14, in the second hour
    assert hourly_alarm(TIME, gap, onset=14, calib=CALIB) == 23  # the second hour's last scan


def test_a_short_spike_inside_an_hour_is_averaged_away():
    gap = np.full(36, 28.0)
    gap[13] = 28.0 + 300.0  # one scan off: the hour's mean moves by 25 MW, inside the threshold
    assert hourly_alarm(TIME, gap, onset=12, calib=CALIB) is None
