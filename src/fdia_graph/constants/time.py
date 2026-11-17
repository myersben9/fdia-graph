"""Time: the calendar, the New York frame, and how a load profile is correlated from scan to scan."""

from __future__ import annotations

import numpy as np

WEEK_SECONDS = 7 * 86400  # seconds in a week: a New York timeline's splits are cut in whole weeks
HOURS_PER_YEAR = 8760  # hours in a year of 365 days: the rows of an hourly load profile
QUARTERS_PER_YEAR = 4 * HOURS_PER_YEAR  # the native 15-minute rows of the building stock profiles [LI21]
FRAME_MINUTES = 5.0  # minutes: one New York frame, NYISO's real-time load interval
HALF_HOUR = np.timedelta64(30, "m")  # the shift from the top of an hour to the centre of that hour
# Per scan: the AR(1) coefficient of the per-bus load jitter of an IEEE load profile (`data/ieee_pool.py`). At
# 0.98 a load evolves smoothly from minute to minute (about 0.5% per minute, against about 4% for white
# noise) at the same stationary bus-to-bus spread. Every released pool was drawn with it.
LOAD_JITTER_RHO = 0.98
