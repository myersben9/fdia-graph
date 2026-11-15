"""The New York tests' shared setup: the operator's dispatch reads the bus reactive loads of the checked-in days."""

from __future__ import annotations

import pytest

from .inputs import DAYS


@pytest.fixture(autouse=True)
def _days_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """`pool_frames` takes the loads file from `NYGRID_LOADS`; the tests run on the days' slice of it."""
    monkeypatch.setenv("NYGRID_LOADS", str(DAYS / "loads_2019_v2_days.npz"))
