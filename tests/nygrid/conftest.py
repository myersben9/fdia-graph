"""The New York tests' shared setup: the operator's dispatch reads the bus reactive loads of the checked-in days."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from .inputs import DAYS


@pytest.fixture(scope="session", autouse=True)
def _days_loads() -> Iterator[None]:
    """`pool_frames` takes the loads file from `NYGRID_LOADS`; the tests run on the days' slice of it. Set for the
    session, so a module's fixture that runs the operator finds it too."""
    patch = pytest.MonkeyPatch()
    patch.setenv("NYGRID_LOADS", str(DAYS / "loads_2019_v2_days.npz"))
    yield
    patch.undo()
