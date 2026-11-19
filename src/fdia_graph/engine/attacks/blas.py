"""One BLAS thread for the length of a search (`one_blas_thread`).

`Minimizer.min_tamper` (minimize.py) runs every search inside it. The limit is threadpoolctl's
(`ThreadpoolController.limit`); without that package nothing is changed."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from threadpoolctl import ThreadpoolController


_blas_pools: list[Optional[ThreadpoolController]] = []  # the process's BLAS pools, found on the first search
# BLAS limits are process-wide, so concurrent searches share one limit: the first to enter sets it,
# the last to leave restores the caller's setting
_blas_lock = threading.Lock()
_blas_held: list[contextlib.AbstractContextManager[object]] = []  # the active limit, while any search runs
_blas_users = [0]  # searches currently inside the limit


@contextlib.contextmanager
def one_blas_thread() -> Iterator[None]:
    """Hold the BLAS pools to one thread for a search, restored after the last concurrent search leaves:
    the search's products are small, so starting BLAS threads costs more than the arithmetic. Without
    threadpoolctl (the `generate` extra installs it) nothing is changed."""
    with _blas_lock:
        if not _blas_pools:
            try:
                from threadpoolctl import ThreadpoolController

                _blas_pools.append(ThreadpoolController())
            except ImportError:
                _blas_pools.append(None)
        controller = _blas_pools[0]
        if controller is not None and _blas_users[0] == 0:
            held = controller.limit(limits=1, user_api="blas")
            held.__enter__()
            _blas_held.append(held)
        _blas_users[0] += 1
    try:
        yield
    finally:
        with _blas_lock:
            _blas_users[0] -= 1
            if _blas_users[0] == 0 and _blas_held:
                _blas_held.pop().__exit__(None, None, None)
