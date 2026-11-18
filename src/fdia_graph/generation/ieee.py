"""An IEEE case by its name as the source of a timeline: the original generator
(`generation.search.SearchSource`) on the case's network and its pool of operating points."""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np

from ..models.config import TimelineSettings
from ..registry import CACHE_DIR, system_id
from ..systems.network import Network
from . import _load_states
from .search import SearchSource
from .source import OutPath


class IeeeSource(SearchSource):
    """The IEEE case `system` ("ieee118" and 118 alike) with its pool of AC operating points
    (`states`: an array, a path, or None for the downloaded pool), the `settings` of the walk, the
    `seed` of the generator and, when the caller has it, each frame's `time` [T]."""

    named = "an IeeeSource"

    def __init__(
        self,
        system: Union[int, str],
        states: Optional[Union[str, np.ndarray]],
        settings: TimelineSettings,
        seed: int = 123,
        time: Optional[np.ndarray] = None,
    ) -> None:
        self.system = system_id(system)
        super().__init__(Network.from_ieee(self.system), _load_states(system, states), settings, seed, time)

    def out_path(self, out: Optional[OutPath]) -> str:
        """`out`, or `timeline_ieee{N}.h5` under the cache directory."""
        return os.fspath(out) if out else os.path.join(CACHE_DIR, f"timeline_ieee{self.system}.h5")
