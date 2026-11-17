"""One continuous attacked timeline per system, written as one HDF5 file.

The timeline is the dataset. It is cut in time order into train, val and test; each split then gets
whole attack episodes at uniform random onsets, and every frame is emitted in time order. Each
frame's meter jitter comes from its own stream keyed by (seed, timestep), so two timelines on one
seed and pool that place different attacks carry the same benign frames.
`generate_timeline` builds the IEEE source (`generation.ieee.IeeeSource`) and runs the one path
every timeline takes, `generation.source.TimelineSource.generate`. The file layout and every knob
are in docs/guides/generation.md.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .generation.ieee import IeeeSource
from .generation.source import OutPath
from .models.config import TimelineSettings


def generate_timeline(
    system: Union[int, str],
    states: Optional[Union[str, np.ndarray]] = None,
    settings: Optional[TimelineSettings] = None,
    seed: int = 123,
    out: Optional[OutPath] = None,
    **knobs: object,
) -> str:
    """Walk one attacked timeline over the IEEE `system`'s operating-point pool, write it as one HDF5
    file and return the path (default: `timeline_ieee{N}.h5` under the cache directory). A flat keyword
    in `knobs` (attacked_frac, families, ramp_rate, hops, ...; the table is in docs/guides/generation.md)
    overrides the same field of `settings` (`TimelineSettings.of`)."""
    return IeeeSource(system, states, TimelineSettings.of(settings, **knobs), seed).generate(out)
