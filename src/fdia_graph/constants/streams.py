"""The keys of the random streams a timeline draws from. Each stream is seeded with the timeline's
seed, a key below and a position, so a draw of one kind never shifts a draw of another. The keys are
arbitrary and fixed: a different key gives a different file."""

from __future__ import annotations

JITTER_STREAM = 0x4A17  # the meter jitter of one frame: keyed by (seed, this, frame)
DESIGN_STREAM = 0xA3D5  # the one draw of an Am design: keyed by (seed, this, the design's task)
RELOCATE_STREAM = 0xA3D6  # the onset an Am episode with no feasible design is moved to
