"""Build settings and the one place random generators are made."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Settings:
    nygrid_dir: Path = Path("data/NYgrid")  # NYgrid checkout at 47698b6
    out_dir: Path = Path("out")
    year: int = 2019
    seed: int = 123


def rng(settings: Settings, stage: str) -> np.random.Generator:
    """A generator for one build stage. Each stage gets its own stream, so rerunning or changing one
    stage never shifts the random draws of another."""
    return np.random.default_rng([settings.seed, zlib.crc32(stage.encode())])
