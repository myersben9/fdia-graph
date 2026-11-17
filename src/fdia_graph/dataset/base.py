"""The loader's shared state and the constants its concerns read.

`FdiaGraph` is five mixins over `DatasetBase` (`GraphMixin`, `AdmittanceMixin`, `RecordsMixin`,
`ExportMixin`, `SequenceMixin`). `DatasetBase` declares every attribute the constructor
sets, so each mixin type-checks on its own, and stubs the methods one mixin calls on another.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from collections.abc import Callable, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import pandas as pd

    from ..models.data import ArraysBundle, EpisodeTable


import h5py
import numpy as np

from .. import schema
from ..errors import MissingCapability
from ..models.choices import Capability
from ..models.inputs import FamilySelection, Requirement


def family_ids(families: Sequence[Union[str, int]]) -> list[int]:
    """Family names or integer codes as integer codes. An unknown name or code is an error."""
    return list(FamilySelection(families).codes)


def _torch() -> ModuleType:
    # Lazy torch import (+ install hint) so `import fdia_graph` stays light.
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError("PyTorch is required: pip install 'fdia-graph[torch]'") from e


# What a view may lack that a consumer needs: the test and what the consumer needs, in words.
CAPABILITIES: dict[str, tuple[Callable[[DatasetBase], bool], str]] = {
    Capability.BENIGN_LAYER: (lambda ds: ds.has_benign, "a timeline view with the benign layer"),
    Capability.CLEAN_LAYER: (lambda ds: ds.has_clean, 'a clean layer (or fit with calibrate="measured")'),
    Capability.PHYSICAL_UNITS: (
        lambda ds: ds.units == "physical",
        "units='physical' datasets (the default): it converts the stored units itself, so a per-unit "
        "view would be converted twice",
    ),
    Capability.TIME_ORDER: (lambda ds: ds._perm is None, "order='time'; this view is a random permutation"),
    Capability.CONSECUTIVE: (
        lambda ds: not len(ds.idx) or bool(np.all(np.diff(ds.idx) == 1)),
        "consecutive frames; a families= or heldout= view is not, nor is a split of whole weeks that "
        "interleave with the other splits' weeks (a New York file)",
    ),
    Capability.SPLIT: (lambda ds: ds.has_split, "a file with the split column"),
    Capability.SWING: (lambda ds: ds.has_swing, "a file with the swing layer"),
    Capability.TEMPORAL: (lambda ds: ds.has_temporal, "a file with the temporal_delta layer"),
    Capability.PMU_CURRENTS: (
        lambda ds: ds.has_currents,
        'a hybrid-meter timeline (meter_model="hybrid"), which carries PMU currents',
    ),
    Capability.RECTANGULAR_CURRENTS: (
        lambda ds: ds.has_currents and ds.current_form == schema.CURRENT_RECTANGULAR,
        "PMU currents stored as real and imaginary parts; a New York file stores magnitude and angle "
        "(its current_form attribute), which would be misread",
    ),
}


class DatasetBase:
    """Attributes set by `FdiaGraph.__init__` and read by the mixins."""

    path: str
    format: str
    units: str
    system: int
    N: int
    E: int
    baseMVA: float
    current_form: str
    edge_feat: list[str]  # the columns of a flow layer, by name (the file's legend)
    slack: Optional[int]
    idx: np.ndarray
    edge_index_np: np.ndarray
    has_physics: bool
    has_temporal: bool
    has_swing: bool
    has_clean: bool
    has_clean_full: bool
    has_benign: bool
    has_currents: bool
    has_split: bool
    edge_status_per_record: Optional[np.ndarray]
    _perm: Optional[np.ndarray]  # order="random": view position -> position in idx
    _episodes: Optional[EpisodeTable]  # the whole file's, None on a shard
    _f: Optional[h5py.File]
    _phys: dict[str, Optional[np.ndarray]]  # graph/* arrays, None where the file predates the schema
    _clean_np: Optional[np.ndarray]
    _eclean_np: Optional[np.ndarray]
    _eclean_full_np: Optional[np.ndarray]
    _mem: Optional[dict[str, np.ndarray]]

    def require(self, *capabilities: str, by: str) -> None:
        """Refuse this view when it lacks a capability `by` (the consumer, in words) needs; the
        tests and the messages are the `CAPABILITIES` table."""
        for cap in Requirement(capabilities, by).capabilities:
            test, needs = CAPABILITIES[cap]
            if not test(self):
                raise MissingCapability(f"{by} needs {needs}")

    def _frame_layers(self) -> dict[str, str]:
        """The layers this file stores per frame beside data/ (record field -> dataset path): the
        attack-removed scan, and the PMU currents of a hybrid-meter file."""
        return {
            **(schema.BENIGN_PATH if self.has_benign else {}),
            **(schema.CURRENT_PATH if self.has_currents else {}),
        }

    def __len__(self) -> int: ...

    # cross-mixin members (defined in the concern mixins)
    @property
    def edge_index(self) -> torch.Tensor: ...

    @property
    def edge_attr(self) -> torch.Tensor: ...

    def _physics_arrays(self, need: Sequence[str]) -> tuple[dict[str, np.ndarray], list[str]]:
        """The graph/* arrays among `need` that the file carries, and the names it lacks."""
        got = {k: v for k in need if (v := self._phys.get(k)) is not None}
        return got, [k for k in need if k not in got]

    def _h(self) -> h5py.File: ...

    def _to_units(self, arr: np.ndarray, kind: str) -> np.ndarray: ...

    def _clean_flows_full(self) -> Optional[np.ndarray]: ...

    def export(
        self, fields: Optional[Sequence[str]] = None, format: str = "numpy"
    ) -> Union[ArraysBundle, pd.DataFrame]: ...
