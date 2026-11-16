"""Whole-file digests of a generated HDF5 file: the proof that a refactor did not move a number.

`file_digest` records the sha256 of every dataset's bytes and every attribute of a file. `check_file`
compares that with a snapshot under a `golden/` folder, bit for bit.

Floating-point results are not bit-identical across platforms and library builds, so the comparison runs
only on the machine that recorded the snapshot (`recorded_on`) and is skipped elsewhere. A refactor is
checked on that machine before it is pushed.

Set `FDIA_UPDATE_DIGESTS=1` to rewrite a snapshot after an intended change; the change's pull request
says which datasets moved and why.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pytest

UPDATE = os.environ.get("FDIA_UPDATE_DIGESTS") == "1"


def recorded_on() -> str:
    """The machine a snapshot holds on: operating system and processor, Python (minor) and numpy."""
    python = ".".join(platform.python_version_tuple()[:2])
    return f"{platform.system()} {platform.machine()} | python {python} | numpy {np.__version__}"


def _attrs(obj: h5py.HLObject) -> dict[str, str]:
    """Every attribute of a file, group or dataset as text, by name."""
    return {k: str(np.asarray(v).tolist()) for k, v in sorted(obj.attrs.items())}


def _dataset(d: h5py.Dataset) -> dict[str, object]:
    """One dataset: dtype, shape, the hash of its bytes (of its text, for strings) and its attributes."""
    a = np.asarray(d[()])  # a scalar dataset reads as a 0-d array
    data = np.ascontiguousarray(a).tobytes() if a.dtype.kind in "biuf" else str(a.tolist()).encode("utf-8")
    return {
        "dtype": str(d.dtype),
        "shape": list(d.shape),
        "sha256": hashlib.sha256(data).hexdigest(),
        "attrs": _attrs(d),
    }


def file_digest(path: Union[str, Path]) -> dict[str, object]:
    """Every dataset of the file (by its path) and the attributes of the file and of every group."""
    datasets: dict[str, object] = {}
    groups: dict[str, object] = {}

    def visit(name: str, obj: h5py.HLObject) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets[name] = _dataset(obj)
        else:
            groups[name] = _attrs(obj)

    with h5py.File(path, "r") as f:
        f.visititems(visit)
        root = _attrs(f)
    return {"recorded_on": recorded_on(), "attrs": root, "groups": groups, "datasets": datasets}


def check_file(name: str, path: Union[str, Path], golden: Path) -> None:
    """Compare the file at `path` with the snapshot `golden/<name>.json`, or write it with
    FDIA_UPDATE_DIGESTS=1. Skips on a machine other than the one that recorded the snapshot."""
    got = file_digest(path)
    snapshot = golden / f"{name}.json"
    if UPDATE:
        golden.mkdir(exist_ok=True)
        snapshot.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return
    assert snapshot.exists(), f"no snapshot {snapshot.name}; run once with FDIA_UPDATE_DIGESTS=1"
    want = json.loads(snapshot.read_text(encoding="utf-8"))
    if want["recorded_on"] != got["recorded_on"]:
        pytest.skip(f"{name} was recorded on {want['recorded_on']}; this is {got['recorded_on']}")
    for part in ("attrs", "groups", "datasets"):
        ours, theirs = got[part], want[part]
        assert isinstance(ours, dict)
        assert set(ours) == set(theirs), f"{name}: {part} differ: {sorted(set(ours) ^ set(theirs))}"
        for key in sorted(theirs):
            assert ours[key] == theirs[key], f"{name}: {part} {key}: {ours[key]!r} != {theirs[key]!r}"
