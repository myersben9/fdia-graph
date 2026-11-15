"""Settings are passed in, never patched onto shared objects: a widened case and an overshoot variant are copies,
and the originals stay as they were (no data needed)."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest


def _toy_case():
    from fdia_graph.generation.nygrid.case import Case

    return Case(**{f.name: (100.0 if f.name == "base_mva" else np.ones(3)) for f in fields(Case)})


def test_voltage_margin_is_a_copy():
    case = _toy_case()
    wide = case.with_voltage_margin(0.02)
    assert np.allclose(wide.vmin, 0.98) and np.allclose(wide.vmax, 1.02)
    assert np.allclose(case.vmin, 1.0) and np.allclose(case.vmax, 1.0)  # the original is untouched


def test_overshoot_variant_is_a_copy():
    pytest.importorskip("pandas")  # the interface module reads the hourly interface tables
    from fdia_graph.engine.attacks.interface import InterfaceFdia

    fam = object.__new__(InterfaceFdia)  # the copy semantics alone, without an attacker
    fam.overshoot = 0.0
    far = fam.with_overshoot(175.0)
    assert far.overshoot == 175.0 and fam.overshoot == 0.0 and far is not fam
