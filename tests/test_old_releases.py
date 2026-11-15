"""The pinned data releases stay readable. The generator makes only At and Am, but a timeline of each of
v0.8.0, v0.8.1 and v0.8.3 loads, splits, filters by family name and scores per family as before; the
record shards and stream files of the releases before v0.8.0 are refused. Each fixture under tests/data was written by the SDK
that built its release, or is a slice of the published file (tests/data/README.md).

Also here: the graph fields a GNN user relies on (the series admittance and the clean flow on every
branch), checked on a timeline."""

import os

import numpy as np
import pytest

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TIMELINE_V083 = os.path.join(DATA, "tiny_timeline_v083.h5")
# one timeline per data release, each written by the SDK that built that release
TIMELINES = {
    "v0.8.0": os.path.join(DATA, "tiny_timeline_v080.h5"),
    "v0.8.1": os.path.join(DATA, "tiny_timeline_v081.h5"),
    "v0.8.3": TIMELINE_V083,
}
OLD_FAMILIES = {"Aq", "Ad", "As", "Ar", "At", "Al", "Am"}


def _family_names(ds):
    from fdia_graph.dataset import FAMILIES

    return {FAMILIES[int(f)] for f in np.unique(ds.export(["family"])["family"])} - {"benign"}


@pytest.mark.parametrize("release", sorted(TIMELINES))
def test_an_old_timeline_loads_every_split_and_family(release):
    from fdia_graph.dataset import FdiaGraph

    path = TIMELINES[release]
    ds = FdiaGraph(path)
    assert ds.has_benign and ds.has_clean and _family_names(ds) == OLD_FAMILIES
    sizes = [len(FdiaGraph(path, split=s)) for s in ("train", "val", "test")]
    assert all(n > 0 for n in sizes) and sum(sizes) == len(ds)
    only = FdiaGraph(path, families=["Aq", "Al"])
    assert _family_names(only) == {"Aq", "Al"}
    held = FdiaGraph(path, split="train", heldout=True)
    assert not {"As", "Ar"} & _family_names(held)
    rec = ds[0]
    assert tuple(rec["clean"].shape) == (14, 4) and tuple(rec["edge_clean_full"].shape) == (ds.E, 2)
    assert ds.edge_attr_np.shape == (ds.E, 8)
    assert len(ds.episodes.onset) > 0 and ds.windows(8)[0].shape[1] == 8


def test_a_v083_timeline_reads_as_pyg():
    pytest.importorskip("torch_geometric")
    from fdia_graph.dataset import FdiaGraph

    g = FdiaGraph(TIMELINE_V083, split="test", format="pyg")[0]
    assert g.edge_index.shape[0] == 2 and g.edge_phys.shape[1] == 8  # the static [E,8] physics
    assert (
        g.edge_attr.shape[1] == 2 and g.edge_clean_full.shape[1] == 2
    )  # flows, and the clean flow everywhere


@pytest.mark.parametrize("release", sorted(TIMELINES))
def test_old_releases_score_per_old_family(release):
    """A per-family estimator and localizer score on an old timeline names the old families."""
    from fdia_graph.dataset import FdiaGraph
    from fdia_graph.localization import SwingThreshold
    from fdia_graph.se import WLS

    path = TIMELINES[release]
    train, test = FdiaGraph(path, split="train"), FdiaGraph(path, split="test")
    se = WLS().fit(train).score(test)
    loc = SwingThreshold().fit(train).score(test)
    present = _family_names(test)
    assert present <= OLD_FAMILIES and {"Aq", "Ad"} <= present
    for name in present:
        assert se[name] is not None and loc[name] is not None


@pytest.mark.parametrize("release", ["v0.8.0", "v0.8.1", "v0.8.3"])
def test_every_timeline_release_resolves_to_its_asset(release):
    from fdia_graph import registry

    spec = registry.resolve("ieee14", release=release)
    assert spec.file == "timeline_ieee14.h5" and spec.sha256 == registry._SHA256[release]["ieee14"]


@pytest.mark.parametrize("release", ["v0.7.1", "v0.7.2"])
def test_a_release_before_the_timelines_is_refused(release):
    from fdia_graph import registry

    with pytest.raises(ValueError, match="fdia-graph 0.21"):
        registry.resolve("ieee14", release=release)


def test_a_file_that_is_no_timeline_is_refused(tmp_path):
    """A record shard (no `kind` attribute) is refused by name, before anything is read from it."""
    import h5py

    from fdia_graph.dataset import FdiaGraph
    from fdia_graph.errors import NotATimeline

    path = tmp_path / "shard.h5"
    with h5py.File(path, "w") as f:
        f.attrs.update({"N": 14, "E": 20})
    with pytest.raises(NotATimeline, match="fdia-graph 0.21"):
        FdiaGraph(str(path))


def test_the_series_admittance_is_one_over_the_series_impedance(timeline):
    """edge_gs + j edge_bs = 1 / (r + j x) on every branch of a timeline (edge_attr columns 4, 5)."""
    import fdia_graph as fg

    ds = fg.load(timeline)
    a = ds.edge_attr_np.astype(np.float64)
    z = a[:, 0] + 1j * a[:, 1]
    want = np.where(np.abs(z) > 1e-12, 1.0 / np.where(np.abs(z) > 1e-12, z, 1.0), 0.0)
    assert np.allclose(a[:, 4] + 1j * a[:, 5], want, rtol=1e-5, atol=1e-6)


def test_the_clean_flow_on_every_branch_matches_the_metered_clean_flow(timeline):
    """edge_clean_full carries the clean flow on every branch and equals edge_clean wherever a flow
    meter reads it (edge_clean zeroes the unmetered ones)."""
    import fdia_graph as fg

    d = fg.load(timeline, split="test").export(["edge_clean", "edge_clean_full", "edge_m"])
    metered = d["edge_m"].astype(bool)
    assert np.allclose(d["edge_clean_full"][metered], d["edge_clean"][metered], rtol=1e-4, atol=1e-3)
    assert np.all(d["edge_clean"][~metered] == 0) and np.any(d["edge_clean_full"][~metered] != 0)
