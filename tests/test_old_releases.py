"""Old data releases stay readable. The generator makes only At and Am, but a timeline of each of
v0.8.0, v0.8.1 and v0.8.3, a v0.7.2 record shard and a v0.7.1 stream file load, split, filter by
family name and score per family as before. Each fixture under tests/data was written by the SDK
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
# the first 100 frames of the published IEEE-118 stream file
STREAM_V071 = os.path.join(DATA, "tiny_stream_v071.npz")
GRAPH_V071 = os.path.join(DATA, "graph_ieee118_v071.npz")  # that release's graph sidecar, as published
OLD_FAMILIES = {"Aq", "Ad", "As", "Ar", "At", "Al", "Am"}


def _family_names(ds):
    from fdia_graph.dataset import FAMILIES

    return {FAMILIES[int(f)] for f in np.unique(ds.export(["family"])["family"])} - {"benign"}


@pytest.mark.parametrize("release", sorted(TIMELINES))
def test_an_old_timeline_loads_every_split_and_family(release):
    from fdia_graph.dataset import FdiaGraph

    path = TIMELINES[release]
    ds = FdiaGraph(path)
    assert ds.is_timeline and ds.has_benign and ds.has_clean and _family_names(ds) == OLD_FAMILIES
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


def test_a_v072_record_shard_scores_per_family():
    from conftest import SHARD_V072

    from fdia_graph.dataset import FdiaGraph
    from fdia_graph.se import WLS

    ds = FdiaGraph(SHARD_V072)
    rep = WLS().fit(ds).score(ds)
    assert rep["benign"] is not None and rep["geo"].angle_mae_deg > 0
    present = _family_names(ds)
    assert present <= OLD_FAMILIES and {"Aq", "Ad", "As", "Ar", "At", "Al"} <= present
    for name in present:  # every attacked family of the shard is scored
        assert rep[name] is not None and rep[name].angle_mae_deg >= 0


def test_a_v071_stream_file_loads_and_windows(monkeypatch):
    """`load_stream` at a pre-timeline release reads the published stream file and that release's
    graph sidecar (the download replaced by the checked-in slices), and `streams.windows` slides
    over it."""
    import fdia_graph as fg
    import fdia_graph.download as download
    from fdia_graph.streams import windows

    files = {"stream_ieee118.npz": STREAM_V071, "graph_ieee118.npz": GRAPH_V071}
    monkeypatch.setattr(download, "ensure_local", lambda spec: files[spec.file])
    with pytest.warns(DeprecationWarning):
        s = fg.load_stream("ieee118", release="v0.7.1")
    assert s.system == 118 and s.node_x.shape == (100, 118, 4) and 0 < s.attacked_frac < 1
    assert s.edge_index.shape == (2, 186) and s.edge_attr.shape == (186, 8) and s.node_m.shape == (118, 4)
    assert len(s.episodes) > 0 and {int(e["family"]) for e in s.episodes} <= set(range(8))
    with pytest.warns(DeprecationWarning):
        X, y = windows(s, 8, stride=4)
    assert X.shape[1:] == (8, 118, 4) and len(X) == len(y)


@pytest.mark.parametrize("release", ["v0.7.1", "v0.7.2", "v0.8.0", "v0.8.1", "v0.8.3"])
def test_every_published_release_resolves_to_its_asset(release):
    from fdia_graph import registry

    spec = registry.resolve("ieee14", release=release)
    want = "timeline_ieee14.h5" if registry.is_timeline_release(release) else "ml_only_ieee14.h5"
    assert spec.file == want


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
