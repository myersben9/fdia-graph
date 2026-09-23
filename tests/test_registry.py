"""The registry's per-release asset layout: record shards and npz pools before v0.8.0, one timeline
per system and HDF5 pools from v0.8.0, and the deprecated `load_stream` following it."""

import numpy as np
import pytest

import fdia_graph as fg
from fdia_graph import registry


def test_release_tags_order_and_layout():
    assert registry.release_tuple("v0.7.2") == (0, 7, 2) and registry.release_tuple("0.10.1") == (0, 10, 1)
    with pytest.raises(ValueError, match="data release"):
        registry.release_tuple("latest")
    assert not registry.is_timeline_release("v0.7.2") and registry.is_timeline_release("v0.8.0")
    assert registry.is_timeline_release("v1.0.0")
    # package versions own the bare tags, so a data release from v0.8.0 lives under "data-"
    assert registry.release_tuple("data-v0.8.0") == (0, 8, 0)
    assert (
        registry.release_tag("v0.8.0") == "data-v0.8.0"
        and registry.release_tag("data-v0.8.0") == "data-v0.8.0"
    )
    assert registry.release_tag("v0.7.2") == "v0.7.2" and registry.release_name("data-v0.9.1") == "v0.9.1"
    assert registry.dataset_file(118, "v0.7.2") == "ml_only_ieee118.h5"
    assert registry.dataset_file(118, "v0.8.0") == "timeline_ieee118.h5"


def test_resolve_follows_the_release():
    old = registry.resolve("ieee14", release="v0.7.2")
    assert old.file == "ml_only_ieee14.h5" and old.release == "v0.7.2" and old.system == 14
    assert old.sha256 == registry._SHA256["v0.7.2"]["ieee14"]
    new = registry.resolve(300, release="v0.8.0")
    assert new.file == "timeline_ieee300.h5" and new.release == "data-v0.8.0"  # the tag the assets live under
    assert registry.resolve(300, release="data-v0.8.0").release == "data-v0.8.0"
    assert (
        registry.pool_spec(14, "v0.8.0").release == "data-v0.8.0"
        and registry.pool_spec(14, "v0.7.2").release == "v0.7.2"
    )
    assert new.sha256 == registry._SHA256.get("v0.8.0", {}).get("ieee300")
    assert registry.resolve("118").file == registry.dataset_file(118, registry._RELEASE)
    assert set(registry.BUILTIN) == {f"ieee{C}" for C in (14, 30, 57, 89, 118, 145, 200, 300)}
    with pytest.raises(KeyError, match="unknown dataset"):
        registry.resolve("ieee999")


def test_a_local_registration_shadows_a_builtin_name(timeline, tmp_path):
    """`list_datasets` reports a local entry under a built-in name as local; `resolve` agrees."""
    path = fg.load(timeline).path
    fg.register_local("ieee14", path)
    try:
        assert fg.list_datasets()["ieee14"] == "local"
        spec = registry.resolve("ieee14")
        assert spec.kind == "local" and spec.path == path
        assert registry.resolve("IEEE14").kind == "local"  # one name in either case, so it is shadowed alike
        assert fg.load("ieee14").is_timeline
    finally:
        local = registry._load_local()
        local.pop("ieee14", None)
        registry._save_local(local)
    assert registry.resolve("ieee14").kind == "builtin"


def test_pool_spec_follows_the_release():
    assert registry.pool_spec(14, "v0.7.2").file == "pool_ieee14.npz"
    assert registry.pool_spec("ieee300", "v0.8.0").file == "pool_ieee300.h5"
    spec = registry.pool_spec(57)
    assert (
        spec.release == registry.release_tag(registry._RELEASE)
        and spec.kind == "builtin"
        and spec.system == 57
    )


def test_load_stream_reads_a_timeline_release_through_the_loader(timeline, monkeypatch):
    """At a timeline release `load_stream` is the loader plus `stream_of`, so the dict it returns
    is the same frames `fg.load(name, order="time")` gives."""
    import fdia_graph.download as download

    path = fg.load(timeline).path
    monkeypatch.setattr(download, "ensure_local", lambda spec: path)
    monkeypatch.setattr(fg, "ensure_local", lambda spec: path)  # `load` bound the name at import
    with pytest.warns(DeprecationWarning, match="load_stream is deprecated"):
        s = fg.load_stream("ieee14", release="v0.8.0")
    ds = fg.load(timeline)
    assert s.system == 14 and s.node_x.shape == (len(ds), 14, 4) and s.node_m.shape == (14, 4)
    assert np.array_equal(s.y, ds.export(["y"])["y"]) and len(s.episodes) == len(ds.episodes)
    assert s.attacked_frac == pytest.approx(float((s.y.sum(axis=1) > 0).mean()))


def test_builtin_names_are_case_insensitive():
    assert registry.resolve("IEEE118")["system"] == 118
    assert registry.resolve(" ieee14 ")["system"] == 14


def test_newest_data_release_skips_a_malformed_tag():
    rel = [{"tag_name": t} for t in ("data-v0.8.0", "data-v0.9.0-rc1", "v0.18.0", "v0.7.2", "data-v0.8.1")]
    assert registry._newest_data_release(rel) == "v0.8.1"
    assert registry._newest_data_release([{"tag_name": "data-vX"}]) is None


def test_a_download_race_keeps_the_installed_file(tmp_path, monkeypatch):
    from fdia_graph import download

    dest, tmp = tmp_path / "a.h5", tmp_path / "a.part"
    dest.write_bytes(b"theirs")
    tmp.write_bytes(b"ours")

    def locked(src, dst):
        raise PermissionError("in use")  # Windows: the winner holds dest open

    monkeypatch.setattr(download.os, "replace", locked)
    download._install(str(tmp), str(dest))  # no error: the installed copy is kept
    assert dest.read_bytes() == b"theirs"
    import hashlib

    good = hashlib.sha256(b"theirs").hexdigest()
    download._install(str(tmp), str(dest), good)  # theirs matches the checksum: kept
    with pytest.raises(PermissionError):
        download._install(str(tmp), str(dest), hashlib.sha256(b"other").hexdigest())  # a stale file: not kept
    dest.unlink()
    with pytest.raises(PermissionError):
        download._install(str(tmp), str(dest))  # nothing installed: a real failure surfaces


def test_the_default_release_is_v081_with_pinned_timelines(monkeypatch):
    """0.19.0 reads data release v0.8.1 by default, every system pinned by sha256, under data-v0.8.1."""
    import importlib

    monkeypatch.delenv("FDIA_GRAPH_RELEASE", raising=False)
    reg = importlib.reload(registry)
    try:
        assert reg._RELEASE == "v0.8.1"
        for C in (14, 30, 57, 89, 118, 145, 200, 300):
            spec = reg.resolve(f"ieee{C}")
            assert spec["release"] == "data-v0.8.1" and spec["file"] == f"timeline_ieee{C}.h5"
            assert len(spec["sha256"]) == 64
        assert reg.resolve("ieee118", release="v0.8.0")["sha256"] != reg.resolve("ieee118")["sha256"]
    finally:
        importlib.reload(registry)
