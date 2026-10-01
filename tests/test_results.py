"""The results store: records are checked on construction, the store round-trips and keeps history,
a run records its provenance, nested reports flatten to records, and the docs' results blocks render
from the store and fail the check when edited by hand."""

from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest

from fdia_graph.errors import ConfigError, NoSuchResult
from fdia_graph.results import METRICS, Record, Run, Store, fill, stale
from fdia_graph.results.run import config_hash

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rec(**kw: object) -> Record:
    return Record(**{"experiment": "demo.x", "metric": "angle_mae_deg", "value": 0.1, **kw})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        {"metric": "angle_mae"},  # not registered
        {"value": math.nan},
        {"value": math.inf},
        {"experiment": "Demo X"},  # not a slug
        {"tags": ("k1.2",)},  # not name=value
        {"tags": ("k=1", "k=2")},  # the same tag twice
        {"sd": -1.0},
        {"ci_lo": 0.1},  # without ci_hi
        {"ci_lo": 0.2, "ci_hi": 0.1},
    ],
)
def test_a_record_refuses_malformed_input(bad: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        _rec(**bad)


def test_every_registered_metric_formats_a_number() -> None:
    for name, m in METRICS.items():
        assert format(1, "d") if m.fmt == "d" else format(1.0, m.fmt), name


def test_the_store_round_trips_and_a_rerun_replaces_only_its_own_rows(tmp_path: object) -> None:
    store = Store(str(tmp_path))
    with Run(
        "demo.x", system="ieee14", store=store, run_id="demo.x-a", timestamp="2026-01-01T00:00:00Z"
    ) as run:
        run.add("angle_mae_deg", 0.1, method="wls", family="geo")
        run.add("devices", 6, method="search", k=1.1, sd=0.5)
    with Run(
        "demo.x", system="ieee14", store=store, run_id="demo.x-b", timestamp="2026-02-01T00:00:00Z"
    ) as run:
        run.add("angle_mae_deg", 0.2, method="wls", family="geo")
    assert len(store.query("demo.x")) == 3
    assert store.one("demo.x", method="search", k="1.1").sd == 0.5
    assert store.one("demo.x", method="wls").value == 0.2  # the newest run wins
    assert {r.value for r in store.query("demo.x", method="wls")} == {0.1, 0.2}  # history kept
    with Run(
        "demo.x", system="ieee14", store=store, run_id="demo.x-a", timestamp="2026-01-01T00:00:00Z"
    ) as run:
        run.add("angle_mae_deg", 0.3, method="wls", family="geo")  # run a again: its old rows go
    assert sorted(r.value for r in store.query("demo.x")) == [0.2, 0.3]
    assert [p.run_id for p in store.runs()] == ["demo.x-a", "demo.x-b"]


def test_one_refuses_an_ambiguous_query(tmp_path: object) -> None:
    store = Store(str(tmp_path))
    with Run("demo.x", store=store) as run:
        run.add("angle_mae_deg", 0.1, method="a")
        run.add("angle_mae_deg", 0.2, method="b")
    with pytest.raises(NoSuchResult):
        store.one("demo.x")
    with pytest.raises(NoSuchResult):
        store.one("demo.x", method="c")


def test_a_failed_run_writes_nothing(tmp_path: object) -> None:
    store = Store(str(tmp_path))
    with pytest.raises(RuntimeError):
        with Run("demo.x", store=store) as run:
            run.add("angle_mae_deg", 0.1)
            raise RuntimeError("half way")
    assert store.query("demo.x") == [] and store.runs() == []


def test_a_run_records_its_provenance(tmp_path: object) -> None:
    import fdia_graph

    run = Run(
        "demo.x",
        system="ieee14",
        settings={"c": 1.5},
        data_release="v0.9.0",
        seed=3,
        store=Store(str(tmp_path)),
    )
    p = run.provenance
    assert p.sdk_version == fdia_graph.__version__ and p.data_release == "v0.9.0" and p.seed == 3
    assert p.config_hash == config_hash({"c": 1.5}) != config_hash({"c": 2.0})
    assert p.run_id.startswith("demo.x-") and p.timestamp.endswith("Z")


def test_add_tree_flattens_a_nested_report(tmp_path: object) -> None:
    run = Run("demo.x", store=Store(str(tmp_path)))
    report = {
        "wls": {"geo": {"angle_mae_deg": 0.1, "voltage_mae_pu": 1e-4}, "note": "skipped"},
        "fed": {"all": {"macro_f1": {"mean": 0.8, "std": 0.01}}},
        "dqn": {"seq": {"attack_cost": [3.0, 45.0]}},
    }
    assert run.add_tree(report, levels=("method", "family")) == 5
    by = {(r.method, r.family, r.metric, r.tag("step")): r for r in run.records}
    assert by[("fed", "all", "macro_f1", "")].sd == 0.01
    assert by[("dqn", "seq", "attack_cost", "1")].value == 45.0
    with pytest.raises(ConfigError):  # a leaf whose key is no metric is refused, not stored under a typo
        run.add_tree({"wls": {"geo": {"angle": 0.1}}}, levels=("method", "family"))


def test_blocks_render_from_the_store_and_go_stale_when_edited(tmp_path: object) -> None:
    store = Store(str(tmp_path))
    with Run("demo.x", system="ieee14", store=store) as run:
        run.add("angle_mae_deg", 0.1, method="wls", family="geo")
        run.add("angle_mae_deg", 0.05, method="prior", family="geo")
    text = (
        "WLS reads <!-- results: value experiment=demo.x method=wls metric=angle_mae_deg --><!-- /results --> deg, "
        "a <!-- results: reduction experiment=demo.x base=wls new=prior metric=angle_mae_deg --><!-- /results -->% cut.\n"
        "<!-- results: table experiment=demo.x rows=method cols=system metric=angle_mae_deg order=wls,prior -->\n<!-- /results -->\n"
    )
    done = fill(text, store)
    assert (
        "-->0.100<!--" in done
        and "-->50<!--" in done
        and "| wls | 0.100 |" in done
        and "| prior | 0.050 |" in done
    )
    assert stale(done, store) == []
    assert stale(done.replace("-->0.100<!--", "-->0.101<!--"), store) == [
        "value experiment=demo.x method=wls metric=angle_mae_deg"
    ]


def _tool(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "results_docs.py"), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_the_repository_docs_match_the_store() -> None:
    out = _tool("--check")
    assert out.returncode == 0, out.stdout + out.stderr


def test_the_check_fails_on_a_hand_edited_block_and_lint_flags_a_typed_number(tmp_path: object) -> None:
    doc = os.path.join(ROOT, "docs", "se", "README.md")
    text = open(doc, encoding="utf-8").read()
    start = text.index("<!-- results: red se ieee14 geo angle_mae_deg wls prior+huber -->")
    end = text.index("<!-- /results -->", start)
    edited = text[:start] + "<!-- results: red se ieee14 geo angle_mae_deg wls prior+huber -->99" + text[end:]
    path = os.path.join(str(tmp_path), "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(edited)
    assert _tool("--check", path).returncode == 1
    typed = os.path.join(str(tmp_path), "typed.md")
    with open(typed, "w", encoding="utf-8") as f:
        f.write(
            "A block <!-- results: v se ieee14 wls geo angle_mae_deg --><!-- /results --> and a typed 0.123 here.\n"
        )
    lint = _tool("--lint", typed).stdout
    assert "1 number(s)" in lint and "0.123" in lint


def test_a_block_inside_fenced_code_is_an_example_and_stays_as_written(tmp_path: object) -> None:
    store = Store(str(tmp_path))
    with Run("demo.x", store=store) as run:
        run.add("angle_mae_deg", 0.1, method="wls")
    example = (
        "```text\n<!-- results: value experiment=demo.x metric=angle_mae_deg -->?<!-- /results -->\n```\n"
    )
    assert fill(example, store) == example and stale(example, store) == []


def test_back_to_back_writes_keep_every_run(tmp_path: object) -> None:
    from fdia_graph.results import Provenance

    store = Store(str(tmp_path))
    for i in range(30):  # many writes inside one file-time tick
        prov = Provenance(
            run_id=f"demo.x-{i}", experiment="demo.x", timestamp=f"2026-01-01T00:00:{i:02d}Z", sdk_version="0"
        )
        store.write(prov, [Record(experiment="demo.x", metric="devices", value=float(i), method=str(i))])
    assert len(store.query("demo.x")) == 30 and len(store.runs()) == 30


def test_two_runs_of_one_second_and_settings_get_distinct_ids(tmp_path: object) -> None:
    store = Store(str(tmp_path))
    stamp = "2026-01-01T00:00:00Z"
    a = Run("demo.x", system="ieee14", settings={"k": 1}, store=store, timestamp=stamp)
    b = Run("demo.x", system="ieee118", settings={"k": 1}, store=store, timestamp=stamp)
    c = Run("demo.x", system="ieee14", settings={"k": 1}, store=store, timestamp=stamp)
    assert len({a.provenance.run_id, b.provenance.run_id, c.provenance.run_id}) == 3


def test_a_tag_filter_takes_any_of_several_values(tmp_path: object) -> None:
    store = Store(str(tmp_path))
    with Run("demo.x", store=store) as run:
        for k in ("1.1", "1.2", "1.3"):
            run.add("devices", 5, method="s", k=k)
    assert {r.tag("k") for r in store.query("demo.x", k=["1.1", "1.2"])} == {"1.1", "1.2"}
    text = "<!-- results: table experiment=demo.x rows=k cols=method metric=devices k=1.1,1.2 -->\n<!-- /results -->"
    assert "| 1.1 | 5 |" in fill(text, store) and "1.3" not in fill(text, store)


def test_a_table_refuses_an_empty_or_ambiguous_selection(tmp_path: object) -> None:
    store = Store(str(tmp_path))
    with Run("demo.x", system="ieee14", store=store) as run:
        run.add("devices", 5, method="s", k="1.1")
        run.add("devices", 6, method="s", k="1.2")
    for spec in (
        "table experiment=demo.x rows=method cols=system metric=devices methd=s",  # a typo selects nothing
        "table experiment=demo.x rows=method cols=system metric=devices",  # k left open: two on a cell
        "table experiment=demo.x rows=method cols=system metric=devices k=1.1 order=s,t",  # no row t
    ):
        with pytest.raises(NoSuchResult):
            fill(f"<!-- results: {spec} -->\n<!-- /results -->", store)


def test_one_spec_inline_and_as_a_block_is_not_stale_after_fill(tmp_path: object) -> None:
    store = Store(str(tmp_path))
    with Run("demo.x", system="ieee14", store=store) as run:
        run.add("devices", 5, method="s")
    spec = "table experiment=demo.x rows=method cols=system metric=devices"
    text = (
        f"inline <!-- results: {spec} --><!-- /results -->\n\n<!-- results: {spec} -->\n<!-- /results -->\n"
    )
    assert stale(fill(text, store), store) == []


def test_tilde_and_indented_fences_are_examples(tmp_path: object) -> None:
    store = Store(str(tmp_path))
    for fence in (
        "~~~\n<!-- results: v x -->?<!-- /results -->\n~~~\n",
        "- item\n\n  ```\n  <!-- results: v x -->?<!-- /results -->\n  ```\n",
    ):
        assert fill(fence, store) == fence and stale(fence, store) == []


def test_a_run_measures_each_key_once(tmp_path: object) -> None:
    from fdia_graph.results import Provenance

    prov = Provenance(
        run_id="demo.x-1", experiment="demo.x", timestamp="2026-01-01T00:00:00Z", sdk_version="0"
    )
    twice = [Record(experiment="demo.x", metric="devices", value=v, method="s") for v in (1.0, 2.0)]
    with pytest.raises(ConfigError):
        Store(str(tmp_path)).write(prov, twice)


def test_an_unterminated_block_fails_the_check(tmp_path: object) -> None:
    text = "fine <!-- results: v se ieee14 wls geo angle_mae_deg -->0.091 and no closing marker\n"
    assert stale(text, Store(str(tmp_path))) != []


def test_latest_and_newest_run_agree_on_a_tie(tmp_path: object) -> None:
    store = Store(str(tmp_path))
    for rid, v in (("demo.x-a", 1.0), ("demo.x-b", 2.0)):
        with Run("demo.x", store=store, run_id=rid, timestamp="2026-01-01T00:00:00Z") as run:
            run.add("devices", v, method="s")
    assert store.one("demo.x").run_id == store.newest_run("demo.x") == "demo.x-b"


def test_a_writer_never_enters_a_held_lock(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    from fdia_graph.results import store as store_mod

    monkeypatch.setattr(store_mod._Lock, "WAIT_S", 0.2)
    lock = os.path.join(str(tmp_path), ".lock")
    open(lock, "w").close()  # another writer holds it, freshly
    with pytest.raises(TimeoutError):
        with Run("demo.x", store=Store(str(tmp_path))) as run:
            run.add("devices", 1.0)
    assert os.path.exists(lock)  # the other writer's lock is left alone


def test_a_whole_system_rerun_with_an_arm_skipped_leaves_no_stale_number(tmp_path: object) -> None:
    """A harness run that covers its system whole (`replaces=True`) drops the experiment's earlier
    records on that system: an arm the rerun skipped reads as missing, never as its old number, and
    the other systems' records stay."""
    store = Store(str(tmp_path))
    for system in ("ieee14", "ieee118"):
        with Run("demo.x", system=system, store=store, replaces=True) as run:
            run.add("angle_mae_deg", 0.1, method="wls", family="geo")
            run.add("angle_mae_deg", 0.2, method="removal", family="geo")
    with Run("demo.x", system="ieee14", store=store, replaces=True) as run:  # removal skipped this time
        run.add("angle_mae_deg", 0.05, method="wls", family="geo")
    assert [r.value for r in store.latest("demo.x", system="ieee14")] == [0.05]
    assert not store.latest("demo.x", system="ieee14", method="removal")
    assert len(store.latest("demo.x", system="ieee118")) == 2


def test_a_whole_system_rerun_that_wrote_nothing_still_replaces(tmp_path: object) -> None:
    """The scope of `replaces=True` is the run's system, not the systems its rows happen to cover:
    a rerun that wrote nothing on its system still drops the old records there."""
    store = Store(str(tmp_path))
    with Run("demo.x", system="ieee14", store=store, replaces=True) as run:
        run.add("angle_mae_deg", 0.1, method="wls", family="geo")
    with Run("demo.x", system="ieee14", store=store, replaces=True):
        pass  # every arm skipped
    assert not store.latest("demo.x", system="ieee14")
