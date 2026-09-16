"""The package namespace: the heavy helpers resolve lazily to the real functions, so their
docstrings and signatures cannot drift from a wrapper's, and `import fdia_graph` stays light."""

import sys

import fdia_graph as fg


def test_lazy_names_resolve_to_the_real_functions():
    import fdia_graph.engine
    import fdia_graph.generation
    import fdia_graph.profiles
    import fdia_graph.streams
    import fdia_graph.torch_data

    assert fg.generate is fdia_graph.generation.generate
    assert fg.generate_stream is fdia_graph.streams.generate_stream
    assert fg.load_stream is fdia_graph.streams.load_stream and fg.windows is fdia_graph.streams.windows
    assert fg.pyg_stream is fdia_graph.torch_data.pyg_stream
    assert fg.torch_windows is fdia_graph.torch_data.torch_windows
    assert fg.load_profile is fdia_graph.profiles.load_profile
    assert fg.fetch_profile is fdia_graph.profiles.fetch_profile
    assert fg.generate_states is fdia_graph.profiles.generate_states
    assert fg.line_outage_candidates is fdia_graph.engine.line_outage_candidates


def test_all_names_exist_and_dir_lists_them():
    assert all(hasattr(fg, n) for n in fg.__all__)
    assert set(fg.__all__) <= set(dir(fg))
    from fdia_graph import generate, windows  # the `from` form goes through the same hook

    assert callable(generate) and callable(windows)


def test_unknown_attribute_still_raises():
    try:
        fg.no_such_thing
    except AttributeError as e:
        assert "no_such_thing" in str(e)
    else:
        raise AssertionError("expected AttributeError")


def test_import_does_not_pull_in_the_generators():
    """A fresh interpreter importing the package must not import pandapower-backed modules."""
    import subprocess

    code = "import sys, fdia_graph; print(sorted(m for m in sys.modules if m.startswith('fdia_graph.')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert (
        "fdia_graph.generation" not in out
        and "fdia_graph.engine" not in out
        and "fdia_graph.torch_data" not in out
    )
