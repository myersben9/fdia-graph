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


HEAVY = ("pandapower", "torch", "torch_geometric", "pandas", "scipy")


def _fresh_import_modules(statement: str) -> str:
    """Run `statement` in a fresh interpreter and return the top-level packages it left loaded."""
    import subprocess

    code = f"import sys; {statement}; print(sorted({{m.split('.')[0] for m in sys.modules}}))"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout


def test_import_does_not_pull_in_heavy_dependencies():
    """`import fdia_graph` loads neither the optional extras nor the helper modules that need them."""
    loaded = _fresh_import_modules("import fdia_graph")
    assert not any(f"'{h}'" in loaded for h in HEAVY), loaded
    assert "'fdia_graph.generation'" not in loaded and "'fdia_graph.torch_data'" not in loaded


def test_star_import_resolves_the_lazy_names_without_heavy_dependencies():
    """`from fdia_graph import *` binds every name in __all__, which resolves the lazy ones and so
    imports their modules; those modules import pandapower, torch and torch_geometric lazily
    themselves, so the star import still costs no optional dependency."""
    loaded = _fresh_import_modules("from fdia_graph import *")
    assert not any(f"'{h}'" in loaded for h in HEAVY), loaded
