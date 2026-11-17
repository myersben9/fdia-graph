"""Every data model of a toolkit lives in its `models` package: they are defined nowhere else, each is
imported from the module that defines it, and the package's own modules depend on nothing but numpy,
the standard library and the named errors (`fdia_graph.errors`). The
checks are functions of the package (tests/helpers.py), so the analysis toolkit holds its own `fdia_analysis.models` to
them (analysis/tests/test_models_package.py)."""

import inspect

import pytest

import fdia_graph
import fdia_graph.engine
import fdia_graph.formulas
import fdia_graph.generation
import fdia_graph.models as models
from fdia_graph.models import data
from fdia_graph.models.base import Bundle
from tests.helpers import (
    STANDARD,
    exported_names,
    imported_through,
    runtime_imports,
    stray_models,
    top_level_definitions,
)

# the New York code keeps its records in its own modules until each is conformed (DEFERRED in
# tools/readability.py)
NEW_YORK = (
    "fdia_graph.generation.nygrid",
    "fdia_graph.engine.attacks.family",
    "fdia_graph.engine.sensing",
)
ERRORS = "fdia_graph.errors"  # the one first-party module a model may import: the errors it raises


def test_every_public_model_is_defined_in_the_package():
    """A public Bundle or NamedTuple defined in a producer module would scatter the models again."""
    strays = stray_models(fdia_graph, skip=NEW_YORK)
    assert strays == {}, f"models defined outside fdia_graph.models: {strays}"


def test_the_package_exports_no_name_and_the_bundles_a_user_receives_are_in_data():
    """A model has one import path, its module's: the package itself holds only its modules. The bundles
    the data dictionary lists are the ones `models.data` defines."""
    assert exported_names(models) == set()
    public = {n for n, o in vars(data).items() if inspect.isclass(o) and o.__module__ == data.__name__}
    assert {n for n in public if issubclass(getattr(data, n), Bundle)} == {
        "RecordBundle",
        "BatchBundle",
        "ArraysBundle",
        "Summary",
        "EpisodeTable",
    }


def test_package_imports_only_numpy_the_standard_library_and_the_errors():
    """Models never import a producer, so no import cycle is possible. The one first-party module a
    model reads is `fdia_graph.errors` (what it raises), which itself imports nothing of the package."""
    allowed = STANDARD | {ERRORS}
    for module, imported in runtime_imports(models).items():
        assert imported <= allowed, f"fdia_graph.models.{module} imports {imported - allowed}"


def test_a_relative_import_of_a_producer_is_seen():
    """The guard reads `from ..errors import X` as `fdia_graph.errors`, so `from ..engine.core import X`
    would be refused; `models.frames` names the generator under TYPE_CHECKING only, which is no import."""
    imported = runtime_imports(models)
    assert ERRORS in imported["config"] and ERRORS in imported["validation"]
    assert "fdia_graph.engine" not in imported["frames"]


@pytest.mark.parametrize("package", [fdia_graph.formulas, fdia_graph.engine])
def test_the_package_exports_no_name(package):
    """`formulas` and `engine` hold only their modules: a formula or the generator is imported from the
    module that defines it."""
    assert exported_names(package) == set()


def test_generation_re_exports_nothing_it_does_not_define():
    """`generation/__init__.py` defines real functions (`generate`, the pool readers, the file header),
    so it has names; every name anything imports from it is one it defines itself, never one it only
    imported for its own use (`FdiaGenerator`, `TimelineSettings`, `register_local`)."""
    through = imported_through(fdia_graph.generation)
    assert through, "nothing imports from fdia_graph.generation: the scan is reading the wrong folders"
    assert through <= top_level_definitions(fdia_graph.generation)
