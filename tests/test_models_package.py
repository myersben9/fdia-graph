"""Every data model of a toolkit lives in its `models` package: the package names them all, they are
defined nowhere else, and its own modules depend on nothing but numpy and the standard library. The
checks are functions of the package (tests/helpers.py), so the analysis toolkit holds its own `fdia_analysis.models` to
them (analysis/tests/test_models_package.py)."""

import fdia_graph
import fdia_graph.models as models
from fdia_graph.models import Bundle
from tests.helpers import STANDARD, models_defined, runtime_imports, stray_models

# the New York code keeps its records in its own modules until each is conformed (DEFERRED in
# tools/readability.py)
NEW_YORK = (
    "fdia_graph.generation.nygrid",
    "fdia_graph.engine.attacks.family",
    "fdia_graph.engine.sensing",
)
CONSTANTS = {"INTACT", "NODE", "EDGE", "CURRENT"}  # the instances the package exports next to the models


def test_every_public_model_is_defined_in_the_package():
    """A public Bundle or NamedTuple defined in a producer module would scatter the models again."""
    strays = stray_models(fdia_graph, skip=NEW_YORK)
    assert strays == {}, f"models defined outside fdia_graph.models: {strays}"


def test_all_names_every_model_and_public_is_a_subset():
    assert set(models.__all__) == models_defined(models) | {"Bundle"} | CONSTANTS
    assert set(models.PUBLIC) <= set(models.__all__)
    assert all(issubclass(getattr(models, n), Bundle) for n in models.PUBLIC)


def test_package_imports_only_numpy_and_the_standard_library():
    """Models never import a producer, so no import cycle is possible."""
    for module, imported in runtime_imports(models).items():
        assert imported <= STANDARD, f"fdia_graph.models.{module} imports {imported - STANDARD}"
