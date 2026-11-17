"""Every data model of the analysis toolkit lives in `fdia_analysis.models`: the package names them
all, they are defined nowhere else, no name of it is also a name of `fdia_graph.models`, and its own
modules depend on nothing but numpy, the standard library and the generation toolkit's models. The
checks are the generation toolkit's (tests/test_models_package.py), run on this package."""

import fdia_analysis
import fdia_analysis.formulas as formulas
import fdia_analysis.models as models
import fdia_graph.formulas
import fdia_graph.models
from fdia_graph.models import Bundle
from tests.helpers import STANDARD, models_defined, runtime_imports, stray_models

# the alarm rules came in with the New York code and keep their records in their own module until
# conformed (DEFERRED in tools/readability.py)
NEW_YORK = ("fdia_analysis.se.detect",)
GENERATION_MODELS = "fdia_graph.models"  # the one first-party package a model of this toolkit builds on


def test_every_public_model_is_defined_in_the_package():
    """A public Bundle or NamedTuple defined in a producer module would scatter the models again."""
    strays = stray_models(fdia_analysis, skip=NEW_YORK)
    assert strays == {}, f"models defined outside fdia_analysis.models: {strays}"


def test_all_names_every_model_and_public_is_a_subset():
    assert set(models.__all__) == models_defined(models)
    assert set(models.PUBLIC) <= set(models.__all__)
    assert all(issubclass(getattr(models, n), Bundle) for n in models.PUBLIC)


def test_no_name_is_exported_by_both_toolkits():
    """A model or a formula has one home: nothing `fdia_analysis` exports is also exported by the
    package of the same kind in `fdia_graph`."""
    assert not set(models.__all__) & set(fdia_graph.models.__all__)
    assert not set(formulas.__all__) & set(fdia_graph.formulas.__all__)


def test_package_imports_only_numpy_the_standard_library_and_the_generation_models():
    """Models never import a producer, so no import cycle is possible."""
    allowed = STANDARD | {GENERATION_MODELS, "fdia_analysis.models"}
    for module, imported in runtime_imports(models).items():
        assert imported <= allowed, f"fdia_analysis.models.{module} imports {imported - allowed}"
