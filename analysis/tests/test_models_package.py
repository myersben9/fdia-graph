"""Every data model of the analysis toolkit lives in `fdia_analysis.models`: they are defined nowhere
else, each is imported from the module that defines it (no package of the toolkit exports a name), no
name of it is also a name of `fdia_graph.models`, and its own modules depend on nothing but numpy, the
standard library and the generation toolkit's models. The checks are the generation toolkit's
(tests/test_models_package.py), run on this package."""

import inspect

import pytest

import fdia_analysis
import fdia_analysis.federated
import fdia_analysis.formulas as formulas
import fdia_analysis.localization
import fdia_analysis.models as models
import fdia_analysis.results
import fdia_analysis.se
import fdia_analysis.se.detect
import fdia_analysis.trust
import fdia_graph.formulas
import fdia_graph.models
from fdia_analysis.models import scores
from fdia_graph.models.base import Bundle
from tests.helpers import (
    STANDARD,
    exported_names,
    functions_defined,
    models_defined,
    runtime_imports,
    stray_models,
)

# the alarm rules came in with the New York code and keep their records in their own module until
# conformed (DEFERRED in tools/readability.py)
NEW_YORK = ("fdia_analysis.se.detect",)
GENERATION_MODELS = "fdia_graph.models"  # the one first-party package a model of this toolkit builds on
# the instances `fdia_graph.models.grid` holds beside its models
CONSTANTS = {"INTACT", "NODE", "EDGE", "CURRENT"}
PACKAGES = (
    models,
    formulas,
    fdia_analysis.results,
    fdia_analysis.federated,
    fdia_analysis.localization,
    fdia_analysis.trust,
    fdia_analysis.se,
    fdia_analysis.se.detect,
)


def test_every_public_model_is_defined_in_the_package():
    """A public Bundle or NamedTuple defined in a producer module would scatter the models again."""
    strays = stray_models(fdia_analysis, skip=NEW_YORK)
    assert strays == {}, f"models defined outside fdia_analysis.models: {strays}"


@pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.__name__)
def test_the_package_exports_no_name(package):
    """A name has one import path, its module's: every package of the toolkit holds only its modules."""
    assert exported_names(package) == set()


def test_the_bundles_a_user_receives_are_the_score_tables():
    """The bundles the data dictionary lists for this toolkit are the ones `models.scores` defines."""
    defined = {n for n, o in vars(scores).items() if inspect.isclass(o) and o.__module__ == scores.__name__}
    assert {n for n in defined if issubclass(getattr(scores, n), Bundle)} == {
        "ErrorPair",
        "EstimatorScores",
        "OverallMetrics",
        "BenignMetrics",
        "FamilyMetrics",
        "LocalizerScores",
        "PerBusMetrics",
        "PerBusScores",
        "GridScores",
        "TrustScores",
        "JacobianOutputs",
    }


def test_no_name_is_defined_by_both_toolkits():
    """A model or a formula has one home: nothing the modules of `fdia_analysis.models` and
    `fdia_analysis.formulas` define is also defined by the package of the same kind in `fdia_graph`."""
    assert not models_defined(models) & (models_defined(fdia_graph.models) | {"Bundle"} | CONSTANTS)
    assert not functions_defined(formulas) & functions_defined(fdia_graph.formulas)


def test_package_imports_only_numpy_the_standard_library_and_the_generation_models():
    """Models never import a producer, so no import cycle is possible."""
    allowed = STANDARD | {GENERATION_MODELS}  # its own modules are not counted, written with dots or in full
    for module, imported in runtime_imports(models).items():
        assert imported <= allowed, f"fdia_analysis.models.{module} imports {imported - allowed}"
