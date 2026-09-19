"""Every data model lives in fdia_graph.models (data models plan, step 5): the package names them
all, they are defined nowhere else, and its own modules depend on nothing but numpy and typing."""

import ast
import dataclasses
import importlib
import inspect
import os
import pkgutil

import fdia_graph
import fdia_graph.models as models
from fdia_graph.models import Bundle

SRC = os.path.dirname(fdia_graph.__file__)


def _is_model(obj) -> bool:
    """A Bundle, a NamedTuple, or a plain dataclass (the field groups)."""
    if not inspect.isclass(obj) or obj is Bundle:
        return False
    return (
        issubclass(obj, Bundle)
        or (issubclass(obj, tuple) and hasattr(obj, "_fields"))
        or dataclasses.is_dataclass(obj)
    )


def _defined_models(module):
    return {n for n, o in vars(module).items() if _is_model(o) and o.__module__ == module.__name__}


def test_every_public_model_is_defined_in_the_package():
    """A public Bundle or NamedTuple defined in a producer module would scatter the models again."""
    strays = {}
    for info in pkgutil.walk_packages([SRC], "fdia_graph."):
        if info.name.startswith("fdia_graph.models"):
            continue
        try:
            mod = importlib.import_module(info.name)
        except ImportError:  # an optional extra (torch, pandapower) is not installed
            continue
        public = {n for n in _defined_models(mod) if not n.startswith("_")}
        if public:
            strays[info.name] = sorted(public)
    assert strays == {}, f"models defined outside fdia_graph.models: {strays}"


def test_all_names_every_model_and_public_is_a_subset():
    defined = set()
    for info in pkgutil.iter_modules(models.__path__):
        defined |= _defined_models(importlib.import_module(f"fdia_graph.models.{info.name}"))
    constants = {"INTACT", "NODE", "EDGE", "BRANCH"}  # the instances the package exports next to the models
    assert defined | {"Bundle"} | constants == set(models.__all__) | constants
    assert set(models.PUBLIC) <= set(models.__all__)
    assert all(issubclass(getattr(models, n), Bundle) for n in models.PUBLIC)


def test_package_imports_only_numpy_and_typing():
    """Models never import a producer, so no import cycle is possible."""
    allowed = {"__future__", "dataclasses", "typing", "numpy"}
    for info in pkgutil.iter_modules(models.__path__):
        tree = ast.parse(open(os.path.join(SRC, "models", info.name + ".py"), encoding="utf8").read())
        typing_only = {
            n
            for stmt in ast.walk(tree)
            if isinstance(stmt, ast.If) and getattr(stmt.test, "id", "") == "TYPE_CHECKING"
            for n in ast.walk(stmt)
        }  # type-only imports (a torch tensor in an annotation) are not runtime dependencies
        for node in ast.walk(tree):
            if node in typing_only:
                continue
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]} if node.level == 0 else set()
            else:
                continue
            assert names <= allowed, f"fdia_graph.models.{info.name} imports {names - allowed}"


def test_old_import_paths_still_work():
    from fdia_graph.dataset import RecordBundle
    from fdia_graph.engine.base import INTACT, MeterPlan
    from fdia_graph.engine.records import Frame, Scan
    from fdia_graph.formulas.network import BranchModel
    from fdia_graph.registry import AssetSpec
    from fdia_graph.se.base import EstimatorScores, TrueState
    from fdia_graph.streams import Stream

    assert RecordBundle is models.RecordBundle and Stream is models.Stream and Frame is models.Frame
    assert Scan is models.Scan and BranchModel is models.BranchModel and AssetSpec is models.AssetSpec
    assert EstimatorScores is models.EstimatorScores and TrueState is models.TrueState
    assert MeterPlan is models.MeterPlan and INTACT is models.INTACT
