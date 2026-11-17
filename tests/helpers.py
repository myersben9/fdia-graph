"""The plain helpers the two test trees share (`tests/` and `analysis/tests/`): where the checkout's
parts are, the tiny timeline's recipe, the old releases' files, and the checks both toolkits hold
their own package to. The fixtures are in the repository root's `conftest.py`."""

import ast
import dataclasses
import importlib
import importlib.util
import inspect
import os
import pathlib
import pkgutil

import numpy as np
import pytest

from fdia_graph import schema
from fdia_graph.errors import ConfigError
from fdia_graph.models.base import Bundle
from fdia_graph.models.validation import Validated

# ---- the checkout ---------------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATION_SRC = os.path.join(REPO, "src")  # where fdia_graph is imported from
ANALYSIS_SRC = os.path.join(REPO, "analysis", "src")  # where fdia_analysis is


def interpreter_env(*source_roots: str) -> dict[str, str]:
    """The environment of a fresh interpreter that imports from `source_roots` of this checkout and
    from nothing else of it (pytest's own path is not inherited by a subprocess)."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join(source_roots)}


# ---- the tiny timeline ----------------------------------------------------------------------------
TINY = "tiny_ieee14"
# The shared tiny timeline: new generation's families (At and the overload Am) on the hybrid meters,
# 1000 frames with 20-frame episodes. The fewest-tamper search is off for At (min_tamper=False) and
# the overload Am searches at most 16 supports, so the fixture builds in about 20 s; the searched
# paths have their own small fixtures (test_minimize, test_overload, test_search_speed). Seed 2 puts
# both families in every split.
TIMELINE_KW = dict(frames=1000, ramp_len=20, seed=2, min_tamper=False, min_budget=16)

# ---- the old releases (tests/data/README.md) ------------------------------------------------------
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TIMELINE_V083 = os.path.join(DATA, "tiny_timeline_v083.h5")
# one timeline per data release, each written by the SDK that built that release
TIMELINES = {
    "v0.8.0": os.path.join(DATA, "tiny_timeline_v080.h5"),
    "v0.8.1": os.path.join(DATA, "tiny_timeline_v081.h5"),
    "v0.8.3": TIMELINE_V083,
}
OLD_FAMILIES = {"Aq", "Ad", "As", "Ar", "At", "Al", "Am"}


def family_names(ds) -> set[str]:
    """The attack families present in a loaded view."""
    from fdia_graph.constants.families import FAMILIES

    return {FAMILIES[int(f)] for f in np.unique(ds.export(["family"])["family"])} - {"benign"}


# ---- the file protocol ----------------------------------------------------------------------------
# every attribute name the schema defines, and the PMU current layers (all of them or none, by meter model)
ATTR_KEYS = frozenset(v for k, v in vars(schema.Attr).items() if not k.startswith("_"))
CURRENT_LAYERS = (schema.PMU_I, schema.PMU_I_M, schema.PMU_I_BENIGN, schema.PMU_I_TAMPER)


# ---- bundles --------------------------------------------------------------------------------------
def agree(b: Bundle) -> None:
    """A bundle's attribute view agrees with its dict view."""
    assert isinstance(b, dict) and isinstance(b, Bundle)
    for name in b._names():
        v = getattr(b, name)
        if v is None:
            assert b.key_of(name) not in b
        else:
            assert b[b.key_of(name)] is v


# ---- the IEEE cases -------------------------------------------------------------------------------
def ieee_net(system: int):
    """pandapower's IEEE case of `system` buses with its base power flow solved, the net a generator
    reads its `Case` from and does not keep."""
    import pandapower
    import pandapower.networks

    from fdia_graph.constants.grid import IEEE_CASES

    net = getattr(pandapower.networks, IEEE_CASES[system])()
    pandapower.runpp(net)
    return net


# ---- meter noise ----------------------------------------------------------------------------------
class FixedDraws:
    """Stands in for a random stream: every normal draw is `value` standard deviations from its mean."""

    def __init__(self, value: float) -> None:
        self.value = value

    def normal(self, loc: float, scale: np.ndarray, size: object = None) -> np.ndarray:
        shape = np.shape(scale) if size is None else size
        return loc + self.value * scale * np.ones(shape)


# ---- the fewest-tamper search ---------------------------------------------------------------------
WINDOW = 6  # snapshots per test window: a ramp's loads move above their noise by then, still fast


def windows(case, n, held=True):
    """n ramp windows accepted on today's region, each with its goal (one design per snapshot) from the
    ramp's first frame; with `held`, only windows where some support held for the window meets every
    constraint (the stealth bound rejects most: a reactive flow near zero has a noise floor of 1e-3).
    `case` is the fixture of that name."""
    from fdia_graph.engine.attacks.episodes import probe_frames
    from fdia_graph.models.frames import LoadGoal

    g, X, k = case
    rng = np.random.default_rng(3)
    out = []
    while len(out) < n:
        t = int(rng.integers(0, len(X) - WINDOW))
        d = g.ramp_design(X, t, (WINDOW, 0.002), k._replace(min_tamper=False))
        if d is None:
            continue
        frames = probe_frames(len(X), t, WINDOW)
        goal = LoadGoal(tuple(g.ramp_step(d, u - t, 0.002) for u in frames))
        states = [X[u] for u in frames]
        if held and g.min_tamper(states, goal, k).devices <= 0:
            continue  # no held support, or nothing moves above noise: nothing for the search to choose
        out.append((states, goal, d))
    return out


# ---- every model lives in a `models` package ------------------------------------------------------
# what a models module may import at run time: numpy and these parts of the standard library; a
# producer of the package is what is refused
STANDARD = frozenset(
    {
        "__future__",
        "dataclasses",
        "typing",
        "numpy",
        "enum",
        "re",
        "math",
        "numbers",
        "collections",
        "datetime",
        "os",
        "warnings",  # a deprecated input form warns where its model reads it
    }
)


def _is_model(obj) -> bool:
    """A Bundle, a NamedTuple, or a plain dataclass."""
    if not inspect.isclass(obj) or obj is Bundle:
        return False
    return (
        issubclass(obj, Bundle)
        or (issubclass(obj, tuple) and hasattr(obj, "_fields"))
        or dataclasses.is_dataclass(obj)
    )


def _defined_models(module) -> set[str]:
    return {n for n, o in vars(module).items() if _is_model(o) and o.__module__ == module.__name__}


def stray_models(package, skip: tuple[str, ...] = ()) -> dict[str, list[str]]:
    """The public models `package` defines outside its `models` package (and the modules of `skip`)."""
    strays = {}
    for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if info.name.startswith((package.__name__ + ".models", *skip)):
            continue
        try:
            mod = importlib.import_module(info.name)
        except ImportError:  # an optional extra (torch, pandapower) is not installed
            continue
        public = {n for n in _defined_models(mod) if not n.startswith("_")}
        if public:
            strays[info.name] = sorted(public)
    return strays


def models_defined(models_package) -> set[str]:
    """Every model the modules of a `models` package define."""
    defined: set[str] = set()
    for info in pkgutil.iter_modules(models_package.__path__):
        defined |= _defined_models(importlib.import_module(f"{models_package.__name__}.{info.name}"))
    return defined


def functions_defined(package) -> set[str]:
    """Every public function the modules of a package define (`fdia_graph.formulas`)."""
    defined: set[str] = set()
    for info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        defined |= {
            n
            for n, o in vars(module).items()
            if inspect.isfunction(o) and o.__module__ == module.__name__ and not n.startswith("_")
        }
    return defined


def exported_names(package) -> set[str]:
    """The public names `package` itself holds that are not its modules, once every module under it is
    imported: empty for a package whose `__init__.py` is a docstring."""
    for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            importlib.import_module(info.name)
        except ImportError:  # an optional extra (torch, pandapower, pyomo) is not installed
            continue
    return {n for n, o in vars(package).items() if not n.startswith("_") and not inspect.ismodule(o)}


def top_level_definitions(package) -> set[str]:
    """The names the `__init__.py` of `package` defines itself: its functions, classes and assignments."""
    defined: set[str] = set()
    for node in ast.parse(pathlib.Path(package.__file__).read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            defined |= {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    return defined


# every folder of Python that imports the two toolkits
SOURCE_FOLDERS = ("src", "analysis/src", "tests", "analysis/tests", "tools", "pipelines", "examples")


def imported_through(package) -> set[str]:
    """The names any module of the repository imports from `package` itself (`from package import X`,
    written absolutely or with dots), its own submodules left out."""
    target = package.__name__
    submodules = {info.name for info in pkgutil.iter_modules(package.__path__)}
    names: set[str] = set()
    for folder in SOURCE_FOLDERS:
        root = pathlib.Path(REPO, folder)
        source = folder if folder.endswith("src") else ""  # where a relative import can be resolved
        for path in root.rglob("*.py"):
            inside = ".".join(path.relative_to(root).parts[:-1])  # the package the file is in
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = node.module or ""
                if node.level:
                    if not source:
                        continue
                    module = importlib.util.resolve_name("." * node.level + module, inside)
                if module == target:
                    names |= {a.name for a in node.names}
    return names - submodules


def runtime_imports(models_package) -> dict[str, set[str]]:
    """module -> what it imports at run time, as "package" or, for a first-party import, "package.sub"
    (`fdia_graph.errors`). A relative import is resolved to the absolute name first (`from ..errors import
    X` in `fdia_graph.models.config` is `fdia_graph.errors`), so a producer cannot hide behind the dots.
    Type-only imports (a torch tensor in an annotation) are not dependencies, nor are a module's imports
    of its own package's siblings, written either way."""
    own = models_package.__name__
    out = {}
    for info in pkgutil.iter_modules(models_package.__path__):
        path = os.path.join(models_package.__path__[0], info.name + ".py")
        tree = ast.parse(open(path, encoding="utf8").read())
        typing_only = {
            n
            for stmt in ast.walk(tree)
            if isinstance(stmt, ast.If) and getattr(stmt.test, "id", "") == "TYPE_CHECKING"
            for n in ast.walk(stmt)
        }
        names: set[str] = set()
        for node in ast.walk(tree):
            if node in typing_only:
                continue
            if isinstance(node, ast.Import):
                names |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                target = "." * node.level + (node.module or "")
                module = importlib.util.resolve_name(target, own) if node.level else target
                # `from . import config` and `from .. import schema` name the module in the imported names
                names |= {f"{module}.{a.name}" for a in node.names} if not node.module else {module}
        short = {".".join(n.split(".")[:2]) if n.startswith("fdia_") else n.split(".")[0] for n in names}
        out[info.name] = short - {own}
    return out


# ---- a malformed input is a ConfigError -----------------------------------------------------------
BAD = ("bogus", 1.5, -1, float("nan"), float("inf"), 10**1000, None, object(), [1, 2], {"a": 1})


def models_of(*modules) -> list[type]:
    """Every checked model the modules define."""
    return [
        cls
        for mod in modules
        for _, cls in inspect.getmembers(mod, inspect.isclass)
        if issubclass(cls, Validated) and cls is not Validated and cls.__module__ == mod.__name__
    ]


def sweep(cls: type, base: dict) -> None:
    """Build `cls` from `base` with each field in turn set to each value of the wrong kind: an input
    error, or the data condition the model names, is fine; any other exception is the finding."""
    for field in dataclasses.fields(cls):
        for bad in BAD:
            try:
                cls(**{**base, field.name: bad})
            except (ConfigError, cls.error):
                pass
            except Exception as e:  # noqa: BLE001  (any other type is the finding)
                pytest.fail(f"{cls.__name__}({field.name}={bad!r}) raised {type(e).__name__}: {e}")


def check_functions(package) -> list[str]:
    """The functions of `package` named `check_...` outside its models: every check lives on a model."""
    root = pathlib.Path(package.__file__).parent
    return [
        f"{path.relative_to(root)}:{node.lineno} {node.name}"
        for path in root.rglob("*.py")
        if "models" not in path.relative_to(root).parts
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("check_")
    ]


def converted_arguments(package, *model_modules) -> list[str]:
    """The calls in `package` that convert one of the caller's own arguments on the way into a model of
    `model_modules`: `Model(tuple(arg))` or `Model(np.asarray(arg))` turns a malformed argument into a
    raw TypeError or ValueError before the model can refuse it; the model's rules do the conversion."""
    src = pathlib.Path(package.__file__).parent
    names = {
        n
        for m in model_modules
        for n, c in vars(m).items()
        if inspect.isclass(c) and issubclass(c, Validated) and c is not Validated
    }
    conversions = {"tuple", "list", "float", "int", "np.asarray", "np.array", "np.ascontiguousarray"}
    hits = []
    for path in sorted(src.rglob("*.py")):
        if "models" in path.parts:
            continue
        for fn in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            for node in ast.walk(fn):
                if not (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names
                ):
                    continue
                for arg in list(node.args) + [k.value for k in node.keywords]:
                    for c in ast.walk(arg):
                        if (
                            isinstance(c, ast.Call)
                            and ast.unparse(c.func) in conversions
                            and c.args
                            and isinstance(c.args[0], ast.Name)
                            and c.args[0].id in params
                        ):
                            hits.append(f"{path.relative_to(src)}:{node.lineno} {ast.unparse(c)}")
    return hits
