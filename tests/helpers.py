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


def ramp_frames_past_the_bound(g, X, path, scale=1.0):
    """How many attacked ramp frames of the timeline at `path`, walked by the generator `g` over the
    pool `X`, hold a metered channel whose attack steps from the frame before by more than its meter's
    sigma at the true reading times `scale`: the ramp's stealth bound, read back from the file. The
    step out counts too: an attacked frame followed by a frame the ramp does not attack must hold its
    whole attack inside the bound. The stored readings are float32, so a step counts only beyond four
    of their spacings."""
    import h5py

    from fdia_graph import schema
    from fdia_graph.constants.channels import NodeChannel
    from fdia_graph.constants.families import AT_CODE

    masks = [np.array(m) > 0 for m in (g.masks.node, g.masks.edge, g.masks.current)]
    pmu = np.zeros(g.n_bus, bool)
    pmu[g.pmu_buses] = True
    masks[0][~pmu, NodeChannel.THETA] = False  # a bus angle is a PMU channel
    pairs = (
        (schema.NODE_X, schema.NODE_BENIGN),
        (schema.EDGE_X, schema.EDGE_BENIGN),
        (schema.PMU_I, schema.PMU_I_BENIGN),
    )
    past = 0
    with h5py.File(path, "r") as f:
        episodes, seq = f[schema.Group.EPISODES], f[schema.SEQ_ID][:]
        for k in np.flatnonzero(episodes[schema.EPISODE_FAMILY][:] == AT_CODE):
            onset, length = int(episodes[schema.EPISODE_ONSET][k]), int(episodes[schema.EPISODE_LENGTH][k])
            first = max(onset - 1, 0)
            read = [(f[x][first : onset + length], f[b][first : onset + length]) for x, b in pairs]
            attack = [x.astype(float) - b.astype(float) for x, b in read]
            spacing = [4 * np.spacing(np.abs(x).astype(np.float32)).astype(float) for x, _ in read]
            for t in np.flatnonzero(seq[onset : onset + length] == k) + onset:
                i = t - first
                x = np.asarray(X[t], np.float64)
                flows = g.emitter.clean_flows_from_states(x[None])[0]
                currents = g.emitter.currents_from_states(x[None])[0]
                sigma = g.emitter.layers(g.noise.sigmas(g.plan.gather(x, flows, currents)))
                leaves = t + 1 >= onset + length or seq[t + 1] != k  # the next frame is not this ramp's
                steps = [
                    (np.abs(a[i] - (a[i - 1] if i else 0.0)) > scale * s + e[i])
                    | (leaves & (np.abs(a[i]) > scale * s + e[i]))
                    for a, s, e in zip(attack, sigma, spacing)
                ]
                past += any(bool((step & m).any()) for step, m in zip(steps, masks))
    return past


def feasible_ramp(g, X, t, knobs):
    """A ramp drawn at t with a false state on the area at every frame of its window, whatever the
    stealth bound says of it (`ramp_design` keeps only those that hold the bound); None when 40 draws
    give none. The draws come from the generator's own stream."""
    from fdia_graph.engine.attacks.episodes import probe_frames

    designs = g.episode_designs
    for _ in range(40):
        d = designs.draw_ramp(g.rng, WINDOW, knobs)
        steps = ((u, designs.ramp_step(d, u - t, 0.002)) for u in probe_frames(len(X), t, WINDOW))
        if all(g.false_state.is_feasible(X[u], step, knobs) for u, step in steps):
            return d
    return None


def windows(case, n, held=True):
    """n ramp windows with a false state on the area at every frame, each with its goal (one design
    per snapshot) from the ramp's first frame; with `held`, only windows where some support held for
    the window keeps the stealth bound and moves a device beyond noise (the bound rejects most: a
    reactive flow near zero has a noise floor of 1e-3). `case` is the fixture of that name."""
    from fdia_graph.engine.attacks.episodes import probe_frames
    from fdia_graph.models.frames import LoadGoal

    g, X, knobs = case
    rng = np.random.default_rng(3)
    out = []
    while len(out) < n:
        t = int(rng.integers(0, len(X) - WINDOW))
        d = feasible_ramp(g, X, t, knobs)
        if d is None:
            continue
        frames = probe_frames(len(X), t, WINDOW)
        goal = LoadGoal(tuple(g.episode_designs.ramp_step(d, u - t, 0.002) for u in frames))
        states = [X[u] for u in frames]
        if held:
            found = g.minimizer.min_tamper(states, goal, knobs)
            if found is None or found.devices == 0:
                continue  # no held support, or nothing moves above noise: nothing for the search to weigh
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
