"""Render the public data models into docs/reference/DATA_DICTIONARY.md.

    python tools/models_doc.py          # rewrite the section between the markers
    python tools/models_doc.py --check  # exit 1 when the file is out of date (the tests run this)

The section lists every public bundle (`fdia_graph.models.PUBLIC`, docs/plans/DATA_MODELS_PLAN.md
steps 4 and 5): its module, what
produces it, and one row per field with the dict key, the type and the comment written next to
the field in the source. It is generated from the dataclasses, so it cannot drift from the code.
"""

from __future__ import annotations

import inspect
import os
import re
import sys
from dataclasses import fields
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(os.path.dirname(HERE), "docs", "reference", "DATA_DICTIONARY.md")
BEGIN, END = "<!-- models:begin -->", "<!-- models:end -->"

# The public bundles are named by fdia_graph.models.PUBLIC, in the order a reader meets them.

_FIELD_LINE = re.compile(r"^\s*(\w+)\s*:\s*(.+?)(?:\s*=\s*[^#]+?)?\s*(?:#\s*(.*))?$")


def _field_comments(cls: type[Any]) -> dict:
    """The comment written after each field in the class body, by field name, gathered over the
    class and its field groups (the nearest definition wins)."""
    out: dict = {}
    for klass in reversed(cls.__mro__):
        if klass is object or not hasattr(klass, "__dataclass_fields__"):
            continue
        try:
            src = inspect.getsource(klass)
        except (OSError, TypeError):
            continue
        for line in src.splitlines():
            m = _FIELD_LINE.match(line)
            if m and not line.lstrip().startswith(("_", "@", "class ", '"""')):
                out[m.group(1)] = (m.group(3) or "").strip()
    return out


def _group_of(cls: type[Any], name: str) -> str:
    """The field group that declares `name`, or "" when the bundle declares it itself."""
    for klass in cls.__mro__[1:]:
        if name in klass.__dict__.get("__dataclass_fields__", {}) and klass.__name__ != "Bundle":
            own = klass.__dict__.get("__annotations__", {})
            if name in own:
                return klass.__name__
    return ""


def _type_name(f: Any) -> str:
    t = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
    t = re.sub(r"^Optional\[(.*)\]$", lambda m: m.group(1), t)  # the table marks optional fields itself
    return t.replace("np.ndarray", "array")


def _render_model(cls: type[Any]) -> list[str]:
    doc = inspect.getdoc(cls) or ""
    comments = _field_comments(cls)
    lines = [f"### `{cls.__name__}` (`{cls.__module__}`)", "", " ".join(doc.split()), ""]
    groups = [
        k.__name__ for k in cls.__mro__[1:] if "__dataclass_fields__" in k.__dict__ and k.__name__ != "Bundle"
    ]
    if groups:
        lines += ["Field groups: " + ", ".join(f"`{g}`" for g in groups) + ".", ""]
    lines += ["| field | dict key | type | required | meaning |", "|---|---|---|---|---|"]
    by_name = {f.name: f for f in fields(cls)}
    for name in cls._names():
        f = by_name[name]
        key = cls.key_of(f.name)
        required = f.name in getattr(cls, "_required", ()) or (
            f.default is not None and "Optional" not in str(f.type)
        )
        meaning = comments.get(f.name, "").replace("|", "&#124;")  # a bare pipe would split the table cell
        group = _group_of(cls, f.name)
        src = f" ({group})" if group else ""
        lines.append(
            f"| `{f.name}` | `{key}` | {_type_name(f)} | {'yes' if required else ''} | {meaning}{src} |"
        )
    return lines + [""]


def render() -> str:
    import importlib

    lines = [
        BEGIN,
        "## Models",
        "",
        "Every dict the package returns is a typed bundle (`fdia_graph.models.Bundle`): a frozen",
        'dataclass that is also the `dict` it always was, so `out["node_x"]`, `**out` and iteration',
        "keep working and `out.node_x` is new. Fields set to None are absent from the dict. Generated",
        "by `tools/models_doc.py` from the dataclasses.",
        "",
    ]
    models = importlib.import_module("fdia_graph.models")
    for name in models.PUBLIC:
        lines += _render_model(getattr(models, name))
    return "\n".join(lines) + END + "\n"


def main(check: bool) -> int:
    text = open(DOC, encoding="utf8").read()
    a, b = text.find(BEGIN), text.find(END)
    new = render()
    updated = (text[:a] + new + text[b + len(END) + 1 :]) if a >= 0 else text.rstrip("\n") + "\n\n" + new
    if check:
        if updated != text:
            print("DATA_DICTIONARY.md models section is out of date: run python tools/models_doc.py")
            return 1
        print("models section up to date")
        return 0
    open(DOC, "w", encoding="utf8", newline="\n").write(updated)
    print("wrote", DOC)
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
