"""Fill and check the results blocks of the docs from the results store (`results/`).

    python tools/results_docs.py --write    # re-render every block in place
    python tools/results_docs.py --check    # gate: fail if any block differs from what it renders
    python tools/results_docs.py --lint     # list decimals and percentages typed outside a block

A measured number in markdown sits in a block, `<!-- results: <query> <args> -->...<!-- /results -->`;
the queries are registered in `results/queries.py`. `--check` runs in the pre-review and in CI, as
`class_diagrams.py --check` does, so a doc can never drift from the store it cites. `--lint` only
reports: a setting such as "0.03 pu" or "k = 1.2" is a legitimate number outside a block.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fdia_graph.results import Store, fill, stale  # noqa: E402
from fdia_graph.results.docs import BLOCK  # noqa: E402

# the files whose numbers the policy covers: released CHANGELOG sections are history and stay as written
SKIP = ("CHANGELOG.md",)
_NUMBER = re.compile(r"(?<![\w.\-/#`])\d+\.\d+%?|(?<![\w.\-/#`])\d+%")
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)


def load_queries() -> None:
    spec = importlib.util.spec_from_file_location(
        "results_queries", os.path.join(ROOT, "results", "queries.py")
    )
    assert spec and spec.loader
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def markdown_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [f for f in out.split() if os.path.basename(f) not in SKIP]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--lint", action="store_true")
    ap.add_argument("files", nargs="*", help="markdown files (default: every tracked .md)")
    a = ap.parse_args()
    load_queries()
    store = Store(os.path.join(ROOT, "results"))
    bad = 0
    for rel in a.files or markdown_files():
        path = os.path.join(ROOT, rel)
        text = open(path, encoding="utf-8").read()
        if a.lint:
            bare = _NUMBER.findall(BLOCK.sub("", _CODE.sub("", text)))
            if bare:
                print(f"{rel}: {len(bare)} number(s) outside a results block: {' '.join(bare[:12])}")
            continue
        if "<!-- results:" not in text:
            continue
        if a.write:
            new = fill(text, store)
            if new != text:
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(new)
                print(f"filled {rel}")
            continue
        for spec in stale(text, store):
            print(f"{rel}: stale results block `{spec}`; run tools/results_docs.py --write")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
