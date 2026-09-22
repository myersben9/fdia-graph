"""Stage the three root pages of the repository for the docs site: README.md becomes
docs/index.md, CONTRIBUTING.md docs/contributing.md and CHANGELOG.md docs/changelog.md. The
README is the single source for the landing page; its links carry a docs/ prefix that is wrong
once the file lives inside docs_dir, so the prefix is stripped on copy, and links between the
three root pages are rewritten to their staged names. Runs in the Vercel build (vercel.json) and
in the versioned deploy (.github/workflows/docs.yml); the three staged files are generated output
and stay untracked.
"""

import pathlib
import re

root = pathlib.Path(__file__).resolve().parents[1]
docs = root / "docs"
ROOT_PAGES = {"README.md": "index.md", "CONTRIBUTING.md": "contributing.md", "CHANGELOG.md": "changelog.md"}
for source, staged in ROOT_PAGES.items():
    text = (root / source).read_text(encoding="utf-8")
    text = re.sub(r"\]\(docs/", "](", text)
    for other, other_staged in ROOT_PAGES.items():
        text = re.sub(r"\]\((?:\./)?" + re.escape(other) + r"(#[^)]*)?\)", r"](" + other_staged + r"\1)", text)
    (docs / staged).write_text(text, encoding="utf-8")
    print(f"wrote docs/{staged}")
