"""Generate docs/index.md from README.md for the mkdocs site.

The README is the single source for the landing page; its links carry a
docs/ prefix that is wrong once the file lives inside docs_dir, so the
prefix is stripped on copy. Runs in the Vercel build (see vercel.json);
docs/index.md is generated output and stays untracked.
"""
import pathlib
import re

root = pathlib.Path(__file__).resolve().parents[1]
text = (root / "README.md").read_text(encoding="utf-8")
text = re.sub(r"\]\(docs/", "](", text)
(root / "docs" / "index.md").write_text(text, encoding="utf-8")
print("wrote docs/index.md")
