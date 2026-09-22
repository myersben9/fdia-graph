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


def prune_nav() -> None:
    """Drop nav entries whose page does not exist in this checkout. The versioned deploy builds an
    older tag with the current mkdocs.yml, whose nav names pages that came later; without this a
    reader of that version would click into a 404."""
    import yaml

    path = root / "mkdocs.yml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    dropped = []

    def keep(item):
        if isinstance(item, str):
            present = (docs / item).exists()
            dropped.extend([] if present else [item])
            return present
        title, target = next(iter(item.items()))
        if isinstance(target, list):
            pruned = [child for child in target if keep(child)]
            item[title] = pruned
            return bool(pruned)
        present = (docs / target).exists()
        dropped.extend([] if present else [target])
        return present

    config["nav"] = [item for item in config.get("nav", []) if keep(item)]
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("nav: pages this checkout lacks, dropped: " + (", ".join(dropped) or "none"))


prune_nav()
