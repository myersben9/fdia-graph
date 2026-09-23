# Contributing

Other groups load this package in their experiments, so the rules keep their numbers stable. Every
rule is a command you can run.

## Setup

```bash
git clone https://github.com/myersben9/fdia-graph
cd fdia-graph
pip install -e ".[generate,se,torch,dev]"
pytest tests                                  # about a minute; one 13 MB pool download
```

`docs/ROADMAP.md` is the map and the reading order.

Every diagram is a rendered image, never a Mermaid block: the README is also the PyPI page, and the
docs are read in the GitHub mobile app, and neither renders Mermaid. Each diagram's source is a
`docs/figures/diagrams/<name>.mmd` next to its `<name>.png` and `<name>.svg`; edit the source and
re-render with `python tools/render_mermaid.py docs/figures/diagrams/<name>.mmd` (Playwright with
Chromium), which writes both images, then commit all three. The class and module diagrams of
`docs/reference/CLASS_MAP.md` are generated: `python tools/class_diagrams.py` rewrites their sources from
the code (CI fails when they are stale), then render them the same way. README images use the raw.githubusercontent.com URL so PyPI shows them; docs pages use
relative paths.

The docs site is built from `docs/` plus the three root pages by `mkdocs.yml`: the Vercel build serves
the main branch at the custom domain, and `.github/workflows/docs.yml` deploys one copy per package
version to GitHub Pages with mike (a tag push deploys that version as `latest`, a push to main
deploys `dev`), so every release's docs stay readable after the next release; the version picker on
each page switches between them. `python scripts/site_index.py && python -m mkdocs build --strict`
checks a docs change locally.

## The rules

| rule | means | command |
|---|---|---|
| nothing a user sees changes without a changelog entry | public API, file formats, the numbers a shard, stream or estimator produces; moved private names get a warning alias for one minor version | `CHANGELOG.md`, `## Unreleased` |
| every change is proven behaviour-free, or its effect is measured | the strict frozen suite compares a tiny shard, a stream and the scores bit for bit; an intended change re-freezes in the same PR and states the deltas | `FDIA_FROZEN_STRICT=1 pytest -W "error::DeprecationWarning:fdia_graph" tests`, `python tools/freeze_reference.py` |
| code reads as its subject | complexity 10, nesting 3, no closure captures, at most 7 parameters, no positional record indexing; every equation a named function in `formulas/` with a source key; every returned bundle a model in `models/` | `python tools/readability.py --report`, `--check --base origin/main` |
| typed, formatted, linted | pyright at zero, ruff format and check clean | `pyright src/fdia_graph`, `ruff format --check src`, `ruff check src tests tools` |
| speed is tracked | per-record timings against the last row from this machine, 3x tolerance | `python tools/bench.py --check` before a release, `python tools/bench.py` after |

## The pull request

![the pull request flow: branch, strict suite, create, CI and the review bots, answer every comment, merge on green](docs/figures/diagrams/contributing_pr_flow.png)

```bash
python tools/pr.py create my-branch "One-line title" body.md   # body: what, why, what you checked
python tools/pr.py wait 80                                     # CI plus every required review bot
python tools/pr.py comments 80
python tools/pr.py reply 80 <comment-id> "what changed"
python tools/pr.py merge 80                                    # refuses unless green with every required bot's review on the head
```

Green means: every job of the smoke workflow (`tests`, `tests (3.9)`, `tests (windows)`, `typecheck`,
`format`, `readability`, the two installs) is a completed success on the head, a job that has not started yet counts as not green,
every other listed check has finished without failure, and every required review bot has reviewed
that exact head.

Three reviewers read a pull request, in this order:

1. `/code-review ultra <number>` in Claude Code, before the bots: a multi-agent review of the whole
   branch with the repository as context. Fold what it finds into the branch first.
2. Copilot, on every push, always required on the head.
3. CodeRabbit (`.coderabbit.yaml` carries our review instructions) and Gemini Code Assist, both
   GitHub apps installed on the repository; `tools/pr.py` requires a bot's review on the head as
   soon as that bot has reviewed the pull request once, so a bot that is not installed never
   blocks a merge and an installed one is never skipped.

Bot comments are suggestions: apply the ones that are right (about one in two has been), answer
every one with what you changed or why not. The tool uses the token the
Git Credential Manager already holds for `git push`; there is no gh CLI on the lab machines. Commit
as yourself, with a message that says what the change does. If CI does not start on a push, check the
account's Actions and Copilot credits before anything else.

## Releasing

![the release flow: bump PR, merge on green, tag on main, GitHub release, publish.yml to PyPI](docs/figures/diagrams/contributing_release_flow.png)

A released version is never re-cut; fix forward. Data releases (timelines, pools) have their own
tags, `data-v<x.y.z>` from v0.8.0 (the bare `v0.x.y` tags belong to package versions; the two
earlier data releases keep `v0.7.1` and `v0.7.2`), and do not move the package version. The
registry maps the short name (`fg.load(..., release="v0.8.0")`) to the tag, and
`tools/upload_assets.py v0.8.0 <files>` refuses to touch a package tag.

## Where things live

| you want to change | look in |
|---|---|
| an attack family, the plausibility band, the meter model | `engine/`, `formulas/attacks.py`, `formulas/noise.py` |
| what a shard or stream contains | `generation.py`, `streams.py`, then `docs/reference/DATA_DICTIONARY.md` |
| how a shard is loaded or exported | `dataset/`, one concern per file |
| an estimator | `se/methods.py`, its algebra in `formulas/estimation.py` |
| a localizer | `localization/methods.py` or `learned.py` |
| a returned record's fields | `models/` |
| a formula | `formulas/`, and its row in `docs/reference/FORMULAS.md` |

The design documents behind the current layout are archived in `docs/plans/`.
