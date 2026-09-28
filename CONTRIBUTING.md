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
the code (CI fails when they are stale), then render them the same way. README images use the
raw.githubusercontent.com URL so PyPI shows them; docs pages use relative paths.

The Models section of `docs/reference/DATA_DICTIONARY.md` is generated too: `python tools/models_doc.py`
rewrites it from the dataclasses in `src/fdia_graph/models/` and the comment beside each field. Edit
the comment in the source, never the section, and rerun the tool; the tests fail when it is stale.

The docs site is built from `docs/` plus the three root pages by `mkdocs.yml`: the Vercel build serves
the main branch at the custom domain, and `.github/workflows/docs.yml` deploys one copy per package
version to GitHub Pages with mike (a tag push deploys that version as `latest`, a push to main
deploys `dev`), so every release's docs stay readable after the next release; the version picker on
each page switches between them. `python scripts/site_index.py && python -m mkdocs build --strict`
checks a docs change locally.

## The rules

| rule | means | command |
|---|---|---|
| nothing a user sees changes without a changelog entry | public API, file formats, the numbers a timeline, estimator or localizer produces; moved private names get a warning alias for one minor version | `CHANGELOG.md`, `## Unreleased` |
| every change is proven behaviour-free, or its effect is measured | the strict frozen suite compares a tiny timeline and its state-estimation and localization scores bit for bit; an intended change re-freezes in the same PR and states the deltas | `FDIA_FROZEN_STRICT=1 pytest -W "error::DeprecationWarning:fdia_graph" tests`, `python tools/freeze_reference.py` |
| code reads as its subject | complexity 10, nesting 3, no closure captures, at most 7 parameters, no positional record indexing; every equation a named function in `formulas/` with a source key; every returned bundle a model in `models/` | `python tools/readability.py --report`, `--check --base origin/main` |
| an input is checked in one place | a new argument is a field on its consumer's model in `models/config.py`, with its rules in the annotation (`Annotated[float, Positive()]`, `OneOf(<Choice>)`) and cross-field rules in `invariants()`; a dataset precondition is a `ds.require(...)` capability; a condition only the data reveals raises a named error from `errors.py`; never a `raise ValueError` in a function body | `python tools/readability.py --check` (refuses a new one), `docs/plans/VALIDATION_PLAN.md` |
| everything is typed | every parameter, return and field carries its real type; an optional dependency's type (pandapower, torch, h5py, scipy, pandas) is imported under `if TYPE_CHECKING:` so the import stays lazy; `Any` only where the value can be anything, listed in `ANY_ALLOWED` in `tools/readability.py` with the reason | `python tools/readability.py --check` (refuses an unlisted `Any` and a stale entry), pyright in CI |
| typed, formatted, linted | pyright at zero, ruff format and check clean | `pyright src/fdia_graph`, `ruff format --check src`, `ruff check src tests tools` |
| generated docs match the code | the class and module diagram sources and the data dictionary's Models section | `python tools/class_diagrams.py --check`, `python tools/models_doc.py --check` |
| speed is tracked | per-record timings against the last row from this machine, 3x tolerance | `python tools/bench.py --check` before a release, `python tools/bench.py` after |

## The pull request

![the pull request flow: branch, the pre-review gate and the checklist review, create, request the review, fix every finding in one push and go through the gate again, merge on green](docs/figures/diagrams/contributing_pr_flow.png)

```bash
python tools/prereview.py                                      # every gate, locally, on this checkout's source
python tools/prereview.py --also-python <path-to-python3.12>  # and the suite on a second Python, as CI runs 3.9 and 3.12
python tools/pr.py create my-branch "One-line title" body.md   # body: what, why, what you checked
python tools/pr.py request-review 80                           # only after the gate and the checklist are clean
python tools/pr.py wait 80                                     # CI plus every required review bot
python tools/pr.py comments 80
python tools/pr.py reply 80 <comment-id> "what changed"
python tools/pr.py merge 80                                    # refuses unless green with every required bot's review on the head
python tools/review_ledger.py 80                               # record the review's findings, print the tally by kind
```

Green means: every job of the smoke workflow (`tests`, `tests (3.9)`, `tests (windows)`, `typecheck`,
`format`, `readability`, the two installs) is a completed success on the head (a job that has not
started yet counts as not green), every other listed check has finished without failure, and every required review bot has reviewed
that exact head.

The automated review is billed per review, so a pull request reaches it only once it is clean, and
the fixes to its findings are batched into one push. The order:

1. `python tools/prereview.py` until it passes: the CI gates, the strict suite, and the checks CI
   does not run (changelog, cited paths, vacuous tests, integer fields, rendered diagrams).
2. The review checklist, `docs/reference/REVIEW_CHECKLIST.md`, against the diff and every touched
   file read whole, searching outward from every change (a `/code-review` in Claude Code with the
   checklist does this). Fold what it finds into the branch.
3. Copilot, requested with `tools/pr.py request-review`, required on the head at merge. Answer its
   findings, fix them all in one push, repeat steps 1 and 2, and only then request it again. Record
   each review with `tools/review_ledger.py`; a kind of finding that keeps coming up becomes a
   checklist item or a check in `tools/prereview.py`.
4. CodeRabbit (`.coderabbit.yaml` carries the repository's review instructions) and Gemini Code Assist, both
   GitHub apps installed on the repository; `tools/pr.py` requires a bot's review on the head as
   soon as that bot has reviewed the pull request once, so a bot that is not installed never
   blocks a merge and an installed one is never skipped.

Bot comments are suggestions: apply the ones that are right and answer every one with what changed
or why not. `tools/pr.py` uses the token the Git Credential Manager holds for `git push`, so it
needs no gh CLI. Commit as yourself, with a message that says what the change does. If CI does not
start on a push, check the account's Actions and Copilot credits first.

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
| what a timeline file contains | `timeline.py` (the walk and the file), `generation.py` (the pool and `generate`), then `docs/reference/DATA_DICTIONARY.md` |
| how a timeline is loaded or exported | `dataset/`, one concern per file |
| an estimator | `se/methods.py`, its algebra in `formulas/estimation.py` |
| a localizer | `localization/methods.py` or `learned.py` |
| the trusted-meter selectors | `trust/`, its algebra in `formulas/trust.py` |
| the federated localizers and prior | `federated/`, its algebra in `formulas/federated.py` |
| a returned record's fields | `models/` |
| a formula | `formulas/`, and its row in `docs/reference/FORMULAS.md` |

The design documents behind the current layout are archived in `docs/plans/`.
