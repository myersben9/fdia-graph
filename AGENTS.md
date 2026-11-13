# Instructions for AI agents

AI coding tools working in this repository load this file. `CLAUDE.md` imports it for Claude Code.
People read `CONTRIBUTING.md`, which is the source of every rule here. If the two disagree,
`CONTRIBUTING.md` wins and this file needs fixing.

Other groups load this package in their experiments. A change that quietly moves a generated number
breaks their results, so the rules keep the numbers stable and the code readable.

## Before you write

- Read "The rules" and "Where things live" in `CONTRIBUTING.md`, then `docs/ROADMAP.md`.
- Read every file you will touch from top to bottom, not only the lines you change.
- Search for the formula, model or helper that already exists before you add one.

## Do this, not that

| Don't | Do this instead |
|---|---|
| check an argument or `raise ValueError` inside a function body | declare the argument on its model in `src/fdia_graph/models/config.py` with its rule in the annotation. For what only the data reveals, raise a named error from `src/fdia_graph/errors.py` |
| write an equation inline | a named function in `src/fdia_graph/formulas/` with its source key, cited like `[WU26 eq. 28]` |
| index a record by position, or spell an HDF5 path | `NODE`, `EDGE` and `BRANCH` from `src/fdia_graph/models/grid.py`, and the names in `src/fdia_graph/schema.py` |
| type a measured number into Markdown | a results block written by `tools/results_docs.py` (see `results/README.md`) |
| edit a generated file | edit its source and rerun the tool: `tools/class_diagrams.py`, `tools/models_doc.py`, `tools/equation_map.py` |
| add a Mermaid block to a page | a rendered image from `tools/render_mermaid.py`, with the `.mmd`, `.png` and `.svg` committed together |
| add a `type: ignore` comment, a broad `except`, or `Any` | fix the type. `Any` is allowed only where `ANY_ALLOWED` in `tools/readability.py` lists it |
| weaken an assertion, skip a test, or change a pinned answer (`tests/test_search_speed.py`, `tests/test_wu_scenarios.py`) to get green | stop and report the failure with its output |
| change the order of random draws without saying so | say it in `CHANGELOG.md`. Every released file must reproduce bit for bit from its seed |
| add helpers, options or handling nobody asked for | the smallest change that does the job. Code that works but is more than the job needs gets sent back |

## Decisions belong to the owner

Some choices are not yours to make. Stop and ask before you write any of these:

- a new or renamed public name, a new default, or a new dependency
- a change to a file format or to a number a timeline, estimator or localizer produces
- a reading of [WU26] the paper leaves open (these are recorded in `schema.DECISIONS`, never only
  in a comment)

For every smaller choice you did make, add a line under a "Decisions" heading in the pull request
body: what you chose and why. The reviewer reads that list first and can overrule any line.

## Before a pull request

1. `python tools/prereview.py` until it passes.
2. Work through `docs/reference/REVIEW_CHECKLIST.md` against the diff and every touched file.
3. Open the pull request as a draft with `python tools/pr.py create`. The body says what changed,
   why, what you checked, and the decisions you made.

Never push to `main`, never force a push onto it, and never commit with `--no-verify`. Do not mark a
pull request ready and do not merge one unless the person you are working for tells you to: marking
it ready spends a billed review. Commit as that person, with a message that says what the change
does.
