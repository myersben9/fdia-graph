# Contributing

How a change gets into fdia-graph. Other groups load this package in their experiments, so the
rules below exist to keep their numbers stable; every one of them is enforced by a command you
can run yourself.

## Setup

```bash
git clone https://github.com/myersben9/fdia-graph
cd fdia-graph
pip install -e ".[generate,se,torch,dev]"     # generation needs pandapower; the tests build a tiny shard
pytest tests                                  # about a minute; downloads a 13 MB operating-point pool once
```

`docs/ROADMAP.md` is the map of the package and the reading order for a new student.

## The rules

1. **Nothing a user can see changes without a changelog entry.** Public API, file formats, and
   the numbers a shard, a stream or an estimator produces. Internal names that move get an alias
   that warns for one minor version, and our own callers switch in the same pull request.
2. **Every change is proven behaviour-free, or its effect is measured.** The strict frozen suite
   compares a tiny shard, a stream and the estimator and localizer scores bit for bit against
   `tests/frozen/`. Run it before every push:

   ```bash
   FDIA_FROZEN_STRICT=1 pytest -W "error::DeprecationWarning:fdia_graph" tests
   ```

   If a change is meant to move numbers (a new attack, a different solver), regenerate the
   references with `python tools/freeze_reference.py` in the same pull request and state the
   deltas in its description. CI runs the same tests with a small cross-platform tolerance.
3. **Code reads as its subject.** Functions stay within the readability limits (complexity 10,
   nesting 3, no closure captures, at most 7 parameters, no positional record indexing); every
   equation from a paper or textbook is a named function in `fdia_graph.formulas` with its source
   key from `docs/reference/REFERENCES.md`; every value bundle a function returns is a model in
   `fdia_graph.models`. `python tools/readability.py --report` shows the state of the package,
   `--check --base origin/main` is the gate CI runs on what you touched.
4. **Typed and formatted.** `pyright src/fdia_graph` reports zero errors, `ruff format --check src`
   passes; both run in CI.

## The pull request

Work on a branch, never on main. When the strict suite, pyright and the gate pass locally:

```bash
python tools/pr.py create my-branch "One-line title" body.md   # body: what, why, what you checked
python tools/pr.py wait 80                                     # CI plus the Copilot review
python tools/pr.py comments 80                                 # read the review
python tools/pr.py reply 80 <comment-id> "what changed"        # answer each comment with what you did
python tools/pr.py merge 80                                    # squash on green, deletes the branch
```

Copilot reviews every push automatically. Its comments are suggestions, not orders: apply the
ones that are right (in this repo about one in two has been), answer every one with what you
changed or why not, and merge only when CI is green and the review has landed on the final head.
`tools/pr.py` uses the token the Git Credential Manager already holds for `git push`; there is
no gh CLI on the lab machines.

Commit as yourself, with a message that says what the change does.

## Releasing

A release is a pull request like any other: bump `version` in `pyproject.toml` and
`__version__` in `src/fdia_graph/__init__.py`, turn the changelog's `## Unreleased` heading into
the version with a short paragraph on what a user sees, and merge. Then, on a clean main at
`origin/main`:

```bash
python tools/release.py v0.18.0 notes.md
```

It tags the merged commit, pushes the tag, creates the GitHub release with the notes, and waits
for `publish.yml` to build and upload to PyPI (Trusted Publishing, no token). A released version
is never re-cut; fix forward with a new version. Data releases (shards, pools, streams) carry
their own tags and do not move the package version; see the release notes of `v0.7.2`.

## Where things live

| you want to change | look in |
|---|---|
| an attack family, the plausibility band, the meter model | `engine/` (the generator), `formulas/attacks.py`, `formulas/noise.py` |
| what a shard or stream contains | `generation.py`, `streams.py`, then `docs/reference/DATA_DICTIONARY.md` |
| how a shard is loaded or exported | `dataset/` (one concern per file) |
| an estimator | `se/methods.py`, with its algebra in `formulas/estimation.py` |
| a localizer | `localization/methods.py` or `learned.py` |
| a returned record's fields | `models/` |
| a formula | `formulas/`, and its row in `docs/reference/FORMULAS.md` |

The plans that shaped the current layout are archived under `docs/plans/`; they explain why the
code looks the way it does, not how to change it.
