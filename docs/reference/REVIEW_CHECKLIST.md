# Review checklist

Run this before a pull request is sent for review. It is built from the 313 findings of the
automated reviewer on 80 pull requests (#52 to #148), grouped by kind and ordered by how often each
kind came up. `python tools/prereview.py` runs every mechanical check. This page covers what only a
reader can check.

A third of all findings were one failure mode: code changed, but a doc, a docstring, a changelog line,
a number or a generated file describing it did not. Most of those findings pointed at a file the pull
request never touched. So the review starts from the diff and searches outward through the whole
repository, not just the changed lines.

The reviewer also reads every file a pull request touches from top to bottom, not only the changed
lines: on #140 to #152, 27 of its 78 findings (35%) were in lines the pull request did not change,
and each one cost a further review round. So the local review reads each touched file whole too.

`python tools/review_ledger.py <num>` adds a pull request's findings to
`docs/reference/review_ledger.csv` and prints the tally by kind. Run it after every automated review.
A kind that keeps coming up becomes a checklist item or a mechanical check.

## Since #140 (the ledger, as of #152)

| kind | findings | pull requests | what now catches it |
|---|---:|---:|---|
| input escapes validation (a raw `TypeError` or `KeyError`, an argument used before its model, a missing `Integer()`, a negative count) | 25 | 5 | the integer-field check; the malformed-input test that arrives with #147; item 4 |
| doc claim does not match the code (a release, a file, a count, an unmerged pull request, an uncited outside claim) | 21 | 5 | the cited-path check; item 15 |
| logic and edge cases | 10 | 5 | item 3 |
| cache and resume | 7 | 1 | item 10 |
| generated file stale | 5 | 4 | the class-diagram, data-dictionary and rendered-diagram checks |
| tooling | 4 | 2 | item 9 |
| compatibility | 3 | 3 | item 8 |
| export missing | 2 | 2 | item 8 |
| performance | 1 | 1 | item 11 |

## What came up, and how often

| kind | findings | pull requests |
|---|---:|---:|
| docs, docstrings or changelog out of step with the code | 77 | 38 |
| logic and edge cases (degenerate inputs, exhausted retries, stale state, a change only partly delivered) | 44 | 20 |
| input validation (integers, NaN and inf, shapes, unknown options, a raw TypeError, checking after a download) | 37 | 14 |
| tooling and CI (workflow inputs in shell, pagination, exit codes) | 33 | 12 |
| tests missing, vacuous or testing the wrong thing | 23 | 18 |
| cache, resume and reproducibility keys | 16 | 7 |
| numbers in docs that do not match the results files | 15 | 8 |
| performance and memory | 13 | 10 |
| generated files stale (diagram images, data dictionary, frozen fixture) | 12 | 9 |
| backward compatibility (reordered parameters, removed or renamed names) | 11 | 11 |
| readability contract | 10 | 6 |
| optional dependencies and extras | 7 | 5 |
| grammar and markdown | 6 | 6 |
| loopholes in the readability gate | 5 | 3 |
| type hints | 4 | 4 |

## The checklist

1. **Sweep for drift.** For every changed name, setting, default, path or behaviour, search the rest
   of the repository for the old wording and fix every hit. That covers docstrings in the same
   module, `docs/`, `README.md`, `CONTRIBUTING.md`, the tool docstrings and older bullets in the same
   `## Unreleased` section. For example: `rg -n "<old name>|<old value>|<old path>" docs src tools
   tests README.md CHANGELOG.md CONTRIBUTING.md`.
2. **Every claim holds end to end.** When the change says "no longer needs X", "unchanged", "exact"
   or "bounded memory", trace every call path (fit, score, estimate and their helpers) and confirm
   it. "Unchanged numbers" means the strict frozen suite passes.
3. **Degenerate inputs.** Check 0, 1, a fraction that rounds to 0, an empty candidate list, every
   retry failing, a validation set with no attacked (or no benign) record, and a refit after tuning.
   A retry loop must never emit the last candidate it rejected.
4. **Every new input is declared on a model.**
   - An integer field carries `Integer()`; `AtLeast` alone accepts 1.5.
   - A float that must be finite carries `Finite()`; NaN and inf pass a `> 0` check.
   - An option is a `OneOf(<Choice>)`.
   - Arrays have their shape stated.
   - Paired lists have a length invariant.
   - Nothing is checked in a function body, and everything is checked before any download or file
     read.
5. **Tests prove the claim.**
   - No `or True` and no assertion that holds by construction.
   - Each new fast path and each feature branch gets its own test.
   - A test's name matches what it loads.
   - A test that reads a published data file runs only under `FDIA_SLOW=1` (the slow workflow),
     never in the default suite.
6. **Numbers in docs are recomputed** from the committed results file: the right metric (macro-F1
   versus node-F1), the right denominator, the right rounding, and no placeholders left.
7. **Generated files are regenerated and committed together.** That means the diagram `.mmd`, `.svg`
   and `.png`, the data dictionary, and the frozen fixture after a change to generation.
8. **Backward compatibility.**
   - A new parameter goes last or is keyword-only.
   - A removed or renamed public name keeps a deprecated alias for one minor version.
   - Dataclass field order does not change.
   - New public names are exported.
9. **Tooling and CI.**
   - Workflow inputs reach a shell through `env:`, never through `${{ }}` inside `run:`.
   - List endpoints are paginated.
   - HTTP errors other than 404 surface.
   - A failure exits non-zero.
   - Files and `np.load` are opened in context managers.
10. **Caches and resumes are keyed by every input** (frames, seed, K, the source pool, the release).
    The release is pinned explicitly, and a corrupt cache recomputes.
11. **Memory and hot loops.**
    - Nothing of size T x N is materialized in float64 on a timeline path.
    - Nothing is allocated per Newton iteration.
    - Estimate the size at IEEE-300 over 72,000 frames; `python tools/bench.py --check` gates the
      timings.
12. **The readability contract.** Channel indices come from `NODE`/`EDGE`/`BRANCH`, file paths and
    attributes from `schema`, each formula exists once, and there are no positional tuples.
13. **Extras.** A new import of scipy, torch or pandapower is declared in the matching extra, guarded
    with an install message, and kept off the `import fdia_graph` path.
14. **Prose.** Read every added paragraph once for grammar, empty code fences and the docs' tone.
15. **Every factual sentence holds in this checkout.**
    - A release named as the default or current matches `registry._RELEASE`.
    - A count (files, systems, findings) is recounted, not remembered.
    - A feature that arrives with an unmerged pull request is said to depend on it.
    - A claim about another project links the file or page that shows it.
16. **Read every touched file whole.** The reviewer does, and a third of its findings sit in lines
    the pull request did not change. Look for the same kind of problem the change fixes (an argument
    still used before its model, a second copy of the old wording) elsewhere in the file.

## The order

1. `python tools/prereview.py` until it passes.
2. Work through the checklist above against the diff and every touched file, searching outward from
   every change, and fix what it finds.
3. Only then request the automated review (`python tools/pr.py request-review <num>`). Batch the
   fixes to its findings into one push, and run steps 1 and 2 again before requesting it again.
4. `python tools/review_ledger.py <num>` after each review, so the tally stays current.
