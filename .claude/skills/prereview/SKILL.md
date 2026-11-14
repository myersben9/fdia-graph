---
name: prereview
description: Runs this repository's pre-review gate and checklist review on the current branch, in the order CONTRIBUTING.md sets. Use before a pull request is opened or marked ready, or when the user says "prereview", "check my branch" or "is this ready for review".
---

# Pre-review

The order comes from "The pull request" in `CONTRIBUTING.md`. Do not reorder it.

## 1. The gate

Run `python tools/prereview.py`. Add `--fast` only when the user asks to skip the test suite, and say
in the report that the suite did not run.

Report each gate's PASS, FAIL or SKIP as printed. On a FAIL, show the failing output as it is and
stop. Fix it only when the user asked for fixes. Never weaken an assertion, skip a test or change a
pinned answer to get a PASS.

## 2. The checklist

When the gate passes, dispatch the `checklist-reviewer` agent on the branch. Pass its findings on
unchanged, most frequent kind first.

## 3. The report

- the gate results
- the checklist findings, each with its item number and `file:line`
- the "Decisions" list for the pull request body: every choice made on this branch that the rules
  did not settle, one line each with the reason

Stop there. Opening the draft, marking it ready and merging are the user's calls.
