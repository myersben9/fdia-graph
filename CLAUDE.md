@AGENTS.md

## Claude Code in this repository

- `/prereview` runs the gate and the checklist review in the order `CONTRIBUTING.md` sets.
- The `checklist-reviewer` agent reviews a branch against `docs/reference/REVIEW_CHECKLIST.md`.
- `.claude/settings.json` adds one hook, `.claude/hooks/guard_git.py`. It refuses a commit with
  `--no-verify` and any push that targets `main`.
