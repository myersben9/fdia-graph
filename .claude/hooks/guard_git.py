"""Claude Code PreToolUse hook: refuse the two git actions this repository never wants from an agent.

Reads the tool call as JSON on stdin. Exit status 2 blocks the call and sends stderr to the agent;
any other outcome lets the call through, so a malformed input never stops work.
"""

import json
import re
import sys

GIT = r"\bgit\b[^|;&\n]*\b"
NO_VERIFY = re.compile(GIT + r"commit\b[^|;&\n]*--no-verify")
PUSH = re.compile(GIT + r"push\b([^|;&\n]*)")
TO_MAIN = re.compile(r"(^|[\s:+])(refs/heads/)?(main|master)(\s|$)")


def refusal(command: str) -> str:
    """The reason `command` is refused, or an empty string when it may run."""
    if NO_VERIFY.search(command):
        return "--no-verify skips the commit checks. Fix what the check reports instead."
    for push in PUSH.finditer(command):
        if TO_MAIN.search(push.group(1)):
            return "a push that targets main. Changes reach main through a pull request (CONTRIBUTING.md)."
    return ""


def main() -> int:
    try:
        command = json.load(sys.stdin)["tool_input"]["command"]
    except (ValueError, KeyError, TypeError):
        return 0
    reason = refusal(command) if isinstance(command, str) else ""
    if reason:
        print(f"Blocked by .claude/hooks/guard_git.py: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
