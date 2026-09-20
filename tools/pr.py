"""Pull requests from the command line, without the gh CLI (CONTRIBUTING.md, "The pull request").

    python tools/pr.py create <branch> "<title>" <body.md>   # open a PR against main
    python tools/pr.py status <num>                          # checks on the head, reviews, comment count
    python tools/pr.py comments <num>                        # every review comment (path, line, body)
    python tools/pr.py reply <num> <comment-id> "<text>"     # answer one review comment
    python tools/pr.py wait <num> [minutes]                  # block until CI has finished and Copilot reviewed
    python tools/pr.py merge <num>                           # squash-merge on green, delete the branch

The token comes from the Git Credential Manager (`git credential fill`), the same one `git push`
uses, so nothing is stored in the repo. `merge` refuses while a check is failing or still running
and while the head has no Copilot review, which is the repo's merge rule.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

import requests

OWNER, REPO = "myersben9", "fdia-graph"
BASE = f"https://api.github.com/repos/{OWNER}/{REPO}"
COPILOT = "copilot"
# The smoke workflow's jobs, every one required green on the head before a merge; a job that has
# not registered yet counts as not green (so a merge cannot slip in while CI is still starting).
REQUIRED = (
    "tests",
    "typecheck",
    "format",
    "readability",
    "install-and-import (3.9)",
    "install-and-import (3.12)",
)
OK_OTHER = ("success", "neutral", "skipped")  # what any other listed check may end with


def token() -> str:
    out = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.strip().splitlines())["password"]


def api(method: str, path: str, **kw: Any) -> Any:
    headers = {
        "Authorization": f"Bearer {token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    r = requests.request(method, BASE + path, headers=headers, timeout=60, **kw)
    if r.status_code >= 300:
        raise SystemExit(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
    return r.json() if r.text else {}


def api_all(path: str) -> list[Any]:
    """Every item of a list endpoint, following pagination (GitHub returns 30 per page by default)."""
    out: list[Any] = []
    page = 1
    while True:
        sep = "&" if "?" in path else "?"
        batch = api("GET", f"{path}{sep}per_page=100&page={page}")
        items = batch.get("check_runs", batch) if isinstance(batch, dict) else batch
        out.extend(items)
        if len(items) < 100:
            return out
        page += 1


def _latest_runs(checks: list[dict[str, Any]]) -> dict[str, tuple[str, str | None]]:
    """One (status, conclusion) per check name: the run that started last. A rerun leaves the
    earlier attempt in the list, so the order GitHub returns must not decide which one counts."""
    latest: dict[str, dict[str, Any]] = {}
    for c in checks:
        prev = latest.get(c["name"])
        if prev is None or (c.get("started_at") or "") >= (prev.get("started_at") or ""):
            latest[c["name"]] = c
    return {name: (c["status"], c["conclusion"]) for name, c in latest.items()}


def _head_state(num: int) -> dict[str, Any]:
    pr = api("GET", f"/pulls/{num}")
    sha = pr["head"]["sha"]
    checks = api_all(f"/commits/{sha}/check-runs")
    reviews = api_all(f"/pulls/{num}/reviews")
    copilot = [r for r in reviews if COPILOT in r["user"]["login"].lower() and r["commit_id"] == sha]
    return {
        "pr": pr,
        "sha": sha,
        "checks": _latest_runs(checks),
        "copilot_on_head": [(r["state"], r["submitted_at"]) for r in copilot],
        "n_comments": len(api_all(f"/pulls/{num}/comments")),
    }


def _not_green(state: dict[str, Any]) -> list[str]:
    """Why the head is not green: every required job missing or not a completed success, and every
    other check that finished with a failure or has not finished; empty when green."""
    checks = state["checks"]
    out = [
        f"{name}: {checks[name][1] or checks[name][0]}" if name in checks else f"{name}: not started"
        for name in REQUIRED
        if checks.get(name) != ("completed", "success")
    ]
    for name, (status, conclusion) in checks.items():
        if name not in REQUIRED and (status != "completed" or conclusion not in OK_OTHER):
            out.append(f"{name}: {conclusion or status}")
    return out


def _green(state: dict[str, Any]) -> bool:
    return not _not_green(state)


def create(branch: str, title: str, body_file: str) -> None:
    body = open(body_file, encoding="utf8").read()
    pr = api("POST", "/pulls", json={"title": title, "head": branch, "base": "main", "body": body})
    print(json.dumps({"number": pr["number"], "url": pr["html_url"]}))


def status(num: int) -> None:
    s = _head_state(num)
    print(
        json.dumps(
            {
                "head": s["sha"][:8],
                "mergeable": s["pr"].get("mergeable"),
                "merged": s["pr"].get("merged"),
                "green": _green(s),
                "checks": s["checks"],
                "copilot_on_head": s["copilot_on_head"],
                "n_comments": s["n_comments"],
            },
            indent=1,
        )
    )


def comments(num: int) -> None:
    for c in api_all(f"/pulls/{num}/comments"):
        kind = "reply" if c.get("in_reply_to_id") else "comment"
        print(
            f"--- {c['path']}:{c.get('line') or c.get('original_line')}  id={c['id']}  {kind} by {c['user']['login']}"
        )
        print(c["body"])
        print()


def reply(num: int, comment_id: int, text: str) -> None:
    api("POST", f"/pulls/{num}/comments/{comment_id}/replies", json={"body": text})
    print("replied to", comment_id)


def wait(num: int, minutes: float = 25) -> None:
    t0 = time.time()
    while True:
        s = _head_state(num)
        done = all(s["checks"].get(name, ("", ""))[0] == "completed" for name in REQUIRED) and all(
            st == "completed" for st, _ in s["checks"].values()
        )
        elapsed = (time.time() - t0) / 60
        if (done and s["copilot_on_head"]) or elapsed > minutes:
            print(
                json.dumps(
                    {
                        "elapsed_min": round(elapsed, 1),
                        "head": s["sha"][:8],
                        "green": _green(s),
                        "checks": {k: v[1] for k, v in s["checks"].items()},
                        "copilot_on_head": s["copilot_on_head"],
                        "n_comments": s["n_comments"],
                    },
                    indent=1,
                )
            )
            return
        time.sleep(60)


def merge(num: int) -> None:
    s = _head_state(num)
    why = _not_green(s)
    if why:
        raise SystemExit(f"not green on {s['sha'][:8]}: " + "; ".join(why))
    if not s["copilot_on_head"]:
        raise SystemExit(f"no Copilot review on {s['sha'][:8]} yet; run `wait {num}` first")
    pr = s["pr"]
    # `sha` binds the merge to the head that was checked: GitHub refuses if a push moved it meanwhile.
    r = api(
        "PUT",
        f"/pulls/{num}/merge",
        json={"merge_method": "squash", "commit_title": f"{pr['title']} (#{num})", "sha": s["sha"]},
    )
    print("merged:", r.get("merged"), r.get("message"))
    if r.get("merged"):
        api("DELETE", f"/git/refs/heads/{pr['head']['ref']}")
        print("deleted branch", pr["head"]["ref"])


def main(argv: list[str]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8", errors="replace"
        )  # review comments carry math symbols; Windows consoles default to cp1252
    if not argv:
        raise SystemExit(__doc__)
    cmd, args = argv[0], argv[1:]
    if cmd == "create":
        create(args[0], args[1], args[2])
    elif cmd == "status":
        status(int(args[0]))
    elif cmd == "comments":
        comments(int(args[0]))
    elif cmd == "reply":
        reply(int(args[0]), int(args[1]), args[2])
    elif cmd == "wait":
        wait(int(args[0]), float(args[1]) if len(args) > 1 else 25)
    elif cmd == "merge":
        merge(int(args[0]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
