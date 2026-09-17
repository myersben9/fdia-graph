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
MIN_CHECKS = 6  # the smoke workflow's jobs; fewer means CI has not started on this head yet


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


def _head_state(num: int) -> dict[str, Any]:
    pr = api("GET", f"/pulls/{num}")
    sha = pr["head"]["sha"]
    checks = api_all(f"/commits/{sha}/check-runs")
    reviews = api_all(f"/pulls/{num}/reviews")
    copilot = [r for r in reviews if COPILOT in r["user"]["login"].lower() and r["commit_id"] == sha]
    return {
        "pr": pr,
        "sha": sha,
        "checks": {c["name"]: (c["status"], c["conclusion"]) for c in checks},
        "copilot_on_head": [(r["state"], r["submitted_at"]) for r in copilot],
        "n_comments": len(api_all(f"/pulls/{num}/comments")),
    }


def _green(state: dict[str, Any]) -> bool:
    checks = state["checks"]
    return len(checks) >= MIN_CHECKS and all(s == "completed" and c == "success" for s, c in checks.values())


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
        done = len(s["checks"]) >= MIN_CHECKS and all(st == "completed" for st, _ in s["checks"].values())
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
    if not _green(s):
        raise SystemExit(f"not green on {s['sha'][:8]}: {s['checks']}")
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
