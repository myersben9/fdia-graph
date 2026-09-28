"""A running tally of what the automated reviewer finds, so the local review can stop repeating it
(docs/reference/REVIEW_CHECKLIST.md).

    python tools/review_ledger.py 147 150        # add these pull requests' findings to the ledger
    python tools/review_ledger.py                # the tally of the ledger as it stands

Every finding the reviewer posts is recorded once in `docs/reference/review_ledger.csv`: the inline
comments and the ones its overview lists as "previously missed" (in files the pull request touched,
but in lines it did not change). Each gets a kind from the first rule of `KINDS` its text matches;
a finding no rule matches is `logic and edge cases`, and a rule is added when a kind keeps coming up. The ledger
keeps a one-line title per finding, not the reviewer's full text.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pr  # noqa: E402  (the pull-request helper beside this file)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "docs", "reference", "review_ledger.csv")
COLUMNS = ("pr", "where", "kind", "path", "title")

# (kind, pattern), first match wins; ordered from the most specific
KINDS = (
    ("generated file stale", r"diagram|\.png|\.svg|models_doc|data dictionary|regenerat|frozen fixture"),
    ("compatibility", r"breaks? callers|deprecat|re-export|signature change|renamed|removed .* name"),
    ("cache and resume", r"\bresume|\bcache|manifest|fragment"),
    (
        "input escapes validation",
        r"ConfigError|raw `?(Type|Key|Value|Index)Error|AtLeast|accepts? (a )?floats?|not (validated|enforced)"
        r"|validat\w* .*before|reject|malformed|Integer\(\)|non-?finite|negative",
    ),
    ("export missing", r"\bexport(ed)? from|not exported|PUBLIC"),
    ("performance", r"per (call|iteration)|every (prediction|call)|allocat|memory|recreates"),
    (
        "doc claim does not match the code",
        r"does not exist|not present|stale|this checkout|current(ly)? (default|tree|checkout)|docstring"
        r"|describes?|claims?|citation|cites|says|inventory|conflicts",
    ),
    ("tooling", r"regex|parser|workflow|git diff|exit (code|status)|\bgate\b"),
)


def kind_of(text: str) -> str:
    return next((k for k, pat in KINDS if re.search(pat, text, re.I)), "logic and edge cases")


def _is_copilot(login: str) -> bool:
    return "copilot" in login.lower()


def findings(num: int) -> list[dict[str, str]]:
    """Every finding the reviewer posted on one pull request, inline and "previously missed"."""
    out = []
    for c in pr.api_all(f"/pulls/{num}/comments"):
        if _is_copilot(c["user"]["login"]) and c.get("in_reply_to_id") is None:
            body = c["body"].strip()
            out.append(dict(where="inline", path=c["path"], title=body.split(". ")[0][:160], text=body))
    for r in pr.api_all(f"/pulls/{num}/reviews"):
        body = r.get("body") or ""
        if not _is_copilot(r["user"]["login"]) or "Previously missed" not in body:
            continue
        missed = re.sub(r"<picture>.*?</picture>", "", body[body.index("Previously missed") :], flags=re.S)
        for m in re.finditer(r"<summary>\s*(.*?)</summary>\s*(.*?)</details>", missed, flags=re.S):
            title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            if title.startswith(("Previously", "In code")):
                continue
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(2)))
            path = re.search(r"`([^`]+?)(?::\d+)?`", text)
            out.append(
                dict(
                    where="missed",
                    path=path.group(1).replace("\u200b", "") if path else "",
                    title=title,
                    text=text,
                )
            )
    for f in out:
        f["kind"] = kind_of(f["title"] + " " + f.pop("text"))
    return out


def read_ledger() -> list[dict[str, str]]:
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def add(nums: list[int]) -> None:
    rows = read_ledger()
    seen = {(r["pr"], r["where"], r["path"], r["title"]) for r in rows}
    for num in nums:
        for f in findings(num):
            key = (str(num), f["where"], f["path"], f["title"])
            if key not in seen:
                seen.add(key)
                rows.append({"pr": str(num), **f})
    rows.sort(key=lambda r: (int(r["pr"]), r["where"], r["title"]))
    with open(LEDGER, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows({c: r[c] for c in COLUMNS} for r in rows)


def tally() -> None:
    rows = read_ledger()
    prs = {r["pr"] for r in rows}
    missed = sum(r["where"] == "missed" for r in rows)
    print(
        f"{len(rows)} findings on {len(prs)} pull requests; {missed} in lines the pull request did not change\n"
    )
    print(f"{'kind':36s} {'findings':>8s} {'PRs':>4s}")
    for kind, n in Counter(r["kind"] for r in rows).most_common():
        print(f"{kind:36s} {n:8d} {len({r['pr'] for r in rows if r['kind'] == kind}):4d}")


def main(argv: list[str]) -> None:
    if argv:
        add([int(a) for a in argv])
    tally()


if __name__ == "__main__":
    main(sys.argv[1:])
