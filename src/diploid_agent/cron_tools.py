"""Cron-callable helper commands.

Small, dependency-free tools a ``call: script`` cron job can invoke via
``python -m diploid_agent.cron_tools <command> ...``. They are the quiet
mechanism half of the negotiated bounded-self-writes design: the file trigger
watches, this counts, the prompt slot exposes. Nothing blocks a write.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SEEN_KEY = "_seen"
DAY_SECONDS = 86400.0


def _local_day(now: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(now))


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def count_appends(args: argparse.Namespace) -> int:
    """Count appended lines of a memory file into a daily-appends ledger.

    The ledger lives at ``--state`` (one file per persona, e.g.
    ``sessions-<persona>/daily_appends.json``)::

        {"days": {"2026-09-16": {"persona": 2, "chat:7945905361": 1}},
         "_seen": {"MEMORY.md": 141, "chat_MEMORY.md@7945905361": 60}}

    An append is ``current_lines - seen_lines`` (floored at 0); edits that only
    reword or remove lines do not count, matching the spec's "appends, not
    characters". Prints a one-line status for the cron result record.
    """
    now = time.time()
    target = Path(args.file).expanduser()
    state_path = Path(args.state).expanduser()
    bucket = args.bucket
    try:
        lines = target.read_text().splitlines()
        line_count = len(lines)
    except OSError:
        print(f"count-appends: {target} unreadable", file=sys.stderr)
        return 1

    state = _load_state(state_path)
    days = state.setdefault("days", {})
    seen = state.setdefault(SEEN_KEY, {})
    seen_key = f"{target.name}@{bucket}"
    last = seen.get(seen_key)
    appended = 0 if last is None else max(0, line_count - int(last))
    seen[seen_key] = line_count

    today = _local_day(now)
    day = days.setdefault(today, {})
    day[bucket] = int(day.get(bucket, 0)) + appended

    # Keep the ledger small: yesterday's counts are already history.
    stale = [
        d
        for d in days
        if (now - time.mktime(time.strptime(d, "%Y-%m-%d"))) > 7 * DAY_SECONDS
    ]
    for d in stale:
        del days[d]

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))

    total = int(day.get(bucket, 0))
    status = f"{bucket}: +{appended} appended, {total}/{args.budget} today"
    if total > args.budget:
        status += " — over soft budget; consider compacting before more appends"
    print(status)
    return 0


def _git(repo: Path, *argv: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *argv],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip()


def guarded_digest(args: argparse.Namespace) -> int:
    """Digest changes to guarded identity files for the operator notice.

    The GUARDED tier of the negotiated bounded-self-writes design: the edit
    goes live immediately, this digest is the asynchronous review notice.
    Shows which files changed, the uncommitted diff, and the last commit
    touching them — Emil keeps rollback either way.
    """
    repo = Path(args.repo).expanduser()
    files = args.files
    if not _git(repo, "rev-parse", "--is-inside-work-tree"):
        print(f"guarded-digest: {repo} is not a git work tree — mtime change only")
        for f in files:
            p = repo / f
            try:
                st = p.stat()
                print(f"- {f}: {st.st_size} bytes, mtime {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}")
            except OSError:
                print(f"- {f}: missing")
        return 0

    status = _git(repo, "status", "--short", "--", *files)
    print(f"guarded files in {repo.name}: {', '.join(files)}")
    print("== status ==")
    print(status or "(clean — no uncommitted changes)")
    diff = _git(repo, "diff", "--", *files)
    if diff:
        print("== diff (uncommitted) ==")
        print("\n".join(diff.splitlines()[: args.max_diff_lines]))
    last = _git(repo, "log", "-1", "--format=%h %ad %s", "--date=iso", "--", *files)
    print("== last commit touching these files ==")
    print(last or "(none)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="diploid_agent.cron_tools")
    sub = parser.add_subparsers(dest="command", required=True)

    ca = sub.add_parser("count-appends", help="Count appended lines into daily_appends.json")
    ca.add_argument("--file", required=True, help="Memory file being watched")
    ca.add_argument("--state", required=True, help="Ledger path (daily_appends.json)")
    ca.add_argument(
        "--bucket",
        required=True,
        help="Counter bucket: 'persona' for MEMORY.md, 'chat:<id>' for chat_MEMORY.md",
    )
    ca.add_argument("--budget", type=int, default=10, help="Soft daily append budget")
    ca.set_defaults(func=count_appends)

    gd = sub.add_parser("guarded-digest", help="Digest guarded-file changes for review")
    gd.add_argument("--repo", required=True, help="Git repo (persona dir) to digest")
    gd.add_argument("--files", nargs="+", required=True, help="Guarded files to report on")
    gd.add_argument("--max-diff-lines", type=int, default=80)
    gd.set_defaults(func=guarded_digest)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
