"""Cron-callable helper commands.

Small, dependency-free tools a ``call: script`` cron job can invoke via
``python -m diploid_agent.cron_tools <command> ...``. They are the quiet
mechanism half of the negotiated bounded-self-writes design: the file trigger
watches, this counts, the prompt slot exposes. Nothing blocks a write.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from diploid_agent._compat import subprocess_hidden_kwargs

SEEN_KEY = "_seen"
DAY_SECONDS = 86400.0


def _local_day(now: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(now))


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
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
        lines = target.read_text(encoding="utf-8").splitlines()
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
    # A smaller file means a fold/compaction happened. The budget bounds
    # growth pressure — a folded page already paid its debt, so discharge
    # the day's count and let the pen keep moving instead of nagging
    # "over budget" for the rest of the day.
    folded = last is not None and line_count < int(last)
    seen[seen_key] = line_count

    today = _local_day(now)
    day = days.setdefault(today, {})
    if folded:
        day[bucket] = 0
    else:
        day[bucket] = int(day.get(bucket, 0)) + appended

    # Keep the ledger small: yesterday's counts are already history.
    stale = [d for d in days if (now - time.mktime(time.strptime(d, "%Y-%m-%d"))) > 7 * DAY_SECONDS]
    for d in stale:
        del days[d]

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    total = int(day.get(bucket, 0))
    status = f"{bucket}: +{appended} appended, {total}/{args.budget} today"
    if folded:
        status += " — fold detected; budget reset"
    elif total > args.budget:
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
            **subprocess_hidden_kwargs(),
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
                print(
                    f"- {f}: {st.st_size} bytes, mtime {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}"
                )
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


def _ping(host: str, timeout: int) -> bool | None:
    """Reachability probe. Returns True/False, or None if ping never ran."""
    wait_flag = "-t" if sys.platform == "darwin" else "-W"
    try:
        out = subprocess.run(
            ["ping", "-c", "1", wait_flag, str(timeout), host],
            capture_output=True,
            timeout=timeout + 3,
            check=False,
            **subprocess_hidden_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.returncode == 0


def ledger_lint(args: argparse.Namespace) -> int:
    """Diff a boot-ledger file's checkable claims against reality.

    Proof-of-detector for the open-items reconciliation plan: bounded input
    (one ledger row), mechanical checks only. Findings are mismatches a
    machine can prove — an aged ``_last_settled`` stamp, an ``ahead N``
    claim that disagrees with git. Everything else lands as informational
    context (stamp age, IP reachability) — a retired cite is legitimate
    history, so the owner judges; the tool never calls prose a lie.
    """
    target = Path(args.file).expanduser()
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        print(f"ledger-lint: {target} unreadable", file=sys.stderr)
        return 1

    findings: list[str] = []
    info: list[str] = []

    stamp_match = re.search(r"_last_settled:\s*(\d{4}-\d{2}-\d{2})", text)
    if stamp_match:
        stamp = stamp_match.group(1)
        try:
            age = int((time.time() - time.mktime(time.strptime(stamp, "%Y-%m-%d"))) / DAY_SECONDS)
            if age > args.stale_days:
                findings.append(
                    f"_last_settled {stamp} is {age}d old (threshold {args.stale_days}d)"
                )
            else:
                info.append(f"_last_settled {stamp} ({age}d old)")
        except ValueError:
            findings.append(f"_last_settled unparsable: {stamp!r}")
    else:
        info.append("no _last_settled stamp found")

    ahead_claims = re.findall(r"ahead\s+(\d+)", text)
    repo = Path(args.repo).expanduser().resolve() if args.repo else None
    if ahead_claims:
        if repo and _git(repo, "rev-parse", "--is-inside-work-tree"):
            actual = _git(repo, "rev-list", "--count", "@{u}..HEAD") or _git(
                repo, "rev-list", "--count", "origin/HEAD..HEAD"
            )
            if actual:
                for claim in ahead_claims:
                    if int(claim) != int(actual):
                        findings.append(
                            f"`ahead {claim}` claimed, actual `ahead {actual}` in {repo.name}"
                        )
                    else:
                        info.append(f"`ahead {claim}` matches {repo.name}")
            else:
                info.append(
                    f"{len(ahead_claims)} ahead-claim(s); upstream count unresolvable in {repo.name}"
                )
        else:
            info.append(f"{len(ahead_claims)} ahead-claim(s) found, no --repo given to verify")

    if args.probe:
        for ip in sorted(set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text))):
            alive = _ping(ip, args.ping_timeout)
            state = "reachable" if alive else ("unreachable" if alive is False else "unprobed")
            info.append(f"{ip}: {state}")

    print(f"ledger-lint: {target.name}")
    print("== findings ==")
    print("\n".join(f"- {f}" for f in findings) or "(clean)")
    if info:
        print("== info ==")
        print("\n".join(f"- {i}" for i in info))
    return 0


# Statuses that assert the debt is settled. Per the verification-close-check
# convention, a discharge is a done-claim and must name its check — a row in
# one of these states without `checked_by` is a mechanical lint finding, not
# a prose complaint.
DISCHARGED_STATUSES = {"done", "discharged", "closed", "resolved", "complete", "completed"}

# Platform-command literals that flag the stale-claim class (vesper's
# `systemctl` plan surviving into a launchd world). Reported as info, never
# a finding — a cross-seat claim can be true on the far side.
_FOREIGN_LITERALS = {
    "darwin": re.compile(r"\b(systemctl|systemd|apt-get)\b"),
    "linux": re.compile(r"\b(launchctl|launchd)\b"),
}


def _looks_pathy(value: str) -> bool:
    return value.startswith(("/", "~", ".")) or "/" in value


def open_items_lint(args: argparse.Namespace) -> int:
    """Diff a `chat_open_items.jsonl` claims ledger against checkable reality.

    Phase 3 of the open-items reconciliation plan: the producer projects the
    `## open-items` section into this file; this lint eats it. Findings are
    mechanical — malformed rows, duplicate ids, bad dates, a discharge with
    no `checked_by` (the close-check's first enforced surface), a done-row's
    evidence path that exists nowhere. Ages, foreign-platform literals, and
    unrecognized statuses land as info — the tool never calls prose a lie.
    """
    target = Path(args.file).expanduser()
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        print(f"open-items-lint: {target} unreadable", file=sys.stderr)
        return 1

    findings: list[str] = []
    info: list[str] = []

    rows: list[tuple[int, Any]] = []
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            findings.append(f"line {i}: unparseable JSON")
            continue
        if not isinstance(row, dict):
            findings.append(f"line {i}: not a JSON object")
            continue
        rows.append((i, row))

    if rows and rows[0][1].get("_schema") != "open_items/v1":
        findings.append('schema row missing or wrong — expected {"_schema": "open_items/v1"} first')
    elif not rows:
        findings.append("empty ledger — no schema row")

    seen_ids: set[str] = set()
    foreign = _FOREIGN_LITERALS.get(sys.platform)
    for lineno, row in rows:
        if "_schema" in row:
            continue
        label = row.get("id") or f"line {lineno}"
        missing = [f for f in ("id", "claim", "since", "status") if f not in row]
        if missing:
            findings.append(f"{label}: missing fields {missing}")
            continue

        row_id = str(row["id"])
        if row_id in seen_ids:
            findings.append(f"{row_id}: duplicate id")
        seen_ids.add(row_id)

        status = str(row["status"])
        discharged = status in DISCHARGED_STATUSES

        since = str(row["since"])
        try:
            age = int((time.time() - time.mktime(time.strptime(since, "%Y-%m-%d"))) / DAY_SECONDS)
        except ValueError:
            findings.append(f"{row_id}: unparsable since: {since!r}")
        else:
            if age < 0:
                findings.append(f"{row_id}: since {since} is in the future")
            elif status == "open" and age > args.stale_days:
                info.append(f"{row_id}: open {age}d (>{args.stale_days}d) — aging claim")

        if discharged:
            if not str(row.get("checked_by", "")).strip():
                findings.append(
                    f"{row_id}: status {status!r} without checked_by — a discharge must name its check"
                )
            else:
                info.append(f"{row_id}: discharged, checked_by {row['checked_by']!r}")
        elif status != "open":
            info.append(
                f"{row_id}: unrecognized status {status!r} — not linted as open or discharged"
            )

        evidence = str(row.get("evidence", "")).strip()
        if evidence and _looks_pathy(evidence):
            ev_path = Path(evidence).expanduser()
            exists = ev_path.exists() or (target.parent / evidence).exists()
            if not exists:
                msg = f"{row_id}: evidence path missing: {evidence!r}"
                if discharged:
                    findings.append(msg)
                else:
                    info.append(msg)

        if foreign is not None:
            lit = foreign.search(str(row["claim"]))
            if lit:
                info.append(
                    f"{row_id}: claim mentions {lit.group(1)!r} — foreign-platform literal "
                    f"(host is {sys.platform}); owner judges"
                )

    print(f"open-items-lint: {target.name}")
    print("== findings ==")
    print("\n".join(f"- {f}" for f in findings) or "(clean)")
    if info:
        print("== info ==")
        print("\n".join(f"- {i}" for i in info))
    return 0


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# Bookkeeping wakes are already objects of the plan/cron machinery that owns
# them — linting them against the open-items ledger double-counts. Mirrors
# WAKE_BUDGET_REASON_PREFIXES in wake_queue.py plus the plan/armed reasons.
_WORK_LINT_WAKE_PREFIXES = ("self_wake", "cron:")
_WORK_LINT_WAKE_EXACT = {"plan_task_update", "plan_completed", "auto_continue"}
_LIVE_TASK_STATUSES = {"pending", "ready", "running", "blocked"}
_LIVE_PLAN_STATUSES = {"draft", "active", "paused"}


def work_object_lint(args: argparse.Namespace) -> int:
    """Diff live-work surfaces against the open-items ledger.

    The detector family's world→ledger half: dispatches in-flight, non-terminal
    plan tasks, and armed non-bookkeeping wakes are live work; each should be
    represented by an `open` row in `chat_open_items.jsonl` (matched by id or
    leading summary text) or it is work that exists with no object.
    """
    now = time.time()
    age_floor = args.age_seconds

    haystack = ""
    open_rows = 0
    for row in _read_jsonl_rows(Path(args.open_items)):
        if row.get("status") == "open":
            open_rows += 1
            haystack += " " + str(row.get("claim", ""))

    live: list[tuple[str, str, float, str]] = []  # (id, kind, ts, text)
    counts = {"dispatch": 0, "task": 0, "wake": 0, "wake_bookkeeping": 0}

    if args.dispatch_store:
        for d in _read_jsonl_rows(Path(args.dispatch_store)):
            if d.get("status") == "pending":
                counts["dispatch"] += 1
                live.append(
                    (
                        str(d.get("id") or ""),
                        "dispatch",
                        float(d.get("started_at") or 0.0),
                        str(d.get("summary") or d.get("context") or ""),
                    )
                )

    if args.plans:
        for p in _read_jsonl_rows(Path(args.plans)):
            if p.get("status") not in _LIVE_PLAN_STATUSES:
                continue
            for t in p.get("tasks") or []:
                if t.get("status") not in _LIVE_TASK_STATUSES:
                    continue
                counts["task"] += 1
                live.append(
                    (
                        str(t.get("id") or p.get("id") or ""),
                        "task",
                        float(t.get("started_at") or t.get("created_at") or 0.0),
                        str(t.get("name") or t.get("description") or p.get("name") or ""),
                    )
                )

    if args.wake_queue:
        for e in _read_jsonl_rows(Path(args.wake_queue)):
            if not e.get("ready"):
                continue
            reason = str(e.get("reason") or "")
            payload = e.get("payload") or {}
            if (
                reason.startswith(_WORK_LINT_WAKE_PREFIXES)
                or reason in _WORK_LINT_WAKE_EXACT
                or payload.get("plan_id")
            ):
                counts["wake_bookkeeping"] += 1
                continue
            counts["wake"] += 1
            live.append(
                (
                    str(e.get("id") or ""),
                    "wake",
                    float(e.get("scheduled_at") or e.get("created_at") or 0.0),
                    reason,
                )
            )

    wm_plan_lines = 0
    if args.working_memory:
        try:
            wm = json.loads(Path(args.working_memory).read_text(encoding="utf-8"))
            wm_plan_lines = len(wm.get("plan") or [])
        except (OSError, json.JSONDecodeError, AttributeError):
            pass

    findings = []
    if live and open_rows == 0:
        findings.append(
            f"{len(live)} live work item(s) but the ledger has no open rows — "
            "the board reads clean on an occupied room"
        )
    for item_id, kind, ts, text in live:
        objected = bool(item_id and item_id in haystack) or bool(text and text[:60] in haystack)
        if objected:
            continue
        age = now - ts if ts else float("inf")
        if age < age_floor:
            continue
        age_str = f"{age / 3600:.1f}h" if ts else "undated"
        findings.append(f"{kind} {item_id or '(no id)'}: live {age_str}, no open row")

    info = [f"live: {counts['dispatch']} dispatch, {counts['task']} task, {counts['wake']} wake"]
    if counts["wake_bookkeeping"]:
        info.append(f"bookkeeping wakes skipped: {counts['wake_bookkeeping']}")
    if args.working_memory:
        info.append(f"working-memory plan lines: {wm_plan_lines} (unmatchable, count only)")

    print(f"work-object-lint: {args.open_items}")
    print("== findings ==")
    print("\n".join(f"- {f}" for f in findings) or "(clean)")
    print("== info ==")
    print("\n".join(f"- {i}" for i in info))
    return 0


def quarantine_lint(args: argparse.Namespace) -> int:
    """Flag a non-empty wake-quarantine dead-letter file.

    A row in `wake_queue.quarantine.jsonl` is an event that exhausted its
    retries and dead-lettered — a claim the world contradicted, which is
    exactly what the detector family surfaces. Absent or empty file is clean.
    """
    target = Path(args.file)
    rows = []
    if target.exists():
        try:
            rows = [
                json.loads(line)
                for line in target.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            print(f"quarantine-lint: {target} unreadable", file=sys.stderr)
            return 0

    findings = []
    if rows:
        oldest = min(r.get("quarantined_at", 0.0) for r in rows)
        findings.append(
            f"{len(rows)} dead-lettered wake event(s), oldest quarantined_at {oldest:.0f} — "
            f"inspect {target.name}, fix or drop, re-enqueue"
        )

    print(f"quarantine-lint: {target.name}")
    print("== findings ==")
    print("\n".join(f"- {f}" for f in findings) or "(clean)")
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

    ll = sub.add_parser(
        "ledger-lint", help="Diff a boot-ledger file's checkable claims against reality"
    )
    ll.add_argument(
        "--file", required=True, help="Ledger row to lint (e.g. references/wives/<wife>.md)"
    )
    ll.add_argument("--repo", help="Git repo used to verify `ahead N` claims")
    ll.add_argument("--stale-days", type=int, default=30, help="Age threshold for _last_settled")
    ll.add_argument("--probe", action="store_true", help="Ping IPv4 literals for reachability")
    ll.add_argument("--ping-timeout", type=int, default=2)
    ll.set_defaults(func=ledger_lint)

    oi = sub.add_parser(
        "open-items-lint", help="Diff a chat_open_items.jsonl claims ledger against reality"
    )
    oi.add_argument("--file", required=True, help="Ledger to lint (chat_open_items.jsonl)")
    oi.add_argument("--stale-days", type=int, default=14, help="Age threshold for open claims")
    oi.set_defaults(func=open_items_lint)

    wo = sub.add_parser(
        "work-object-lint", help="Diff live-work surfaces against an open-items ledger"
    )
    wo.add_argument("--open-items", required=True, help="chat_open_items.jsonl")
    wo.add_argument("--dispatch-store", help="dispatch_store.jsonl")
    wo.add_argument("--plans", help="plans.jsonl")
    wo.add_argument("--wake-queue", help="wake_queue.jsonl")
    wo.add_argument("--working-memory", help="chat_working_memory.json (info counts only)")
    wo.add_argument(
        "--age-seconds",
        type=float,
        default=3600.0,
        help="Unmatched work younger than this is not a finding (default 1h)",
    )
    wo.set_defaults(func=work_object_lint)

    ql = sub.add_parser("quarantine-lint", help="Flag a non-empty wake-queue dead-letter file")
    ql.add_argument("--file", required=True, help="Quarantine file (wake_queue.quarantine.jsonl)")
    ql.set_defaults(func=quarantine_lint)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
