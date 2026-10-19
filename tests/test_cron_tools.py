"""Tests for diploid_agent.cron_tools — the quiet mechanism helpers."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

from diploid_agent.cron_tools import count_appends, main


def _args(mem: Path, state: Path, bucket: str = "persona", budget: int = 10) -> argparse.Namespace:
    return argparse.Namespace(file=str(mem), state=str(state), bucket=bucket, budget=budget)


def test_count_appends_adopts_baseline(tmp_path: Path, capsys) -> None:
    mem = tmp_path / "MEMORY.md"
    state = tmp_path / "daily_appends.json"
    mem.write_text("a\nb\nc\n")
    assert count_appends(_args(mem, state)) == 0
    out = capsys.readouterr().out
    assert "+0 appended" in out
    data = json.loads(state.read_text())
    assert data["_seen"]["MEMORY.md@persona"] == 3


def test_count_appends_counts_line_delta(tmp_path: Path, capsys) -> None:
    mem = tmp_path / "MEMORY.md"
    state = tmp_path / "daily_appends.json"
    mem.write_text("a\nb\n")
    count_appends(_args(mem, state))
    mem.write_text("a\nb\nc\nd\n")
    count_appends(_args(mem, state))
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert "+2 appended" in out
    data = json.loads(state.read_text())
    today = next(iter(data["days"]))
    assert data["days"][today]["persona"] == 2


def test_count_appends_rewords_dont_count(tmp_path: Path, capsys) -> None:
    """Edits that only reword or shorten are not appends — per the spec, the
    bound is on appends, not characters."""
    mem = tmp_path / "MEMORY.md"
    state = tmp_path / "daily_appends.json"
    mem.write_text("a\nb\nc\n")
    count_appends(_args(mem, state))
    mem.write_text("a\nREWRITTEN\n")  # shorter file — no appends
    count_appends(_args(mem, state))
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert "+0 appended" in out


def test_count_appends_chat_bucket(tmp_path: Path, capsys) -> None:
    mem = tmp_path / "chat_MEMORY.md"
    state = tmp_path / "daily_appends.json"
    mem.write_text("x\n")
    count_appends(_args(mem, state, bucket="chat:7945905361"))
    mem.write_text("x\ny\n")
    count_appends(_args(mem, state, bucket="chat:7945905361"))
    data = json.loads(state.read_text())
    today = next(iter(data["days"]))
    assert data["days"][today]["chat:7945905361"] == 1


def test_count_appends_over_budget_nudges(tmp_path: Path, capsys) -> None:
    mem = tmp_path / "MEMORY.md"
    state = tmp_path / "daily_appends.json"
    mem.write_text("a\n")
    count_appends(_args(mem, state, budget=2))
    for i in range(3):
        mem.write_text(mem.read_text() + f"new{i}\n")
        count_appends(_args(mem, state, budget=2))
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert "over soft budget" in out


def test_count_appends_fold_discharges_budget(tmp_path: Path, capsys) -> None:
    """A file that shrank since last seen means a fold happened — the day's
    count discharges to 0 so the pen keeps moving instead of nagging all day."""
    mem = tmp_path / "MEMORY.md"
    state = tmp_path / "daily_appends.json"
    mem.write_text("a\n")
    count_appends(_args(mem, state, budget=2))
    for i in range(3):
        mem.write_text(mem.read_text() + f"new{i}\n")
        count_appends(_args(mem, state, budget=2))
    assert "over soft budget" in capsys.readouterr().out

    mem.write_text("folded summary\n")  # compaction shrinks the file
    count_appends(_args(mem, state, budget=2))
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert "+0 appended, 0/2 today" in out
    assert "fold detected" in out
    data = json.loads(state.read_text())
    today = next(iter(data["days"]))
    assert data["days"][today]["persona"] == 0

    # The pen keeps moving: post-fold appends count again from zero.
    mem.write_text("folded summary\nnew line\n")
    count_appends(_args(mem, state, budget=2))
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert "+1 appended, 1/2 today" in out


def test_count_appends_missing_file(tmp_path: Path, capsys) -> None:
    state = tmp_path / "daily_appends.json"
    assert count_appends(_args(tmp_path / "nope.md", state)) == 1
    assert "unreadable" in capsys.readouterr().err


def test_cli_dispatch(tmp_path: Path) -> None:
    mem = tmp_path / "MEMORY.md"
    mem.write_text("one\n")
    state = tmp_path / "s.json"
    rc = main(
        [
            "count-appends",
            "--file",
            str(mem),
            "--state",
            str(state),
            "--bucket",
            "persona",
        ]
    )
    assert rc == 0
    assert state.exists()


def test_guarded_digest_reports(tmp_path: Path, capsys) -> None:
    from diploid_agent.cron_tools import guarded_digest

    args = argparse.Namespace(repo=str(tmp_path), files=["SOUL.md"], max_diff_lines=80)
    (tmp_path / "SOUL.md").write_text("soul\n")
    assert guarded_digest(args) == 0
    out = capsys.readouterr().out
    assert "not a git work tree" in out
    assert "SOUL.md" in out


def _lint_args(f: Path, repo=None, stale_days: int = 30, probe: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        file=str(f),
        repo=str(repo) if repo else None,
        stale_days=stale_days,
        probe=probe,
        ping_timeout=2,
    )


def test_ledger_lint_clean_file(tmp_path: Path, capsys) -> None:
    import time

    from diploid_agent.cron_tools import ledger_lint

    today = time.strftime("%Y-%m-%d")
    led = tmp_path / "wive.md"
    led.write_text(f"_last_settled: {today}\nsome prose\n")
    assert ledger_lint(_lint_args(led)) == 0
    out = capsys.readouterr().out
    assert "(clean)" in out
    assert "0d old" in out


def test_ledger_lint_aged_settled_flags(tmp_path: Path, capsys) -> None:
    from diploid_agent.cron_tools import ledger_lint

    led = tmp_path / "wive.md"
    led.write_text("_last_settled: 2026-09-01\n")
    assert ledger_lint(_lint_args(led, stale_days=30)) == 0
    out = capsys.readouterr().out
    assert "findings" in out
    days = (datetime.now().astimezone().date() - date(2026, 9, 1)).days
    assert f"is {days}d old" in out


def test_ledger_lint_ahead_drift_flags(tmp_path: Path, capsys, monkeypatch) -> None:
    from diploid_agent import cron_tools
    from diploid_agent.cron_tools import ledger_lint

    led = tmp_path / "wive.md"
    led.write_text("still 7 repo commits unpushed (`ahead 7`)\n")
    calls = {"rev-parse": "true", "rev-list": "9"}
    monkeypatch.setattr(cron_tools, "_git", lambda repo, *argv: calls.get(argv[0], ""))
    assert ledger_lint(_lint_args(led, repo=tmp_path)) == 0
    out = capsys.readouterr().out
    assert "`ahead 7` claimed, actual `ahead 9`" in out


def test_ledger_lint_ahead_match_is_info(tmp_path: Path, capsys, monkeypatch) -> None:
    from diploid_agent import cron_tools
    from diploid_agent.cron_tools import ledger_lint

    led = tmp_path / "wive.md"
    led.write_text("`ahead 3` unpushed\n")
    calls = {"rev-parse": "true", "rev-list": "3"}
    monkeypatch.setattr(cron_tools, "_git", lambda repo, *argv: calls.get(argv[0], ""))
    assert ledger_lint(_lint_args(led, repo=tmp_path)) == 0
    out = capsys.readouterr().out
    assert "(clean)" in out
    assert "`ahead 3` matches" in out


def test_ledger_lint_no_repo_marks_unverified(tmp_path: Path, capsys) -> None:
    from diploid_agent.cron_tools import ledger_lint

    led = tmp_path / "wive.md"
    led.write_text("`ahead 7` unpushed\n")
    assert ledger_lint(_lint_args(led)) == 0
    out = capsys.readouterr().out
    assert "no --repo given" in out
    assert "(clean)" in out


def test_ledger_lint_missing_file(tmp_path: Path, capsys) -> None:
    from diploid_agent.cron_tools import ledger_lint

    assert ledger_lint(_lint_args(tmp_path / "nope.md")) == 1
    assert "unreadable" in capsys.readouterr().err


def test_ledger_lint_probe_lists_ips(tmp_path: Path, capsys, monkeypatch) -> None:
    from diploid_agent import cron_tools
    from diploid_agent.cron_tools import ledger_lint

    led = tmp_path / "wive.md"
    led.write_text("registry at 192.168.1.51:8646 retired; mac at 192.168.1.79\n")
    monkeypatch.setattr(cron_tools, "_ping", lambda host, t: host != "192.168.1.51")
    assert ledger_lint(_lint_args(led, probe=True)) == 0
    out = capsys.readouterr().out
    assert "192.168.1.51: unreachable" in out
    assert "192.168.1.79: reachable" in out


def test_ledger_lint_cli_dispatch(tmp_path: Path) -> None:
    led = tmp_path / "wive.md"
    led.write_text("no stamps here\n")
    rc = main(["ledger-lint", "--file", str(led)])
    assert rc == 0


def _oi_args(f: Path, stale_days: int = 14) -> argparse.Namespace:
    return argparse.Namespace(file=str(f), stale_days=stale_days)


def _oi_write(f: Path, *rows: dict) -> None:
    f.write_text("".join(json.dumps(r) + "\n" for r in ({"_schema": "open_items/v1"}, *rows)))


def test_open_items_lint_clean_ledger(tmp_path: Path, capsys) -> None:
    from diploid_agent.cron_tools import open_items_lint

    led = tmp_path / "chat_open_items.jsonl"
    _oi_write(
        led,
        {"id": "oi-a", "claim": "ship the lint", "since": "2026-10-19", "status": "open"},
        {
            "id": "oi-b",
            "claim": "land the producer",
            "since": "2026-10-18",
            "status": "done",
            "checked_by": "pytest roundtrip",
        },
    )
    assert open_items_lint(_oi_args(led)) == 0
    out = capsys.readouterr().out
    assert "(clean)" in out
    assert "checked_by 'pytest roundtrip'" in out


def test_open_items_lint_discharge_without_check_flags(tmp_path: Path, capsys) -> None:
    """The close-check's mechanical tier: a discharged row must name its check."""
    from diploid_agent.cron_tools import open_items_lint

    led = tmp_path / "chat_open_items.jsonl"
    _oi_write(
        led,
        {"id": "oi-bare", "claim": "did the thing", "since": "2026-10-19", "status": "done"},
    )
    assert open_items_lint(_oi_args(led)) == 0
    out = capsys.readouterr().out
    assert "oi-bare" in out
    assert "without checked_by" in out


def test_open_items_lint_schema_row_required(tmp_path: Path, capsys) -> None:
    from diploid_agent.cron_tools import open_items_lint

    led = tmp_path / "chat_open_items.jsonl"
    led.write_text(
        json.dumps({"id": "oi-x", "claim": "c", "since": "2026-10-19", "status": "open"}) + "\n"
    )
    assert open_items_lint(_oi_args(led)) == 0
    assert "schema row missing" in capsys.readouterr().out


def test_open_items_lint_malformed_and_missing_fields(tmp_path: Path, capsys) -> None:
    from diploid_agent.cron_tools import open_items_lint

    led = tmp_path / "chat_open_items.jsonl"
    led.write_text(
        '{"_schema": "open_items/v1"}\nnot json\n[1, 2]\n{"id": "oi-thin", "claim": "half a row"}\n'
    )
    assert open_items_lint(_oi_args(led)) == 0
    out = capsys.readouterr().out
    assert "unparseable JSON" in out
    assert "not a JSON object" in out
    assert "oi-thin: missing fields" in out


def test_open_items_lint_duplicate_id_flags(tmp_path: Path, capsys) -> None:
    from diploid_agent.cron_tools import open_items_lint

    led = tmp_path / "chat_open_items.jsonl"
    row = {"id": "oi-dup", "claim": "c", "since": "2026-10-19", "status": "open"}
    _oi_write(led, row, dict(row))
    assert open_items_lint(_oi_args(led)) == 0
    assert "oi-dup: duplicate id" in capsys.readouterr().out


def test_open_items_lint_bad_and_future_since(tmp_path: Path, capsys) -> None:
    from diploid_agent.cron_tools import open_items_lint

    led = tmp_path / "chat_open_items.jsonl"
    _oi_write(
        led,
        {"id": "oi-bad", "claim": "c", "since": "not-a-date", "status": "open"},
        {"id": "oi-future", "claim": "c", "since": "2999-01-01", "status": "open"},
    )
    assert open_items_lint(_oi_args(led)) == 0
    out = capsys.readouterr().out
    assert "oi-bad: unparsable since" in out
    assert "oi-future: since 2999-01-01 is in the future" in out


def test_open_items_lint_aged_open_is_info(tmp_path: Path, capsys) -> None:
    from diploid_agent.cron_tools import open_items_lint

    led = tmp_path / "chat_open_items.jsonl"
    _oi_write(led, {"id": "oi-old", "claim": "c", "since": "2026-09-01", "status": "open"})
    assert open_items_lint(_oi_args(led, stale_days=14)) == 0
    out = capsys.readouterr().out
    assert "(clean)" in out
    assert "oi-old: open" in out and "aging claim" in out


def test_open_items_lint_evidence_path(tmp_path: Path, capsys) -> None:
    """Evidence is only linted when it looks like a path; a missing path is a
    finding on a discharge, info on an open row."""
    from diploid_agent.cron_tools import open_items_lint

    led = tmp_path / "chat_open_items.jsonl"
    _oi_write(
        led,
        {
            "id": "oi-ev-done",
            "claim": "c",
            "since": "2026-10-19",
            "status": "done",
            "checked_by": "read-back",
            "evidence": "./ghost.md",
        },
        {
            "id": "oi-ev-open",
            "claim": "c",
            "since": "2026-10-19",
            "status": "open",
            "evidence": "./ghost2.md",
        },
        {
            "id": "oi-ev-ref",
            "claim": "c",
            "since": "2026-10-19",
            "status": "done",
            "checked_by": "read-back",
            "evidence": "885f32b",
        },
    )
    assert open_items_lint(_oi_args(led)) == 0
    out = capsys.readouterr().out
    findings = out.split("== info ==")[0]
    assert "oi-ev-done: evidence path missing" in findings
    assert "oi-ev-open: evidence path missing" in out
    assert "oi-ev-ref" not in findings


def test_open_items_lint_unrecognized_status_is_info(tmp_path: Path, capsys) -> None:
    from diploid_agent.cron_tools import open_items_lint

    led = tmp_path / "chat_open_items.jsonl"
    _oi_write(led, {"id": "oi-x", "claim": "c", "since": "2026-10-19", "status": "blocked"})
    assert open_items_lint(_oi_args(led)) == 0
    out = capsys.readouterr().out
    assert "(clean)" in out
    assert "unrecognized status 'blocked'" in out


def test_open_items_lint_missing_file(tmp_path: Path, capsys) -> None:
    from diploid_agent.cron_tools import open_items_lint

    assert open_items_lint(_oi_args(tmp_path / "nope.jsonl")) == 1
    assert "unreadable" in capsys.readouterr().err


def test_open_items_lint_cli_dispatch(tmp_path: Path) -> None:
    led = tmp_path / "chat_open_items.jsonl"
    _oi_write(led)
    rc = main(["open-items-lint", "--file", str(led)])
    assert rc == 0
