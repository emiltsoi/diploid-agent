"""Tests for diploid_agent.cron_tools — the quiet mechanism helpers."""

from __future__ import annotations

import argparse
import json
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
        file=str(f), repo=str(repo) if repo else None, stale_days=stale_days, probe=probe, ping_timeout=2
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
    assert "is 45d old" in out or "is 46d old" in out


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
