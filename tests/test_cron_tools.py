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
