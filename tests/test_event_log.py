"""Tests for the per-chat durable event log."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from diploid_agent.runtime.event_log import EventLog


def _log(tmp_path: Path) -> EventLog:
    return EventLog(chat_dir_fn=lambda chat_id: tmp_path / chat_id)


def test_append_assigns_monotonic_seq_and_persists(tmp_path: Path) -> None:
    log = _log(tmp_path)
    e1 = log.append("chat-1", "turn_started", {"user_message": "hi"})
    e2 = log.append("chat-1", "result", {"reply": "hello"})
    assert e1["seq"] == 1
    assert e2["seq"] == 2
    assert e2["kind"] == "result"

    path = tmp_path / "chat-1" / "chat_events.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["payload"]["reply"] == "hello"


def test_since_returns_only_later_events(tmp_path: Path) -> None:
    log = _log(tmp_path)
    for i in range(5):
        log.append("chat-1", "result", {"i": i})
    out = log.since("chat-1", since=2)
    assert [e["seq"] for e in out["events"]] == [3, 4, 5]
    assert out["next_seq"] == 6
    assert out["truncated"] is False


def test_seq_resumes_from_file_tail_after_restart(tmp_path: Path) -> None:
    _log(tmp_path).append("chat-1", "result", {"reply": "before"})
    log2 = _log(tmp_path)  # fresh instance = service restart
    e = log2.append("chat-1", "result", {"reply": "after"})
    assert e["seq"] == 2
    out = log2.since("chat-1", since=0)
    assert [ev["payload"]["reply"] for ev in out["events"]] == ["before", "after"]


def test_ring_bound_marks_truncation(tmp_path: Path) -> None:
    log = _log(tmp_path)
    for i in range(600):
        log.append("chat-1", "result", {"i": i})
    out = log.since("chat-1", since=0, limit=500)
    assert out["truncated"] is True
    assert out["events"][0]["seq"] == 101
    assert out["events"][-1]["seq"] == 600


def test_next_seq_starts_at_one(tmp_path: Path) -> None:
    assert _log(tmp_path).next_seq("chat-1") == 1


def test_since_long_poll_wakes_on_append(tmp_path: Path) -> None:
    log = _log(tmp_path)

    def appender() -> None:
        time.sleep(0.05)
        log.append("chat-1", "result", {"reply": "later"})

    threading.Thread(target=appender, daemon=True).start()
    start = time.perf_counter()
    out = log.since("chat-1", since=0, wait=5.0)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
    assert out["events"][0]["payload"]["reply"] == "later"


def test_since_long_poll_returns_empty_after_timeout(tmp_path: Path) -> None:
    log = _log(tmp_path)
    start = time.perf_counter()
    out = log.since("chat-1", since=0, wait=0.05)
    assert time.perf_counter() - start < 1.0
    assert out["events"] == []
    assert out["next_seq"] == 1


def test_rings_are_per_chat(tmp_path: Path) -> None:
    log = _log(tmp_path)
    log.append("chat-a", "result", {"reply": "a"})
    log.append("chat-b", "result", {"reply": "b"})
    assert log.since("chat-a")["events"][0]["seq"] == 1
    assert log.since("chat-b")["events"][0]["seq"] == 1


def test_malformed_lines_in_file_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "chat-1" / "chat_events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"seq": 1, "kind": "result", "payload": {}}\nnot json\n{"seq": 3, "kind": "result", "payload": {}}\n',
        encoding="utf-8",
    )
    log = _log(tmp_path)
    assert log.next_seq("chat-1") == 4
