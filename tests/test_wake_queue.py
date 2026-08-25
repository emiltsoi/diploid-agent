"""Tests for the persistent, multi-process safe WakeQueue."""

import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

from devin_fleet_harness.models import WakeEvent
from devin_fleet_harness.runtime.wake_queue import WakeQueue

mp.set_start_method("spawn", force=True)


def _pop_worker(path: Path, result_queue: Any, now: float, lease_seconds: float) -> None:
    q = WakeQueue(path)
    due = q.pop_due(now=now, lease_seconds=lease_seconds)
    result_queue.put([e.id for e in due])


def test_enqueue_and_pop_due(tmp_path: Path) -> None:
    q = WakeQueue(tmp_path / "wake.jsonl")
    e = q.enqueue(
        WakeEvent(
            id="w1",
            chat_id="c1",
            reason="dispatch",
            priority=1,
            scheduled_at=0.0,
            created_at=time.time(),
        )
    )
    q.ready(e.id)
    due = q.pop_due(now=time.time() + 1, lease_seconds=60.0)
    assert len(due) == 1
    assert due[0].id == "w1"
    assert due[0].leased_until is not None


def test_pop_due_only_ready(tmp_path: Path) -> None:
    q = WakeQueue(tmp_path / "wake.jsonl")
    q.enqueue(
        WakeEvent(
            id="w1",
            chat_id="c1",
            reason="dispatch",
            priority=1,
            scheduled_at=0.0,
            created_at=time.time(),
        )
    )
    assert q.pop_due(now=time.time() + 1, lease_seconds=60.0) == []


def test_complete_is_idempotent(tmp_path: Path) -> None:
    q = WakeQueue(tmp_path / "wake.jsonl")
    q.enqueue(
        WakeEvent(
            id="w1",
            chat_id="c1",
            reason="dispatch",
            priority=1,
            scheduled_at=0.0,
            created_at=time.time(),
        )
    )
    q.ready("w1")
    e1 = q.complete("w1")
    e2 = q.complete("w1")
    assert e1 is not None
    assert e2 is None


def test_fail_reschedules_and_clears_lease(tmp_path: Path) -> None:
    q = WakeQueue(tmp_path / "wake.jsonl")
    q.enqueue(
        WakeEvent(
            id="w1",
            chat_id="c1",
            reason="dispatch",
            priority=1,
            scheduled_at=0.0,
            created_at=time.time(),
        )
    )
    q.ready("w1", now=0.0)
    claimed = q.pop_due(now=1.0, lease_seconds=60.0)
    assert len(claimed) == 1
    assert claimed[0].leased_until == 61.0

    e = q.fail("w1", retry_after=30.0, now=10.0)
    assert e is not None
    assert e.scheduled_at == 40.0
    assert e.attempts == 1
    assert e.leased_until is None

    assert q.pop_due(now=20.0, lease_seconds=60.0) == []
    again = q.pop_due(now=40.0, lease_seconds=60.0)
    assert len(again) == 1
    assert again[0].id == "w1"


def test_pop_due_does_not_return_same_event_within_lease(tmp_path: Path) -> None:
    q = WakeQueue(tmp_path / "wake.jsonl")
    q.enqueue(
        WakeEvent(
            id="w1",
            chat_id="c1",
            reason="dispatch",
            priority=1,
            scheduled_at=0.0,
            created_at=time.time(),
        )
    )
    q.ready("w1", now=0.0)
    first = q.pop_due(now=1.0, lease_seconds=60.0)
    assert len(first) == 1
    assert q.pop_due(now=30.0, lease_seconds=60.0) == []
    expired = q.pop_due(now=61.0, lease_seconds=60.0)
    assert len(expired) == 1
    assert expired[0].id == "w1"


def test_complete_on_missing_or_removed_returns_none(tmp_path: Path) -> None:
    q = WakeQueue(tmp_path / "wake.jsonl")
    assert q.complete("w1") is None
    q.enqueue(
        WakeEvent(
            id="w1",
            chat_id="c1",
            reason="dispatch",
            priority=1,
            scheduled_at=0.0,
            created_at=time.time(),
        )
    )
    q.ready("w1")
    assert q.complete("w1") is not None
    assert q.complete("w1") is None


def test_pending_filters_by_chat(tmp_path: Path) -> None:
    q = WakeQueue(tmp_path / "wake.jsonl")
    q.enqueue(
        WakeEvent(
            id="w1",
            chat_id="c1",
            reason="dispatch",
            priority=1,
            scheduled_at=0.0,
            created_at=time.time(),
        )
    )
    q.enqueue(
        WakeEvent(
            id="w2",
            chat_id="c2",
            reason="dispatch",
            priority=1,
            scheduled_at=0.0,
            created_at=time.time(),
        )
    )
    assert len(q.pending(chat_id="c1")) == 1
    assert q.pending(chat_id="c1")[0].chat_id == "c1"


def test_pop_due_is_exclusive_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "wake.jsonl"
    q = WakeQueue(path)
    q.enqueue(
        WakeEvent(
            id="w1",
            chat_id="c1",
            reason="dispatch",
            priority=1,
            scheduled_at=0.0,
            created_at=time.time(),
            ready=True,
        )
    )

    result_queue: mp.Queue = mp.Queue()
    processes = [
        mp.Process(target=_pop_worker, args=(path, result_queue, 100.0, 60.0)) for _ in range(2)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join(timeout=5)

    results = [result_queue.get(timeout=1) for _ in range(len(processes))]
    all_ids = [event_id for r in results for event_id in r]
    assert all_ids.count("w1") == 1


def test_rehydration_sees_leased_event(tmp_path: Path) -> None:
    q = WakeQueue(tmp_path / "wake.jsonl")
    q.enqueue(
        WakeEvent(
            id="w1",
            chat_id="c1",
            reason="dispatch",
            priority=1,
            scheduled_at=0.0,
            created_at=time.time(),
            ready=True,
        )
    )
    due = q.pop_due(now=1.0, lease_seconds=60.0)
    assert len(due) == 1

    q2 = WakeQueue(tmp_path / "wake.jsonl")
    event = q2.get("w1")
    assert event is not None
    assert event.leased_until == 61.0
