"""Tests for the persistent, multi-process safe WakeQueue."""

import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

from diploid_agent.models import WakeEvent
from diploid_agent.runtime.wake_queue import WakeQueue

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


def _seed_event(q: WakeQueue, event_id: str = "w1") -> None:
    q.enqueue(
        WakeEvent(
            id=event_id,
            chat_id="c1",
            reason="dispatch",
            priority=1,
            scheduled_at=0.0,
            created_at=0.0,
            ready=True,
        )
    )


def test_fail_quarantines_past_max_retries(tmp_path: Path) -> None:
    q = WakeQueue(tmp_path / "wake.jsonl")
    _seed_event(q)

    q.fail("w1", retry_after=30.0, now=1.0, max_retries=2, last_error="first")
    q.fail("w1", retry_after=30.0, now=2.0, max_retries=2, last_error="second")
    assert q.get("w1") is not None

    e = q.fail("w1", retry_after=30.0, now=3.0, max_retries=2, last_error="third strike")
    assert e is not None
    assert e.attempts == 3
    assert q.get("w1") is None
    assert q.pending() == []

    rows = q.quarantined()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "w1"
    assert row["attempts"] == 3
    assert row["quarantined_at"] == 3.0
    assert row["last_error"] == "third strike"


def test_fail_max_retries_zero_quarantines_first_failure(tmp_path: Path) -> None:
    q = WakeQueue(tmp_path / "wake.jsonl")
    _seed_event(q)
    q.fail("w1", retry_after=30.0, now=1.0, max_retries=0, last_error="nope")
    assert q.get("w1") is None
    rows = q.quarantined()
    assert len(rows) == 1
    assert rows[0]["attempts"] == 1


def test_fail_none_max_retries_unbounded(tmp_path: Path) -> None:
    q = WakeQueue(tmp_path / "wake.jsonl")
    _seed_event(q)
    for i in range(20):
        e = q.fail("w1", retry_after=1.0, now=float(i), max_retries=None)
        assert e is not None
    assert q.get("w1") is not None
    assert q.get("w1").attempts == 20
    assert q.quarantined() == []


def test_fail_default_preserves_unbounded(tmp_path: Path) -> None:
    q = WakeQueue(tmp_path / "wake.jsonl")
    _seed_event(q)
    for i in range(20):
        q.fail("w1", retry_after=1.0, now=float(i))
    assert q.get("w1") is not None
    assert q.quarantined() == []


def test_quarantine_row_survives_rehydration(tmp_path: Path) -> None:
    q = WakeQueue(tmp_path / "wake.jsonl")
    _seed_event(q)
    q.fail("w1", retry_after=1.0, now=1.0, max_retries=0, last_error="poison")

    q2 = WakeQueue(tmp_path / "wake.jsonl")
    assert q2.get("w1") is None
    rows = q2.quarantined()
    assert len(rows) == 1
    assert rows[0]["last_error"] == "poison"


def test_quarantined_event_not_due(tmp_path: Path) -> None:
    q = WakeQueue(tmp_path / "wake.jsonl")
    _seed_event(q)
    q.fail("w1", retry_after=0.0, now=1.0, max_retries=0)
    assert q.pop_due(now=2.0, lease_seconds=60.0) == []
    assert q.due_count(now=2.0) == 0
    assert q.pending_count() == 0


def _digest(path: Path, sender: str = "peer-a", **kw) -> WakeQueue:
    q = WakeQueue(path / "wake.jsonl")
    q.enqueue(
        WakeEvent(
            id=f"mesh-digest:c1:{sender}",
            chat_id="c1",
            reason="mesh_digest",
            priority=1,
            scheduled_at=time.time() + 600,
            payload={"sender": sender, "messages": [{"display": "m0"}]},
            ready=True,
        )
    )
    return q


def test_merge_mesh_fields_parameterized(tmp_path: Path) -> None:
    """key_field/list_field let non-plan digests fold: an armed future mesh
    digest absorbs a second arrival by sender, messages concatenate."""
    q = _digest(tmp_path)
    incoming = WakeEvent(
        id="mesh-digest:c1:peer-a",
        chat_id="c1",
        reason="mesh_digest",
        priority=1,
        scheduled_at=time.time() + 600,
        payload={"sender": "peer-a", "messages": [{"display": "m1"}], "extra": "new"},
        ready=True,
    )
    merged = q.merge_into_ready(incoming, key="peer-a", key_field="sender", list_field="messages")
    assert merged is not None
    assert [m["display"] for m in merged.payload["messages"]] == ["m0", "m1"]
    assert merged.payload["extra"] == "new"  # non-list keys take newer value


def test_merge_mesh_wrong_key_does_not_fold(tmp_path: Path) -> None:
    q = _digest(tmp_path)
    incoming = WakeEvent(
        id="mesh-digest:c1:peer-b",
        chat_id="c1",
        reason="mesh_digest",
        priority=1,
        scheduled_at=time.time() + 600,
        payload={"sender": "peer-b", "messages": [{"display": "m1"}]},
        ready=True,
    )
    assert (
        q.merge_into_ready(incoming, key="peer-b", key_field="sender", list_field="messages")
        is None
    )


def test_merge_returns_event_not_bool(tmp_path: Path) -> None:
    """Return type changed to WakeEvent | None — planning.py's `if not
    merge(...)` truthiness still holds (None = enqueue, event = absorbed)."""
    q = _digest(tmp_path)
    incoming = WakeEvent(
        id="x",
        chat_id="c1",
        reason="mesh_digest",
        priority=1,
        scheduled_at=time.time(),
        payload={"sender": "peer-a", "messages": []},
        ready=True,
    )
    assert q.merge_into_ready(incoming, key="peer-a", key_field="sender", list_field="messages")


def test_fire_now_forces_due_no_attempts(tmp_path: Path) -> None:
    q = _digest(tmp_path)
    digest_id = "mesh-digest:c1:peer-a"
    fired = q.fire_now(digest_id)
    assert fired is not None
    assert fired.scheduled_at <= time.time()
    assert fired.attempts == 0  # not fail(): no attempts, no backoff
    due = q.pop_due(now=time.time(), lease_seconds=60.0)
    assert [e.id for e in due] == [digest_id]


def test_fire_now_skips_missing_and_not_ready(tmp_path: Path) -> None:
    q = WakeQueue(tmp_path / "wake.jsonl")
    assert q.fire_now("nonexistent") is None
    q.enqueue(WakeEvent(id="nr", chat_id="c1", reason="mesh_digest", priority=1, scheduled_at=0.0))
    assert q.fire_now("nr") is None  # not armed yet


def test_fire_matching_fires_class_not_id(tmp_path: Path) -> None:
    """Flush-by-scan: every armed unleased digest for the peer fires —
    including msg_id-suffixed ids and coexisting duplicates — while a leased
    or different-peer digest is left alone."""
    q = _digest(tmp_path)  # mesh-digest:c1:peer-a, armed, sender=peer-a
    q.enqueue(
        WakeEvent(
            id="mesh-digest:c1:peer-a:m2",
            chat_id="c1",
            reason="mesh_digest",
            priority=1,
            scheduled_at=time.time() + 600,
            payload={"sender": "peer-a", "messages": []},
            ready=True,
        )
    )
    q.enqueue(
        WakeEvent(
            id="mesh-digest:c1:peer-b:m3",
            chat_id="c1",
            reason="mesh_digest",
            priority=1,
            scheduled_at=time.time() + 600,
            payload={"sender": "peer-b", "messages": []},
            ready=True,
        )
    )
    fired = q.fire_matching(chat_id="c1", reason="mesh_digest", key_field="sender", key="peer-a")
    assert {e.id for e in fired} == {"mesh-digest:c1:peer-a", "mesh-digest:c1:peer-a:m2"}
    assert all(e.scheduled_at <= time.time() for e in fired)
    assert q.get("mesh-digest:c1:peer-b:m3").scheduled_at > time.time()


def test_fire_matching_leaves_leased_alone(tmp_path: Path) -> None:
    q = _digest(tmp_path)
    due = q.pop_due(now=time.time() + 601, lease_seconds=300.0)
    assert len(due) == 1
    assert (
        q.fire_matching(chat_id="c1", reason="mesh_digest", key_field="sender", key="peer-a") == []
    )


def test_mid_lease_park_does_not_overwrite(tmp_path: Path) -> None:
    """Leg-1 pin: enqueue of a msg_id-unique parked event while the same-peer
    digest is leased leaves both rows intact; complete() of the in-flight
    digest cannot eat the parked message."""
    q = _digest(tmp_path)  # armed digest id mesh-digest:c1:peer-a
    due = q.pop_due(now=time.time() + 601, lease_seconds=300.0)
    assert [e.id for e in due] == ["mesh-digest:c1:peer-a"]
    # Ingress parks under a unique id when merge refuses (leased).
    incoming = WakeEvent(
        id="mesh-digest:c1:peer-a:m9",
        chat_id="c1",
        reason="mesh_digest",
        priority=1,
        scheduled_at=time.time() + 600,
        payload={"sender": "peer-a", "messages": [{"display": "m9"}]},
        ready=True,
    )
    assert (
        q.merge_into_ready(incoming, key="peer-a", key_field="sender", list_field="messages")
        is None
    )
    q.enqueue(incoming)
    q.complete("mesh-digest:c1:peer-a")
    parked = q.get("mesh-digest:c1:peer-a:m9")
    assert parked is not None
    assert parked.payload["messages"] == [{"display": "m9"}]
