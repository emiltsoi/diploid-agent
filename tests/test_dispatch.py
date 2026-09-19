"""Tests for dispatch tracking."""

import time
from pathlib import Path

from diploid_agent.dispatch import DispatchStatus, DispatchStore


def test_dispatch_store_adds_and_completes(tmp_path: Path) -> None:
    store = DispatchStore()
    dispatch = store.add("chat-1", "session-1")
    assert dispatch.chat_id == "chat-1"
    assert dispatch.session_id == "session-1"
    assert dispatch.status == DispatchStatus.PENDING
    assert store.get(dispatch.id) == dispatch

    store.complete(dispatch.id, "result text")
    completed = store.get(dispatch.id)
    assert completed is not None
    assert completed.status == DispatchStatus.COMPLETED
    assert completed.result == "result text"


def test_dispatch_store_fails_dispatch() -> None:
    store = DispatchStore()
    dispatch = store.add("chat-1", "session-1")
    store.fail(dispatch.id, "error text")
    failed = store.get(dispatch.id)
    assert failed is not None
    assert failed.status == DispatchStatus.FAILED
    assert failed.result == "error text"


def test_dispatch_store_get_unknown_returns_none() -> None:
    store = DispatchStore()
    assert store.get("dispatch-does-not-exist") is None


def test_dispatch_store_persists_and_rehydrates(tmp_path: Path) -> None:
    path = tmp_path / "dispatches.jsonl"
    store = DispatchStore(path)
    dispatch = store.add("chat-1", "session-1", context="do work")

    # Simulate a process restart: load the same file into a new store.
    reloaded = DispatchStore(path)
    rehydrated = reloaded.get(dispatch.id)
    assert rehydrated is not None
    assert rehydrated.chat_id == "chat-1"
    assert rehydrated.session_id == "session-1"
    assert rehydrated.context == "do work"
    assert rehydrated.status == DispatchStatus.PENDING


def test_dispatch_store_completes_rehydrated_dispatch(tmp_path: Path) -> None:
    path = tmp_path / "dispatches.jsonl"
    store = DispatchStore(path)
    dispatch = store.add("chat-1", "session-1")

    reloaded = DispatchStore(path)
    reloaded.complete(dispatch.id, "finished")
    assert reloaded.get(dispatch.id).status == DispatchStatus.COMPLETED

    # Yet another process should also see the completed state.
    again = DispatchStore(path)
    assert again.get(dispatch.id).status == DispatchStatus.COMPLETED
    assert again.get(dispatch.id).result == "finished"


def test_dispatch_store_loads_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "does-not-exist.jsonl"
    store = DispatchStore(path)
    assert not path.exists()
    assert store.get("anything") is None


def test_dispatch_store_skips_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "dispatches.jsonl"
    valid = '{"id": "d-1", "chat_id": "c", "session_id": "s", "status": "pending"}\n'
    corrupt = "this is not json\n"
    incomplete = '{"id": "d-2", "chat_id": "c"}\n'
    path.write_text(valid + corrupt + incomplete)

    store = DispatchStore(path)
    assert len(store._dispatches) == 1
    assert store.get("d-1") is not None
    assert store.get("d-1").status == DispatchStatus.PENDING
    assert store.get("not-there") is None


def test_dispatch_store_path_none_stays_in_memory(tmp_path: Path) -> None:
    store = DispatchStore()
    dispatch = store.add("chat-1", "session-1")
    assert not (tmp_path / "dispatch_store.jsonl").exists()
    assert store.get(dispatch.id) is not None


def test_dispatch_store_evicts_old_terminal(tmp_path: Path) -> None:
    """Terminal dispatches past the retention window are swept on save."""
    path = tmp_path / "dispatches.jsonl"
    store = DispatchStore(path)
    old = store.add("chat-1", "s-1", started_at=time.time() - 8 * 86400)
    store.complete(old.id, "done")
    assert store.get(old.id) is None
    reloaded = DispatchStore(path)
    assert reloaded.get(old.id) is None


def test_dispatch_store_keeps_old_pending(tmp_path: Path) -> None:
    """Pending dispatches are never evicted regardless of age."""
    path = tmp_path / "dispatches.jsonl"
    store = DispatchStore(path)
    old = store.add("chat-1", "s-1", started_at=time.time() - 30 * 86400)
    assert store.get(old.id) is not None
    store.add("chat-1", "s-2")  # force another evict pass
    assert store.get(old.id) is not None


def test_dispatch_store_caps_terminal_count(tmp_path: Path, monkeypatch) -> None:
    """Over the cap, the oldest terminal dispatches are evicted first."""
    monkeypatch.setattr("diploid_agent.dispatch._TERMINAL_MAX_KEPT", 3)
    path = tmp_path / "dispatches.jsonl"
    store = DispatchStore(path)
    ids = []
    for i in range(6):
        d = store.add("chat-1", f"s-{i}", started_at=time.time() - (100 - i))
        store.fail(d.id, "x")
        ids.append(d.id)
    kept = [store.get(i) for i in ids]
    assert kept[:3] == [None, None, None]
    assert all(k is not None for k in kept[3:])
