"""Tests for dispatch tracking."""

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
