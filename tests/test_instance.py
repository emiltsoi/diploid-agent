"""Tests for the cross-process InstanceManager lock."""

import multiprocessing as mp
import threading
import time
from pathlib import Path
from typing import Any

from acp_fleet_harness.runtime.instance import InstanceManager

mp.set_start_method("spawn", force=True)


def _acquire_and_hold(
    sessions_root: Path,
    chat_id: str,
    instance_id: str,
    result_queue: Any,
    hold_event: Any,
    done_event: Any,
) -> None:
    im = InstanceManager(sessions_root, instance_id, ttl_seconds=2.0)
    acquired = im.acquire(chat_id)
    result_queue.put(acquired)
    if acquired:
        hold_event.set()
        done_event.wait(timeout=5)
        im.release(chat_id)


def _contend(
    sessions_root: Path,
    chat_id: str,
    instance_id: str,
    result_queue: Any,
    barrier: Any,
) -> None:
    im = InstanceManager(sessions_root, instance_id, ttl_seconds=2.0)
    barrier.wait(timeout=5)
    acquired = im.acquire(chat_id)
    result_queue.put((instance_id, acquired))
    if acquired:
        time.sleep(0.05)
        im.release(chat_id)


def _dead_child(
    sessions_root: Path,
    chat_id: str,
    instance_id: str,
    acquired_event: Any,
) -> None:
    im = InstanceManager(sessions_root, instance_id, ttl_seconds=0.1)
    if im.acquire(chat_id):
        acquired_event.set()
    # Intentionally do not release; the child exits and the fd/lock vanish.


def test_acquire_and_release(tmp_path: Path) -> None:
    im = InstanceManager(tmp_path, "i-1", ttl_seconds=2.0)
    assert im.acquire("chat-1") is True
    assert im.is_held("chat-1") is True
    assert im.is_ours("chat-1") is True
    im.release("chat-1")
    assert im.is_held("chat-1") is False


def test_heartbeat_keeps_lock_alive(tmp_path: Path) -> None:
    im = InstanceManager(tmp_path, "i-1", ttl_seconds=0.3, heartbeat_interval=0.05)
    assert im.acquire("chat-1") is True
    time.sleep(0.5)
    assert im.is_held("chat-1") is True
    assert im.is_ours("chat-1") is True
    im.release("chat-1")


def test_two_real_processes_cannot_hold_same_chat(tmp_path: Path) -> None:
    result_queue: mp.Queue = mp.Queue()
    barrier = mp.Barrier(2)
    p1 = mp.Process(
        target=_contend,
        args=(tmp_path, "chat-1", "i-1", result_queue, barrier),
    )
    p2 = mp.Process(
        target=_contend,
        args=(tmp_path, "chat-1", "i-2", result_queue, barrier),
    )
    p1.start()
    p2.start()
    p1.join(timeout=5)
    p2.join(timeout=5)

    results = [result_queue.get(timeout=1) for _ in range(2)]
    acquired = {instance_id: ok for instance_id, ok in results}
    assert any(acquired.values())
    assert not all(acquired.values())


def test_dead_child_releases_lock(tmp_path: Path) -> None:
    acquired_event = mp.Event()
    child = mp.Process(
        target=_dead_child,
        args=(tmp_path, "chat-1", "i-dead", acquired_event),
    )
    child.start()
    assert acquired_event.wait(timeout=5)
    child.join(timeout=5)
    assert not child.is_alive()

    # Give the OS a moment to reap the child's file descriptor lock.
    time.sleep(0.05)

    im = InstanceManager(tmp_path, "i-alive", ttl_seconds=2.0)
    assert im.acquire("chat-1") is True
    assert im.is_ours("chat-1") is True
    im.release("chat-1")


def test_second_attempt_by_same_manager_is_reentrant(tmp_path: Path) -> None:
    im = InstanceManager(tmp_path, "i-1", ttl_seconds=2.0)
    assert im.acquire("chat-1") is True
    assert im.acquire("chat-1") is True
    assert im.is_ours("chat-1") is True
    im.release("chat-1")
    assert im.is_held("chat-1") is False


def test_different_thread_in_same_process_cannot_reenter(tmp_path: Path) -> None:
    im = InstanceManager(tmp_path, "i-1", ttl_seconds=2.0)
    barrier = threading.Barrier(2)
    results: list[bool] = []

    def try_acquire() -> None:
        barrier.wait(timeout=2)
        results.append(im.acquire("chat-1"))

    t1 = threading.Thread(target=try_acquire)
    t2 = threading.Thread(target=try_acquire)
    t1.start()
    t2.start()
    t1.join(timeout=3)
    t2.join(timeout=3)

    assert len(results) == 2
    assert any(results)
    assert not all(results)
    if results[0]:
        im.release("chat-1")
    else:
        im.release("chat-1")
