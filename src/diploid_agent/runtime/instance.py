"""Per-chat singleton instance guard with real cross-process locking."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class InstanceManager:
    """Per-chat lockfiles with real cross-process ``fcntl.flock`` locks.

    The lock file lives at ``sessions/<chat_id>/instance.lock`` and contains
    the PID, instance id, and last heartbeat. ``acquire`` uses a non-blocking
    exclusive ``flock``; if another process holds the lock it returns ``False``.
    A daemon heartbeat thread rewrites the file while this process owns any
    chat locks, and ``release`` truncates the lock file to empty and closes
    the file descriptor.
    """

    def __init__(
        self,
        sessions_root: Path,
        instance_id: str,
        ttl_seconds: float = 60.0,
        heartbeat_interval: float | None = None,
    ) -> None:
        self._sessions_root = Path(sessions_root).expanduser()
        self._instance_id = instance_id
        self._ttl_seconds = ttl_seconds
        self._heartbeat_interval = heartbeat_interval or ttl_seconds / 2
        self._held: set[str] = set()
        self._holder: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()
        self._fds: dict[str, int] = {}
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_heartbeat = threading.Event()

    def _lock_path(self, chat_id: str) -> Path:
        chat_dir = self._sessions_root / chat_id.replace("/", "_")
        return chat_dir / "instance.lock"

    def _read_lock(self, chat_id: str) -> dict | None:
        path = self._lock_path(chat_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _read_fd(self, fd: int) -> dict | None:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 4096)
            if not raw:
                return None
            return json.loads(raw.decode())
        except (json.JSONDecodeError, OSError):
            return None

    def _write_lock_fd(self, fd: int) -> None:
        data = json.dumps(
            {
                "pid": os.getpid(),
                "instance_id": self._instance_id,
                "heartbeat": time.time(),
            }
        )
        encoded = data.encode()
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, encoded)

    def _pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def is_held(self, chat_id: str) -> bool:
        data = self._read_lock(chat_id)
        if data is None:
            return False
        age = time.time() - data.get("heartbeat", 0)
        if age > self._ttl_seconds:
            return False
        return self._pid_alive(data.get("pid", -1))

    def is_ours(self, chat_id: str) -> bool:
        data = self._read_lock(chat_id)
        if data is None:
            return False
        return data.get("instance_id") == self._instance_id

    def acquire(self, chat_id: str) -> bool:
        current_pid = os.getpid()
        current_tid = threading.get_ident()
        with self._lock:
            if chat_id in self._held:
                holder_pid, holder_tid = self._holder[chat_id]
                if holder_pid == current_pid and holder_tid == current_tid:
                    fd = self._fds[chat_id]
                    self._write_lock_fd(fd)
                    return True
                # Held by a different thread in this process; do not re-enter.
                return False

            path = self._lock_path(chat_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_RDWR | os.O_CREAT)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError):
                try:
                    os.close(fd)
                except OSError:
                    pass
                return False

            old_data = self._read_fd(fd)
            if old_data is not None:
                old_pid = old_data.get("pid")
                old_instance = old_data.get("instance_id")
                age = time.time() - old_data.get("heartbeat", 0)
                if (
                    old_pid != os.getpid()
                    and old_instance != self._instance_id
                    and old_pid is not None
                    and self._pid_alive(old_pid)
                    and age <= self._ttl_seconds
                ):
                    # A different, live instance still owns this chat.
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    return False

            self._write_lock_fd(fd)
            self._held.add(chat_id)
            self._holder[chat_id] = (current_pid, current_tid)
            self._fds[chat_id] = fd
            self._start_heartbeat()
            return True

    def release(self, chat_id: str) -> None:
        with self._lock:
            self._held.discard(chat_id)
            self._holder.pop(chat_id, None)
            fd = self._fds.pop(chat_id, None)
            if fd is not None:
                try:
                    os.ftruncate(fd, 0)
                    os.lseek(fd, 0, os.SEEK_SET)
                except OSError:
                    pass
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    os.close(fd)
                except OSError:
                    pass
            if not self._held:
                self._stop_heartbeat.set()

    def heartbeat(self) -> None:
        with self._lock:
            for chat_id in list(self._held):
                fd = self._fds.get(chat_id)
                if fd is None:
                    continue
                try:
                    self._write_lock_fd(fd)
                except OSError:
                    logger.exception("heartbeat failed for %s", chat_id)

    def start_heartbeat(self) -> None:
        """Start the background heartbeat thread."""
        self._start_heartbeat()

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._stop_heartbeat.clear()

        def _run() -> None:
            while not self._stop_heartbeat.wait(timeout=self._heartbeat_interval):
                self.heartbeat()

        self._heartbeat_thread = threading.Thread(target=_run, daemon=True)
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        """Signal the heartbeat thread to stop and wait for it to finish."""
        self._stop_heartbeat.set()
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=self._heartbeat_interval + 1.0)
        self._heartbeat_thread = None
