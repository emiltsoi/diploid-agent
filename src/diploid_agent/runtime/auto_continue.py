"""Auto-continue suppression state."""

from __future__ import annotations

import threading
import time


class RuntimeAutoContinue:
    """Track per-chat and global auto-continue suppression."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._suppressed: dict[str, float] = {}
        self._globally_suppressed_until: float = 0.0

    def suppress(self, chat_id: str | None = None, seconds: float = 300.0) -> None:
        """Suppress auto-continue for a chat or globally for a number of seconds."""
        until = time.time() + seconds
        with self._lock:
            if chat_id is None:
                self._globally_suppressed_until = until
            else:
                self._suppressed[chat_id] = until

    def is_suppressed(self, chat_id: str) -> bool:
        """Return True if auto-continue should be suppressed for this chat."""
        now = time.time()
        with self._lock:
            if now < self._globally_suppressed_until:
                return True
            until = self._suppressed.get(chat_id, 0)
            if now < until:
                return True
            self._suppressed.pop(chat_id, None)
            return False
