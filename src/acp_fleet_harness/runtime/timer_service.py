"""Background wake queue consumer that posts timer events to the event bus."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from acp_fleet_harness.config import TimerConfig
from acp_fleet_harness.runtime.event_bus import Event

if TYPE_CHECKING:
    from acp_fleet_harness.runtime.agent_runtime import AgentRuntime

logger = logging.getLogger(__name__)


class TimerService:
    """Background thread that polls the wake queue and fires due chat events."""

    def __init__(
        self,
        runtime: AgentRuntime,
        config: TimerConfig,
    ) -> None:
        self._runtime = runtime
        self._config = config
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._config.interval_seconds))
            self._thread = None

    @property
    def running(self) -> bool:
        return self._running

    def _run(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception:
                logger.exception("Timer tick failed")
            time.sleep(self._config.interval_seconds)

    def _tick(self) -> None:
        if not self._config.enabled:
            return
        now = time.time()
        for event in self._runtime.wake_queue.pop_due(
            now=now,
            lease_seconds=self._config.lease_seconds,
        ):
            self._runtime.event_bus.post(
                Event(
                    type="timer.fired",
                    payload={
                        "event_id": event.id,
                        "chat_id": event.chat_id,
                        "reason": event.reason,
                        "silent": event.silent,
                    },
                )
            )
