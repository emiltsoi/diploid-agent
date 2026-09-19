"""Runtime lifecycle: startup, shutdown, restart notices, event-bus dispatch."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from diploid_agent.plan.models import TaskType
from diploid_agent.plugins.contexts import ShutdownContext
from diploid_agent.runtime.event_bus import Event
from diploid_agent.runtime.state import RuntimeState

if TYPE_CHECKING:
    from diploid_agent.config import Config
    from diploid_agent.memory import MemoryManager
    from diploid_agent.models import ChatState
    from diploid_agent.plan.manager import PlanManager
    from diploid_agent.plugins import PluginManager
    from diploid_agent.runtime.event_bus import EventBus
    from diploid_agent.runtime.instance import InstanceManager
    from diploid_agent.runtime.outbox import RuntimeOutbox
    from diploid_agent.runtime.planning import RuntimePlanning
    from diploid_agent.runtime.restart import RuntimeRestart
    from diploid_agent.runtime.store import ChatSessionStore
    from diploid_agent.runtime.subagent import RuntimeSubagent
    from diploid_agent.runtime.timer_service import TimerService
    from diploid_agent.runtime.typing import RuntimeTyping
    from diploid_agent.runtime.wake_queue import WakeQueue
    from diploid_agent.task.engine import TaskEngine
    from diploid_agent.transport.base import RuntimeAPI


logger = logging.getLogger(__name__)

# Replies that mean a wake could not run now but should be retried later.
_WAKE_RETRY_REPLIES = {
    "Chat is busy; wake re-enqueued.",
    "A turn is already in progress for this chat.",
    "Another instance is currently handling this chat.",
    "A turn is already in progress; continuation queued.",
    "A session operation is in progress for this chat.",
}

# Replies that should be dropped rather than delivered to the user. These are
# internal diagnostics, not conversation content.
_WAKE_DROP_REPLIES = {
    "Unknown or already completed wake event.",
}


class RuntimeLifecycle:
    """Start and stop the runtime's background services."""

    def __init__(
        self,
        *,
        state: RuntimeState,
        config: Config,
        lock: threading.RLock,
        event_bus: EventBus,
        wake_queue: WakeQueue | None,
        instance_manager: InstanceManager,
        task_engine: TaskEngine,
        timer_service: TimerService,
        cron_service: Any | None = None,
        typing: RuntimeTyping,
        restart: RuntimeRestart,
        outbox: RuntimeOutbox,
        plugins: PluginManager,
        chat_store: ChatSessionStore,
        memory_managers: dict[str, MemoryManager],
        store: dict[str, ChatState],
        ingress_handlers: dict[str, Any],
        instance_id: str,
        instance_started_at: float,
        plan_manager: PlanManager,
        planning: RuntimePlanning,
        subagent: RuntimeSubagent,
        runtime_api: RuntimeAPI,
    ) -> None:
        self._state = state
        self._config = config
        self._lock = lock
        self._event_bus = event_bus
        self._wake_queue = wake_queue
        self._instance_manager = instance_manager
        self._task_engine = task_engine
        self._timer_service = timer_service
        self._cron_service = cron_service
        self._typing = typing
        self._restart = restart
        self._outbox = outbox
        self._plugins = plugins
        self._chat_store = chat_store
        self._memory_managers = memory_managers
        self._store = store
        self._ingress_handlers = ingress_handlers
        self._instance_id = instance_id
        self._instance_started_at = instance_started_at
        self._plan_manager = plan_manager
        self._planning = planning
        self._subagent = subagent
        # Handed to the mesh ingress factory and used for the public ``wake``
        # entry point from the timer handler; not a general back-reference.
        self._runtime_api = runtime_api
        self._event_handlers: dict[str, Any] = {
            "timer.fired": self._handle_timer_fired,
            "task.completed": self._handle_task_completed,
            "task.failed": self._handle_task_failed,
        }

    def _load_mesh_ingress(self) -> None:
        """Load the configured mesh ingress handler if mesh is enabled."""
        from diploid_agent.transport.ingress import load_ingress_handler

        mesh = self._config.harness.mesh
        if not mesh.enabled:
            return
        try:
            handler = load_ingress_handler(mesh.ingress_module, runtime=self._runtime_api)
            self._ingress_handlers["mesh"] = handler
        except Exception:
            logger.exception("Failed to load mesh ingress handler: %s", mesh.ingress_module)

    def start(self) -> None:
        """Start background services. Idempotent."""
        if self._state.started:
            return
        self._state.started = True
        if not self._event_bus.running:
            self._event_bus.start()
        self._event_bus.subscribe(self._on_event)

        # Drop auto-continue wakes that were created by a previous process.
        # Queued user messages and other system wakes are kept, and the
        # conversation/session state used for resume is not touched.
        if self._wake_queue is not None:
            try:
                count = self._wake_queue.cancel_older_than(
                    self._instance_started_at,
                    reason="auto_continue",
                )
                if count:
                    logger.info(
                        "Cancelled %d stale auto-continue wake(s) on startup",
                        count,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to cancel stale auto-continue wakes",
                    exc_info=exc,
                )

        if self._config.harness.timer.enabled:
            self._timer_service.start()
        if self._cron_service is not None:
            self._cron_service.start()
        self._instance_manager.start_heartbeat()
        self._load_mesh_ingress()
        self._outbox._send_restart_notices()

    def send_restart_notices(self) -> None:
        """Notify recently active chats that the service has restarted."""
        self._outbox._send_restart_notices()

    def create_direct_notifier(self) -> Any:
        """Create a notifier that bypasses the outbox if possible."""
        return self._outbox._create_direct_notifier()

    def shutdown(self, drain_timeout: float = 120.0) -> None:
        """Drain active turns, notify plugins, and stop background workers."""
        self._state.started = False
        self._state.restart_draining.set()
        try:
            if not self._restart._wait_for_active_turns(drain_timeout):
                logger.warning(
                    "Shutdown drain cap (%.0fs) expired with turn(s) still active",
                    drain_timeout,
                )
        except Exception:
            logger.exception("Failed to drain active turns during shutdown")
        self._timer_service.stop()
        if self._cron_service is not None:
            self._cron_service.stop()
        self._typing.stop()
        try:
            self._event_bus.unsubscribe(self._on_event)
        except ValueError:
            pass
        self._instance_manager.stop_heartbeat()
        self._task_engine.shutdown(wait=False)
        self._event_bus.stop()
        now = time.time()
        for chat_id in list(self._store.keys()):
            record = self._chat_store.active_record(chat_id)
            self._plugins.on_shutdown(
                chat_id,
                ShutdownContext(
                    chat_id=chat_id,
                    record=record,
                    reason="shutdown",
                    now=now,
                    instance_id=self._instance_id,
                    instance_started_at=self._instance_started_at,
                ),
            )
        self._plugins.stop_all()

        with self._lock:
            managers = list(self._memory_managers.values())
            self._memory_managers.clear()
        for manager in managers:
            manager.close()

    # ------------------------------------------------------- event handlers

    def _on_event(self, event: Event) -> None:
        handler = self._event_handlers.get(event.type)
        if handler is not None:
            handler(event)

    def _handle_timer_fired(self, event: Event) -> None:
        payload = event.payload
        event_id = payload["event_id"]
        wake_event = self._wake_queue.get(event_id)
        retry_after = self._config.harness.timer.retry_after_seconds
        if wake_event and wake_event.payload:
            retry_after = wake_event.payload.get("retry_after", retry_after)
        try:
            result = self._runtime_api.wake(
                payload["chat_id"],
                event_id=event_id,
                reason=payload["reason"],
                silent=payload.get("silent", True),
            )
            if result.turn_number is None:
                # The wake did not result in a real turn. Retry the transient
                # cases; otherwise just drop without completing.
                if result.reply in _WAKE_RETRY_REPLIES:
                    self._wake_queue.fail(event_id, retry_after=retry_after)
                    return
                # Some wake results are internal diagnostics that should not be
                # surfaced as user-visible replies.
                if result.reply in _WAKE_DROP_REPLIES:
                    self._wake_queue.complete(event_id)
                    return
                # Some wake results are final messages (e.g. budget notice or
                # "dispatch already completed") that should still be delivered.
                if (result.reply or result.notice) and self._outbox._outbox_delivery_enabled:
                    self._outbox._deliver_chat_result(payload["chat_id"], result)
                self._wake_queue.complete(event_id)
                return
            if self._cron_service is not None and wake_event is not None:
                cron_job_id = (wake_event.payload or {}).get("cron_job_id")
                if cron_job_id:
                    self._cron_service.record_turn_delivery(cron_job_id)
            self._wake_queue.complete(event_id)
        except Exception:
            logger.exception("Wake failed for %s", event_id)
            self._wake_queue.fail(event_id, retry_after=retry_after)

    def _handle_task_completed(self, event: Event) -> None:
        payload = event.payload
        task = self._plan_manager.complete_task(
            payload["plan_id"],
            payload["task_id"],
            result=payload.get("result", ""),
            log=payload.get("log", ""),
            stop_reason=payload.get("stop_reason"),
            cancelled=payload.get("cancelled", False),
            partial=payload.get("partial", False),
            timed_out=payload.get("timed_out", False),
        )
        if task is None:
            logger.warning(
                "Task %s not found in plan %s for completion",
                payload.get("task_id"),
                payload.get("plan_id"),
            )
            return
        if task.type == TaskType.SUBAGENT:
            self._subagent._complete_subagent_task(task)
            return
        plan = self._plan_manager.get_plan(payload["plan_id"])
        if plan is None:
            return
        self._planning._enqueue_plan_task_wake(plan, task)
        self._planning._maybe_enqueue_plan_conclusion(plan)

    def _handle_task_failed(self, event: Event) -> None:
        payload = event.payload
        task = self._plan_manager.fail_task(
            payload["plan_id"],
            payload["task_id"],
            log=payload.get("log", payload.get("error", "")),
            stop_reason=payload.get("stop_reason"),
            cancelled=payload.get("cancelled", False),
            partial=payload.get("partial", False),
            timed_out=payload.get("timed_out", False),
        )
        if task is None:
            logger.warning(
                "Task %s not found in plan %s for failure",
                payload.get("task_id"),
                payload.get("plan_id"),
            )
            return
        if task.type == TaskType.SUBAGENT:
            self._subagent._complete_subagent_task(task)
            return
        plan = self._plan_manager.get_plan(payload["plan_id"])
        if plan is None:
            return
        self._planning._enqueue_plan_task_wake(plan, task)
        self._planning._maybe_enqueue_plan_conclusion(plan)
