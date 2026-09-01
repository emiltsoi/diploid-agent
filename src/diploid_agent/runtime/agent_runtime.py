"""AgentRuntime: service container and non-turn public API."""

from __future__ import annotations

import functools
import json
import logging
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from diploid_agent.config import (
    Config,
    NotificationsConfig,
    PluginConfig,
    TaskConfig,
    TelegramConfig,
    TimerConfig,
    WakerConfig,
)
from diploid_agent.context import ContextBuilder
from diploid_agent.dispatch import Dispatch, DispatchStatus, DispatchStore
from diploid_agent.engine import AgentEngine, TurnResult, build_engine
from diploid_agent.engine.router import ModelRoute, ModelRouter
from diploid_agent.mcp import McpManager
from diploid_agent.memory import MemoryManager, RecallResult
from diploid_agent.metrics import MetricsCollector
from diploid_agent.models import (
    ActiveTurn,
    ChatResult,
    ChatState,
    RuntimeStatus,
    SessionRecord,
    WakeEvent,
)
from diploid_agent.persona_composer import PersonaPrompt
from diploid_agent.plan.manager import PlanManager
from diploid_agent.plan.models import Plan, PlanStatus, Task, TaskStatus, TaskType
from diploid_agent.plugin_incidents import PluginIncidentStore
from diploid_agent.plugins import PluginManager
from diploid_agent.plugins.contexts import (
    PromoteContext,
    RetainContext,
    ShutdownContext,
)
from diploid_agent.runtime.config_manager import RuntimeConfigManager
from diploid_agent.runtime.event_bus import Event, EventBus
from diploid_agent.runtime.instance import InstanceManager
from diploid_agent.runtime.mcp_skills import RuntimeMcpSkills
from diploid_agent.runtime.metrics import RuntimeMetrics
from diploid_agent.runtime.outbox import RuntimeOutbox
from diploid_agent.runtime.plugins import RuntimePlugins
from diploid_agent.runtime.prompts import RuntimePrompts
from diploid_agent.runtime.store import ChatSessionStore
from diploid_agent.runtime.subagent import RuntimeSubagent
from diploid_agent.runtime.timer_service import TimerService
from diploid_agent.runtime.turn_controller import TurnController
from diploid_agent.runtime.wake_queue import WakeQueue
from diploid_agent.skills import SkillManager
from diploid_agent.task.engine import TaskEngine
from diploid_agent.transport.base import RuntimeAPI

logger = logging.getLogger(__name__)

# Replies that mean a wake could not run now but should be retried later.
_WAKE_RETRY_REPLIES = {
    "Chat is busy; wake re-enqueued.",
    "A turn is already in progress for this chat.",
    "Another instance is currently handling this chat.",
    "A turn is already in progress; continuation queued.",
}


def _locked(method):
    """Run a AgentRuntime method under its RLock."""

    @functools.wraps(method)
    def wrapper(self: AgentRuntime, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class AgentRuntime(RuntimeAPI):
    """Persistent chat runtime backed by Devin ACP."""

    def __init__(self, config: Config):
        self.config = config
        self.sessions_root = Path(config.harness.sessions_root).expanduser()
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.store_path = Path(config.harness.session_store_path).expanduser()
        self.metrics = MetricsCollector()
        self.engine = self._create_engine(metrics=self.metrics)
        self._lock = threading.RLock()
        self._config_manager = RuntimeConfigManager(self)
        self._chat_store = ChatSessionStore(self)
        self._store = self._chat_store._store
        self._active_turns: dict[str, ActiveTurn] = {}
        self._active_chat_skills: dict[str, set[str]] = {}
        self._runtime_metrics = RuntimeMetrics(self)
        self._memory_managers: dict[str, MemoryManager] = {}
        self.instance_id = f"harness-{uuid.uuid4().hex[:12]}"
        self.instance_started_at = time.time()
        self._router = ModelRouter(config)

        # Load external plugin search paths before PluginManager imports anything.
        for plugin_path in self.config.harness.plugin_paths:
            if plugin_path.exists() and str(plugin_path) not in sys.path:
                sys.path.append(str(plugin_path))

        self._chat_store._load_store()
        dispatch_store_path = Path(config.harness.dispatch_store_path).expanduser()
        self.dispatch_store = DispatchStore(dispatch_store_path)

        wake_store_path = Path(config.harness.wake_store_path).expanduser()
        self.wake_queue = WakeQueue(wake_store_path)

        plan_root = Path(config.harness.plan.root).expanduser()
        plan_root.mkdir(parents=True, exist_ok=True)
        self._config_manager._load_runtime_overrides()
        self.event_bus = EventBus()
        self.event_bus.start()
        self.plan_manager = PlanManager(plan_root)
        self.task_engine = TaskEngine(
            self.plan_manager,
            self.event_bus,
            engine=self.engine,
            config=self.config,
            task_config=config.harness.task,
            on_task_start=self._on_task_started,
            on_task_done=self._on_task_done,
            on_service_restart=self._on_service_restart,
        )

        self.timer_service = TimerService(
            self,
            config=self.config.harness.timer,
        )

        self._event_handlers: dict[str, Callable[[Event], None]] = {
            "timer.fired": self._handle_timer_fired,
            "task.completed": self._handle_task_completed,
            "task.failed": self._handle_task_failed,
        }
        self._plan_conclusion_enqueued: set[str] = set()
        self._typing_counts: dict[str, int] = {}
        self._typing_threads: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self._typing_lock = threading.Lock()
        self._started = False

        self._outbox = RuntimeOutbox(self)

        self.instance_manager = InstanceManager(
            self.sessions_root,
            self.instance_id,
            ttl_seconds=config.harness.instance_ttl_seconds,
        )

        if config.harness.session_prune_enabled:
            self._prune_all()
        self._runtime_metrics._rehydrate_metrics()

        # Durable record of plugin incidents (sandbox, lifecycle, health, watchdog).
        self._incidents = PluginIncidentStore(self.store_path.parent / "plugin-incidents.jsonl")

        # Rate-limit ACP-child-initiated service restarts.
        self._last_service_restart_at: float = 0.0
        self._service_restart_cooldown_seconds = 60.0

        # Per-chat and global auto-continue suppression (e.g., before a restart).
        self._auto_continue_suppressed: dict[str, float] = {}
        self._auto_continue_globally_suppressed_until: float = 0.0

        # Plugins can declare MCP servers and skills; add them before McpManager.
        self._plugins = PluginManager(
            plugins=list(self.config.harness.plugins),
            sessions_root=self.sessions_root,
            instance_id=self.instance_id,
            instance_started_at=self.instance_started_at,
            dispatch_store=self.dispatch_store,
            runtime=self,
            incident_store=self._incidents,
        )
        failed = self._plugins.validate_all()
        if failed:
            for name in failed:
                self._incidents.record(
                    plugin=name,
                    phase="startup",
                    error=f"Startup validation failed for {name}",
                    action="disabled",
                )
            self._plugins.disable_plugins(set(failed))
            self.config.harness.plugins = self._plugins._plugins
            self._config_manager._save_runtime_overrides()
        self._plugin_mcp_server_names: set[str] = set()

        # Ingress handlers for pluggable transport protocols (e.g. mesh).
        self._ingress_handlers: dict[str, Any] = {}

        self.mcp = McpManager(config)
        self.skills = SkillManager(
            personas_root=Path(self.config.persona.profile_root).parent,
            shared_root=self.config.harness.skills.shared_root,
            chat_cwd_root=self.sessions_root,
        )
        self._mcp_skills = RuntimeMcpSkills(self)
        self._runtime_plugins = RuntimePlugins(self)
        self._runtime_plugins._register_plugin_mcp_servers()

        # Prompt assembly is delegated to a dedicated builder.
        self.context_builder = ContextBuilder(
            self.config,
            self._plugins,
            self._memory_manager,
            self.skills,
            self._active_skill_names,
        )
        self.context_builder.metrics = self._runtime_metrics._per_chat_metrics

        self._prompts = RuntimePrompts(self)
        self._subagent = RuntimeSubagent(self)

        self.turn_controller = TurnController(self)

        self.notifier = self._create_notifier()

    def _create_engine(self, metrics: MetricsCollector | None = None) -> AgentEngine:
        api_key = None
        if self.config.secrets:
            api_key = self.config.secrets.windsurf_api_key
        return build_engine(
            self.config.engine,
            api_key=api_key,
            metrics=metrics,
            service_name=f"{self.config.persona.name}.service",
            on_service_restart=self._on_service_restart,
        )

    @property
    def client(self) -> AgentEngine:
        """Backward-compatible alias for the engine."""
        return self.engine

    @client.setter
    def client(self, value: AgentEngine) -> None:
        self.engine = value

    @_locked
    def _on_service_restart(self, service: str, reason: str) -> None:
        """Handle a service restart request from the ACP child.

        Instead of letting the child kill the harness directly, we schedule a
        short-delayed `systemd-run` that restarts the service after the current
        turn has a chance to finish and the final reply is delivered.
        """
        now = time.time()
        if now - self._last_service_restart_at < self._service_restart_cooldown_seconds:
            logger.warning(
                "Ignoring repeat restart request for %s (cooldown active)",
                service,
            )
            return
        self._last_service_restart_at = now

        logger.warning(
            "ACP child requested restart of %s (reason: %s); scheduling graceful restart",
            service,
            reason,
        )

        # Cancel any pending auto-continue wakes so the restart does not loop.
        if self.wake_queue is not None:
            try:
                self.wake_queue.cancel(reason="auto_continue")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to cancel auto-continue wakes before restart: %s", exc)

        self.suppress_auto_continue()

        # Record the incident for observability.
        if self._incidents is not None:
            try:
                self._incidents.record(
                    plugin="self_management",
                    phase="graceful_restart",
                    error=f"ACP child requested restart of {service}: {reason}",
                    action="scheduled",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to record restart incident: %s", exc)

        # Schedule the actual service restart with a short delay so the in-flight
        # reply can be sent before the process goes down.
        self._schedule_systemd_restart(service, delay=10.0, chat_id=None, reason=reason)

    def _schedule_systemd_restart(
        self,
        service: str,
        delay: float,
        chat_id: str | None,
        reason: str,
    ) -> None:
        """Run a short-delayed systemd-run that restarts the named service."""

        def _do_restart() -> None:
            try:
                subprocess.Popen(
                    [
                        "systemd-run",
                        "--user",
                        f"--on-active={delay}s",
                        "--timer-property=AccuracySec=1s",
                        "/usr/bin/systemctl",
                        "--user",
                        "restart",
                        service,
                    ],
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if chat_id is not None:
                    logger.info(
                        "Scheduled graceful restart of %s in %ss (from chat %s)",
                        service,
                        delay,
                        chat_id,
                    )
                else:
                    logger.info("Scheduled graceful restart of %s in %ss", service, delay)
            except Exception:
                logger.exception("Failed to schedule graceful restart of %s", service)

        # Run the scheduler in its own thread so the ACP control listener is not
        # blocked.
        threading.Thread(target=_do_restart, daemon=True).start()

    def suppress_auto_continue(self, chat_id: str | None = None, seconds: float = 300.0) -> None:
        """Suppress auto-continue for a chat or globally for a number of seconds."""
        until = time.time() + seconds
        with self._lock:
            if chat_id is None:
                self._auto_continue_globally_suppressed_until = until
            else:
                self._auto_continue_suppressed[chat_id] = until

    def is_auto_continue_suppressed(self, chat_id: str) -> bool:
        """Return True if auto-continue should be suppressed for this chat."""
        now = time.time()
        with self._lock:
            if now < self._auto_continue_globally_suppressed_until:
                return True
            until = self._auto_continue_suppressed.get(chat_id, 0)
            if now < until:
                return True
            self._auto_continue_suppressed.pop(chat_id, None)
            return False

    def _create_notifier(self):
        """Create the runtime's configured notifier."""
        return self._outbox._create_notifier()

    @property
    def _outbox_delivery_enabled(self) -> bool:
        return self._outbox._outbox_delivery_enabled

    def _enqueue_outbox(self, chat_id: str, chat_result: ChatResult) -> None:
        """Put a final ChatResult in the outbox for the transport to deliver."""
        self._outbox._enqueue_outbox(chat_id, chat_result)

    def _safe_notifier_send(self, chat_id: str, text: str, notifier: Any = None) -> None:
        """Send a notification, swallowing exceptions and logging them."""
        self._outbox._safe_notifier_send(chat_id, text, notifier=notifier)

    def _deliver_chat_result(self, chat_id: str, chat_result: ChatResult) -> None:
        """Send a final ChatResult through the configured delivery channel."""
        self._outbox._deliver_chat_result(chat_id, chat_result)

    def outbox_pop(
        self,
        chat_id: str | None = None,
        wait: float = 0.0,
    ) -> ChatResult | None:
        """Return the next ChatResult for a chat, blocking up to ``wait`` seconds."""
        return self._outbox.outbox_pop(chat_id, wait=wait)

    def _call_unlocked(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Release the RLock while running a long call, then reacquire."""
        do_release = self._lock._is_owned()
        if do_release:
            self._lock.release()
        try:
            return fn(*args, **kwargs)
        finally:
            if do_release:
                self._lock.acquire()

    def call_engine_unlocked(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Public wrapper that releases the runtime RLock around a call.

        Plugins can use this from gate hooks (e.g. ``before_turn``) to call
        the engine without blocking the runtime.
        """
        return self._call_unlocked(fn, *args, **kwargs)

    # ---------------------------------------------------------------- load/save

    def _load_store(self) -> None:
        self._chat_store._load_store()

    def _append_record(self, record: SessionRecord) -> None:
        self._chat_store._append_record(record)

    def _compact_store(self) -> None:
        self._chat_store._compact_store()

    # ---------------------------------------------------------------- session dirs

    def _chat_dir(self, chat_id: str) -> Path:
        return self._chat_store._chat_dir(chat_id)

    def _archive_dir(self, chat_id: str, session_number: int) -> Path:
        return self._chat_store._archive_dir(chat_id, session_number)

    def _durable_file_names(self) -> set[str]:
        return self._chat_store._durable_file_names()

    def _copy_session_dir(self, source: Path, target: Path) -> None:
        self._chat_store._copy_session_dir(source, target)

    def _archive_active_session(self, chat_id: str, record: SessionRecord) -> None:
        """Copy the active directory into the archive for `record`."""
        self._chat_store._archive_active_session(chat_id, record)

    def _clear_active_session(self, chat_id: str) -> None:
        self._chat_store._clear_active_session(chat_id)

    # ---------------------------------------------------------------- active state

    def _chat_state(self, chat_id: str) -> ChatState:
        return self._chat_store._chat_state(chat_id)

    def _active_record(self, chat_id: str) -> SessionRecord | None:
        return self._chat_store._active_record(chat_id)

    def _next_session_number(self, chat_id: str) -> int:
        return self._chat_store._next_session_number(chat_id)

    def _generate_label(self, chat_id: str, user_message: str) -> str:
        """Auto-generate a short label from the first user message."""
        return self._chat_store._generate_label(chat_id, user_message)

    # ---------------------------------------------------------------- metrics

    @property
    def _per_chat_metrics(self) -> dict[str, dict[str, Any]]:
        return self._runtime_metrics._per_chat_metrics

    @property
    def _global_metrics(self) -> dict[str, Any]:
        return self._runtime_metrics._global_metrics

    @property
    def _recent_turns(self) -> deque[dict[str, Any]]:
        return self._runtime_metrics._recent_turns

    def _rehydrate_metrics(self) -> None:
        return self._runtime_metrics._rehydrate_metrics()

    def _record_turn_metrics(
        self,
        chat_id: str,
        turn_number: int,
        model: str,
        usage: dict[str, Any] | None,
        latency_seconds: float,
    ) -> dict[str, Any]:
        return self._runtime_metrics._record_turn_metrics(
            chat_id, turn_number, model, usage, latency_seconds
        )

    def _metrics_context_for_prompt(self, chat_id: str, compact: bool = False) -> str | None:
        return self._runtime_metrics._metrics_context_for_prompt(chat_id, compact=compact)

    def mcp_list(self, chat_id: str) -> str:
        return self._mcp_skills.mcp_list(chat_id)

    def mcp_enable(self, chat_id: str, name: str) -> str:
        return self._mcp_skills.mcp_enable(chat_id, name)

    def mcp_disable(self, chat_id: str, name: str) -> str:
        return self._mcp_skills.mcp_disable(chat_id, name)

    def skill_list(self, chat_id: str) -> str:
        return self._mcp_skills.skill_list(chat_id)

    def skill_enable(self, chat_id: str, name: str) -> str:
        return self._mcp_skills.skill_enable(chat_id, name)

    def skill_disable(self, chat_id: str, name: str) -> str:
        return self._mcp_skills.skill_disable(chat_id, name)

    def skill_create(self, chat_id: str, name: str, content: str) -> str:
        return self._mcp_skills.skill_create(chat_id, name, content)

    def get_metrics(self, chat_id: str | None = None) -> dict[str, Any]:
        """Return cumulative metrics for a chat or globally."""
        return self._runtime_metrics.get_metrics(chat_id)

    def get_prometheus_metrics(self) -> str:
        """Return metrics in Prometheus exposition format."""
        return self._runtime_metrics.get_prometheus_metrics()

    def health(self) -> dict[str, Any]:
        """Return the current health of the runtime and its dependencies."""
        return self._runtime_metrics.health()

    def _hindsight_health(self) -> bool:
        """Probe the Hindsight backend health endpoint."""
        return self._runtime_metrics._hindsight_health()

    # ---------------------------------------------------------------- helpers

    def _active_mcp_server_names(self, chat_id: str) -> list[str]:
        return self._mcp_skills._active_mcp_server_names(chat_id)

    def _active_mcp_servers(self, chat_id: str) -> list[dict[str, Any]]:
        return self._mcp_skills._active_mcp_servers(chat_id)

    def _default_active_skills(self) -> set[str]:
        return self._mcp_skills._default_active_skills()

    def _active_skill_names(self, chat_id: str) -> set[str]:
        return self._mcp_skills._active_skill_names(chat_id)

    def match_and_activate_skills(self, chat_id: str, user_message: str) -> set[str]:
        return self._mcp_skills.match_and_activate_skills(chat_id, user_message)

    def _memory_manager(self, chat_id: str) -> MemoryManager:
        if chat_id not in self._memory_managers:
            self._memory_managers[chat_id] = MemoryManager(
                config=self.config.harness.memory,
                persona=self.config.persona,
                sessions_root=self.sessions_root,
                chat_id=chat_id,
                devin_client=self.engine,
                metrics=self.metrics,
            )
        return self._memory_managers[chat_id]

    def _register_plugin_mcp_servers(self) -> None:
        self._runtime_plugins._register_plugin_mcp_servers()

    def plugin_event(
        self,
        chat_id: str,
        plugin: str,
        *,
        event: str | None = None,
        raw_args: str | None = None,
        **params: Any,
    ) -> ChatResult:
        return self._runtime_plugins.plugin_event(
            chat_id, plugin, event=event, raw_args=raw_args, **params
        )

    def plugin_list(self, chat_id: str) -> list[dict[str, Any]]:
        return self._runtime_plugins.plugin_list(chat_id)

    def plugin_set_enabled(self, chat_id: str, name: str, enabled: bool) -> ChatResult:
        return self._runtime_plugins.plugin_set_enabled(chat_id, name, enabled)

    def plugin_reload(self, chat_id: str, name: str) -> ChatResult:
        return self._runtime_plugins.plugin_reload(chat_id, name)

    def plugin_add(self, config: PluginConfig) -> ChatResult:
        return self._runtime_plugins.plugin_add(config)

    def plugin_remove(self, name: str) -> ChatResult:
        return self._runtime_plugins.plugin_remove(name)

    def plugin_toggle(
        self, name: str, enabled: bool, chat_id: str | None = None
    ) -> ChatResult:
        return self._runtime_plugins.plugin_toggle(name, enabled, chat_id=chat_id)

    def plugin_rollback(self, steps: int = 1) -> ChatResult:
        return self._runtime_plugins.plugin_rollback(steps)

    def plugin_sandbox(
        self, module: str, plugin: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._runtime_plugins.plugin_sandbox(module, plugin)

    def plugin_create(
        self,
        name: str,
        module: str | None = None,
        prompt_slot: str = "self_state",
        state_file: str | None = None,
        mcp_server: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._runtime_plugins.plugin_create(
            name,
            module=module,
            prompt_slot=prompt_slot,
            state_file=state_file,
            mcp_server=mcp_server,
            config=config,
        )

    def incidents(self) -> list[dict[str, Any]]:
        return self._runtime_plugins.incidents()

    def incidents_for_plugin(self, name: str) -> list[dict[str, Any]]:
        return self._runtime_plugins.incidents_for_plugin(name)

    def record_incident(
        self,
        plugin: str,
        phase: str,
        error: str,
        action: str = "",
        chat_id: str = "",
    ) -> ChatResult:
        return self._runtime_plugins.record_incident(
            plugin, phase, error, action=action, chat_id=chat_id
        )

    def start(self) -> None:
        """Start background services. Idempotent."""
        if self._started:
            return
        self._started = True
        if not self.event_bus.running:
            self.event_bus.start()
        self.event_bus.subscribe(self._on_event)

        # Drop auto-continue wakes that were created by a previous process.
        # Queued user messages and other system wakes are kept, and the
        # conversation/session state used for resume is not touched.
        if self.wake_queue is not None:
            try:
                count = self.wake_queue.cancel_older_than(
                    self.instance_started_at,
                    reason="auto_continue",
                )
                if count:
                    logger.info(
                        "Cancelled %d stale auto-continue wake(s) on startup",
                        count,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to cancel stale auto-continue wakes: %s", exc)

        if self.config.harness.timer.enabled:
            self.timer_service.start()
        self.instance_manager.start_heartbeat()
        self._load_mesh_ingress()
        self._outbox._send_restart_notices()

    def _send_restart_notices(self) -> None:
        """Notify recently active chats that the service has restarted."""
        self._outbox._send_restart_notices()

    def _create_direct_notifier(self):
        """Create a notifier that bypasses the outbox if possible."""
        return self._outbox._create_direct_notifier()

    def shutdown(self) -> None:
        """Notify all plugins and stop background workers."""
        self._started = False
        if hasattr(self, "timer_service"):
            self.timer_service.stop()
        with self._typing_lock:
            threads = list(self._typing_threads.values())
            self._typing_counts.clear()
            self._typing_threads.clear()
        for _, stop_event in threads:
            stop_event.set()
        for thread, _ in threads:
            thread.join(timeout=1.0)
        try:
            self.event_bus.unsubscribe(self._on_event)
        except ValueError:
            pass
        if hasattr(self, "instance_manager"):
            self.instance_manager.stop_heartbeat()
        if hasattr(self, "task_engine"):
            self.task_engine.shutdown(wait=False)
        if hasattr(self, "event_bus"):
            self.event_bus.stop()
        now = time.time()
        for chat_id in list(self._store.keys()):
            record = self._active_record(chat_id)
            self._plugins.on_shutdown(
                chat_id,
                ShutdownContext(
                    chat_id=chat_id,
                    record=record,
                    reason="shutdown",
                    now=now,
                    instance_id=self.instance_id,
                    instance_started_at=self.instance_started_at,
                ),
            )
        self._plugins.stop_all()

        with self._lock:
            managers = list(self._memory_managers.values())
            self._memory_managers.clear()
        for manager in managers:
            manager.close()

    @property
    def _runtime_overrides_path(self) -> Path:
        return self._config_manager._runtime_overrides_path

    def get_status(self) -> RuntimeStatus:
        """Return the current runtime daemon status."""
        return self._config_manager.get_status()

    def get_task_config(self) -> TaskConfig:
        """Return the current live task configuration."""
        return self._config_manager.get_task_config()

    def update_task_config(self, task_config: TaskConfig) -> str:
        """Update the live task configuration."""
        return self._config_manager.update_task_config(task_config)

    def _save_runtime_overrides(self) -> bool:
        """Persist runtime config overrides to disk atomically."""
        return self._config_manager._save_runtime_overrides()

    def get_notifications_config(self) -> NotificationsConfig:
        """Return the current live notifications configuration."""
        return self._config_manager.get_notifications_config()

    def update_notifications_config(self, notifications_config: NotificationsConfig) -> str:
        """Update the live notifications configuration."""
        return self._config_manager.update_notifications_config(notifications_config)

    def get_timer_config(self) -> TimerConfig:
        """Return the current live timer configuration."""
        return self._config_manager.get_timer_config()

    def update_timer_config(self, timer_config: TimerConfig) -> str:
        """Update the live timer configuration."""
        return self._config_manager.update_timer_config(timer_config)

    def get_waker_config(self) -> WakerConfig:
        """Return the current live waker configuration."""
        return self._config_manager.get_waker_config()

    def update_waker_config(self, waker_config: WakerConfig) -> str:
        """Update the live waker configuration."""
        return self._config_manager.update_waker_config(waker_config)

    def get_telegram_config(self) -> TelegramConfig:
        """Return the current live Telegram configuration."""
        return self._config_manager.get_telegram_config()

    def update_telegram_config(self, telegram_config: TelegramConfig) -> str:
        """Update the live Telegram configuration."""
        return self._config_manager.update_telegram_config(telegram_config)

    def get_plugins_config(self) -> list[PluginConfig]:
        """Return the current live plugin list."""
        return self._config_manager.get_plugins_config()

    def update_plugins_config(self, plugins: list[PluginConfig]) -> str:
        """Update the live plugin list."""
        return self._config_manager.update_plugins_config(plugins)

    def get_config(self) -> dict[str, Any]:
        """Return the current live runtime configuration (excluding secrets)."""
        return self._config_manager.get_config()

    def update_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a partial runtime configuration update."""
        return self._config_manager.update_config(patch)

    def _on_event(self, event: Event) -> None:
        handler = self._event_handlers.get(event.type)
        if handler is not None:
            handler(event)

    def _typing_heartbeat(self, chat_id: str, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self.notifier.typing(chat_id)
            except Exception:
                logger.exception("Typing heartbeat for %s failed", chat_id)
            if stop_event.wait(4.0):
                break

    def _on_task_started(self, plan_id: str, task: Task) -> None:
        if task.chat_id is None:
            return
        with self._typing_lock:
            count = self._typing_counts.get(task.chat_id, 0)
            self._typing_counts[task.chat_id] = count + 1
            if count == 0:
                stop_event = threading.Event()
                thread = threading.Thread(
                    target=self._typing_heartbeat,
                    args=(task.chat_id, stop_event),
                    daemon=True,
                    name=f"typing-{task.chat_id}",
                )
                self._typing_threads[task.chat_id] = (thread, stop_event)
                thread.start()

    def _on_task_done(self, plan_id: str, task: Task, outcome: tuple[str, str, int]) -> None:
        if task.chat_id is None:
            return
        with self._typing_lock:
            count = self._typing_counts.get(task.chat_id, 0)
            if count <= 0:
                return
            count -= 1
            self._typing_counts[task.chat_id] = count
            if count == 0:
                entry = self._typing_threads.pop(task.chat_id, None)
                if entry is not None:
                    _, stop_event = entry
                    stop_event.set()

    def _handle_timer_fired(self, event: Event) -> None:
        payload = event.payload
        event_id = payload["event_id"]
        wake_event = self.wake_queue.get(event_id)
        retry_after = self.config.harness.timer.retry_after_seconds
        if wake_event and wake_event.payload:
            retry_after = wake_event.payload.get("retry_after", retry_after)
        try:
            result = self.wake(
                payload["chat_id"],
                event_id=event_id,
                reason=payload["reason"],
                silent=payload.get("silent", True),
            )
            if result.turn_number is None:
                # The wake did not result in a real turn. Retry the transient
                # cases; otherwise just drop without completing.
                if result.reply in _WAKE_RETRY_REPLIES:
                    self.wake_queue.fail(event_id, retry_after=retry_after)
                    return
                # Some wake results are final messages (e.g. budget notice or
                # "dispatch already completed") that should still be delivered.
                if (result.reply or result.notice) and self._outbox_delivery_enabled:
                    self._deliver_chat_result(payload["chat_id"], result)
                self.wake_queue.complete(event_id)
                return
            self.wake_queue.complete(event_id)
        except Exception:
            logger.exception("Wake failed for %s", event_id)
            self.wake_queue.fail(event_id, retry_after=retry_after)

    def _handle_task_completed(self, event: Event) -> None:
        payload = event.payload
        task = self.plan_manager.complete_task(
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
            self._complete_subagent_task(task)
            return
        plan = self.plan_manager.get_plan(payload["plan_id"])
        if plan is None:
            return
        self._enqueue_plan_task_wake(plan, task)
        self._maybe_enqueue_plan_conclusion(plan)

    def _handle_task_failed(self, event: Event) -> None:
        payload = event.payload
        task = self.plan_manager.fail_task(
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
            self._complete_subagent_task(task)
            return
        plan = self.plan_manager.get_plan(payload["plan_id"])
        if plan is None:
            return
        self._enqueue_plan_task_wake(plan, task)
        self._maybe_enqueue_plan_conclusion(plan)

    def _build_system_notice(
        self,
        persona: PersonaPrompt,
        recall: RecallResult,
        chat_status: dict[str, Any],
    ) -> str | None:
        """Build a notice when memory is truncated for a first turn prompt."""
        return self._prompts._build_system_notice(persona, recall, chat_status)

    def _trim_reply_quote_to(self, quote: str, limit: int) -> str:
        """Trim a reply-to quote to a given budget, with a truncation marker."""
        return self._prompts._trim_reply_quote_to(quote, limit)

    def _trim_reply_quote(self, quote: str) -> str:
        """Trim a reply-to quote to the configured budget, with a truncation marker."""
        return self._prompts._trim_reply_quote(quote)

    def _telegram_message_registry_path(self, chat_id: str) -> Path:
        return self._prompts._telegram_message_registry_path(chat_id)

    def _load_telegram_message_registry(self, chat_id: str) -> dict[int, dict[str, Any]]:
        return self._prompts._load_telegram_message_registry(chat_id)

    def _format_user_message(
        self,
        user_message: str,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        chat_id: str | None = None,
    ) -> str:
        """Wrap the user message with a labeled reply-to reference if present."""
        return self._prompts._format_user_message(
            user_message,
            reply_to=reply_to,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
            chat_id=chat_id,
        )

    def _build_first_prompt(
        self,
        chat_id: str,
        user_message: str,
        *,
        model: str | None = None,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        continuation_anchor: str | None = None,
    ) -> tuple[str, str | None, dict[str, bool]]:
        """Build a first-turn prompt and any memory truncation notice."""
        return self._prompts._build_first_prompt(
            chat_id,
            user_message,
            model=model,
            reply_to=reply_to,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
            continuation_anchor=continuation_anchor,
        )

    def _follow_up_prompt(
        self,
        user_message: str,
        *,
        chat_id: str,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        continuation_anchor: str | None = None,
    ) -> str:
        """Build a follow-up prompt for an existing session."""
        return self._prompts._follow_up_prompt(
            user_message,
            chat_id=chat_id,
            reply_to=reply_to,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
            continuation_anchor=continuation_anchor,
        )

    def _model(self, record: SessionRecord | None) -> str:
        return self._prompts._model(record)

    def resolve_model(
        self,
        chat_id: str,
        user_message: str,
        record: SessionRecord | None = None,
    ) -> ModelRoute:
        """Resolve the model for a user message, checking budget and lane rules."""
        return self._prompts.resolve_model(chat_id, user_message, record=record)

    def routing_context(
        self,
        chat_id: str,
        record: SessionRecord | None = None,
    ) -> dict[str, Any]:
        """Return the current routing/budget context for a chat."""
        return self._prompts.routing_context(chat_id, record=record)

    def _partial_notice(self, result: TurnResult, continue_word: str = "Continue") -> str:
        """Return a user-facing notice for an interrupted/partial ACP turn."""
        return self._prompts._partial_notice(result, continue_word=continue_word)

    def is_continuation_message(self, user_message: str) -> bool:
        """Return True if the user message is a continuation trigger."""
        return self._prompts.is_continuation_message(user_message)

    def _continuation_anchor(self, record: SessionRecord | None, user_message: str) -> str | None:
        """Return a prompt anchor when resuming an interrupted turn."""
        return self._prompts._continuation_anchor(record, user_message)

    def _turn_number(self, record: SessionRecord | None) -> int:
        return self._prompts._turn_number(record)

    def _create_record(
        self,
        chat_id: str,
        session_number: int,
        session_id: str,
        model: str,
        reply: str,
        memory_flags: dict[str, bool],
        *,
        parent: int | None = None,
        label: str | None = None,
    ) -> SessionRecord:
        return self._prompts._create_record(
            chat_id,
            session_number,
            session_id,
            model,
            reply,
            memory_flags,
            parent=parent,
            label=label,
        )

    def _start_new_session(
        self,
        chat_id: str,
        prompt: str,
        model: str,
        *,
        mcp_servers: list[dict[str, Any]] | None = None,
        skill_names: set[str] | None = None,
        on_chunk: Any | None = None,
        on_update: Any | None = None,
    ) -> tuple[TurnResult, str]:
        """Create a new ACP session and return the prompt result + session id."""
        return self._prompts._start_new_session(
            chat_id,
            prompt,
            model,
            mcp_servers=mcp_servers,
            skill_names=skill_names,
            on_chunk=on_chunk,
            on_update=on_update,
        )

    def _check_chat_memory_transition(self, chat_id: str, record: SessionRecord) -> str | None:
        """Return a system notice if the chat memory just exceeded its cap."""
        return self._prompts._check_chat_memory_transition(chat_id, record)

    def _check_persona_memory_transition(self, record: SessionRecord) -> str | None:
        """Return a system notice if the persona memory just exceeded its cap."""
        return self._prompts._check_persona_memory_transition(record)

    # ---------------------------------------------------------------- public API

    def get_model(self, chat_id: str) -> str:
        """Return the model currently used for a chat, or the default."""
        return self._prompts.get_model(chat_id)

    def _context_usage(self, record: SessionRecord) -> dict[str, Any]:
        """Return context-window and prompt-budget usage for a chat record."""
        return self._runtime_metrics._context_usage(record)

    def status(self, chat_id: str) -> dict[str, Any]:
        """Return the harness-recorded status for a chat."""
        with self._lock:
            record = self._active_record(chat_id)
            if not record:
                return {"chat_id": chat_id, "active": False}

        memory_stats: dict[str, Any] = {}
        try:
            memory_stats = self._memory_manager(chat_id).stats()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load memory stats for %s: %s", chat_id, exc)

        context_usage = self._context_usage(record)
        background_tasks = self.subagent_status(chat_id)

        active_turn = self.turn_status(chat_id, wait=0.0)

        with self._lock:
            record = self._active_record(chat_id)
            if not record:
                return {"chat_id": chat_id, "active": False}

            if context_usage and record.last_stop_reason:
                context_usage["last_turn"]["stop_reason"] = record.last_stop_reason

            return {
                "chat_id": chat_id,
                "active": True,
                "persona": record.persona,
                "model": record.model,
                "session_number": record.session_number,
                "session_id": record.session_id,
                "cwd": record.cwd,
                "turn_number": record.turn_number,
                "persona_memory_exceeded": record.persona_memory_exceeded,
                "chat_memory_exceeded": record.chat_memory_exceeded,
                "memory": memory_stats,
                "last_turn_metrics": record.last_turn_metrics,
                "cumulative_metrics": record.cumulative_metrics,
                "context_usage": context_usage,
                "enabled_mcp_servers": record.enabled_mcp_servers,
                "enabled_skills": record.enabled_skills,
                "disabled_skills": record.disabled_skills,
                "background_tasks": background_tasks,
                "active_turn": active_turn,
            }

    def list_sessions(self, chat_id: str) -> dict[str, Any]:
        """Return all non-pruned sessions for a chat, with the active one marked."""
        with self._lock:
            state = self._chat_state(chat_id)
            active = self._active_record(chat_id)
            sessions = []
            for record in sorted(state.sessions.values(), key=lambda r: r.session_number):
                sessions.append(
                    {
                        "number": record.session_number,
                        "label": record.label,
                        "model": record.model,
                        "turn_number": record.turn_number,
                        "updated_at": record.updated_at,
                        "parent": record.parent,
                        "is_active": active is not None
                        and record.session_number == active.session_number,
                    }
                )
            return {
                "chat_id": chat_id,
                "active": active.session_number if active else None,
                "sessions": sessions,
            }

    @_locked
    def memory(self, chat_id: str) -> str:
        """Return the per-chat memory content."""
        return self._memory_manager(chat_id).memory_content()

    @_locked
    def summarize(self, chat_id: str) -> ChatResult:
        """Trigger a manual summarization for a chat."""
        record = self._active_record(chat_id)
        model = self._model(record)
        mgr = self._memory_manager(chat_id)
        self._call_unlocked(mgr._summarize, model)

        notice = None
        if record:
            notice = self._check_chat_memory_transition(chat_id, record)
            self._append_record(record)

        return ChatResult(reply="Summarization complete.", notice=notice)

    @_locked
    def recall(
        self,
        chat_id: str,
        query: str,
        tags: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Recall relevant memories for a query."""
        max_tokens = max_tokens or self.config.harness.memory.hindsight.max_recall_tokens
        reply = self._memory_manager(chat_id).backend.recall(
            query,
            tags=tags,
            max_tokens=max_tokens,
        )
        return ChatResult(reply=reply or "No relevant memory found.")

    @_locked
    def retain(
        self,
        chat_id: str,
        content: str,
        tags: list[str] | None = None,
        context: str | None = None,
    ) -> ChatResult:
        """Retain an observation in the chat's memory."""
        ctx = self._plugins.before_retain(
            chat_id,
            RetainContext(
                chat_id=chat_id,
                content=content,
                tags=tags or [],
                context=context,
            ),
        )
        self._memory_manager(chat_id).retain(ctx.content, tags=ctx.tags, context=ctx.context)
        self._plugins.after_retain(chat_id, ctx)
        return ChatResult(reply="Retained.")

    @_locked
    def promote(self, chat_id: str, fact: str) -> ChatResult:
        """Promote a fact to the persona's memory."""
        record = self._active_record(chat_id)
        ctx = self._plugins.before_promote(
            chat_id,
            PromoteContext(chat_id=chat_id, fact=fact, record=record),
        )
        self._memory_manager(chat_id).promote_to_persona(ctx.fact)

        record = self._active_record(chat_id)
        notice = None
        if record:
            notice = self._check_persona_memory_transition(record)
            self._append_record(record)
            self._plugins.after_promote(
                chat_id,
                PromoteContext(chat_id=chat_id, fact=ctx.fact, record=record),
            )

        return ChatResult(reply="Promoted to persona memory.", notice=notice)

    def list_models(self) -> list[str]:
        """Return the list of models the ACP server accepts."""
        return self.engine.list_models()

    def register_ingress_handler(self, protocol: str, handler: Any) -> None:
        """Register a protocol-specific inbound HTTP handler."""
        self._ingress_handlers[protocol] = handler

    async def handle_ingress(self, protocol: str, request: Any) -> Any:
        """Dispatch an inbound HTTP request to the registered handler."""
        from fastapi import HTTPException, Request, status

        if not isinstance(request, Request):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid ingress request object",
            )
        handler = self._ingress_handlers.get(protocol)
        if handler is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown ingress protocol: {protocol}",
            )
        return await handler.handle(request)

    def _load_mesh_ingress(self) -> None:
        """Load the configured mesh ingress handler if mesh is enabled."""
        from diploid_agent.transport.ingress import load_ingress_handler

        mesh = self.config.harness.mesh
        if not mesh.enabled:
            return
        try:
            handler = load_ingress_handler(mesh.ingress_module, runtime=self)
            self.register_ingress_handler("mesh", handler)
        except Exception:
            logger.exception("Failed to load mesh ingress handler: %s", mesh.ingress_module)

    def process(
        self,
        chat_id: str,
        user_message: str,
        *,
        model: str | None = None,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        notify: bool = True,
    ) -> ChatResult:
        if not self.instance_manager.acquire(chat_id):
            # The chat is busy with another turn. Rather than dropping the user
            # message, queue it as a high-priority wake so it runs when the
            # current turn releases the lock.
            payload = {
                "user_message": user_message,
                "model": model,
                "reply_to": reply_to,
                "reply_to_is_bot": reply_to_is_bot,
                "reply_to_message_id": reply_to_message_id,
                "notify": True,
                "retry_after": 2.0,
            }
            self.wake_queue.enqueue(
                WakeEvent(
                    id="",
                    chat_id=chat_id,
                    reason="user_request",
                    priority=10,
                    scheduled_at=time.time(),
                    payload=payload,
                    silent=False,
                    created_at=time.time(),
                    ready=True,
                )
            )
            return ChatResult(
                reply="I'll get back to you in a moment.",
                notice="This chat is busy; your message was queued.",
            )
        try:
            result = self.turn_controller.process(
                chat_id,
                user_message,
                model=model,
                reply_to=reply_to,
                reply_to_is_bot=reply_to_is_bot,
                reply_to_message_id=reply_to_message_id,
                notify=notify,
                other_instance_running=False,
            )
            if result is not None and reply_to_message_id is not None:
                result.reply_to_message_id = reply_to_message_id
            return result
        finally:
            self.instance_manager.release(chat_id)

    @_locked
    def dispatch(
        self,
        chat_id: str,
        context: str | None = None,
    ) -> ChatResult:
        return self.turn_controller.dispatch(chat_id, context=context)

    def continue_turn(self, dispatch_id: str, result: str) -> ChatResult:
        dispatch = self.dispatch_store.get(dispatch_id)
        if dispatch is None:
            return ChatResult(reply="Unknown dispatch.")
        chat_id = dispatch.chat_id
        if not self.instance_manager.acquire(chat_id):
            return ChatResult(reply="Another instance is currently handling this chat.")
        wake_id = f"wake-{dispatch_id}"
        self.wake_queue.ready(wake_id, now=time.time())
        try:
            chat_result = self.turn_controller.continue_turn(dispatch_id, result)
            if chat_result.turn_number is not None:
                self.wake_queue.complete(wake_id)
            return chat_result
        except Exception:
            self.wake_queue.fail(
                wake_id,
                retry_after=self.config.harness.waker.retry_after,
            )
            raise
        finally:
            self.instance_manager.release(chat_id)

    def wake(
        self,
        chat_id: str,
        event_id: str | None = None,
        reason: str | None = None,
        silent: bool | None = None,
    ) -> ChatResult:
        event = None
        if event_id is not None:
            event = self.wake_queue.get(event_id)
            if event is None:
                return ChatResult(reply="Unknown or already completed wake event.")
            chat_id = event.chat_id
            reason = event.reason
            if silent is None:
                silent = event.silent

        if silent is None:
            silent = False
        reason = reason or "user_request"

        if not self.instance_manager.acquire(chat_id):
            return ChatResult(reply="Chat is busy; wake re-enqueued.")

        try:
            payload = event.payload if event else {}
            if reason == "plan_task_update":
                user_message = self._build_plan_task_update_message(payload)
            elif reason == "plan_completed":
                user_message = self._build_plan_completed_message(payload)
            else:
                if payload and isinstance(payload.get("user_message"), str):
                    user_message = payload["user_message"]
                else:
                    user_message = f"[system wake: {reason}]"
                    if payload:
                        user_message += f"\n{json.dumps(payload, default=str)}"

            model = payload.get("model")
            reply_to = payload.get("reply_to")
            reply_to_is_bot = payload.get("reply_to_is_bot")
            reply_to_message_id = payload.get("reply_to_message_id")
            wake_notify = payload.get("notify", not silent)

            if reason == "dispatch" and "dispatch_id" in payload:
                dispatch = self.dispatch_store.get(payload["dispatch_id"])
                if dispatch and dispatch.status in (
                    DispatchStatus.PENDING,
                    DispatchStatus.TIMEOUT,
                    DispatchStatus.CANCELLED,
                    DispatchStatus.FAILED,
                ):
                    return self.continue_turn(
                        payload["dispatch_id"],
                        payload.get("result", dispatch.result or ""),
                    )

            result = self.turn_controller.process(
                chat_id,
                user_message,
                model=model,
                reply_to=reply_to,
                reply_to_is_bot=reply_to_is_bot,
                reply_to_message_id=reply_to_message_id,
                wake_event=event,
                notify=wake_notify,
                other_instance_running=False,
            )
            if result is not None and reply_to_message_id is not None:
                result.reply_to_message_id = reply_to_message_id
            return result
        finally:
            self.instance_manager.release(chat_id)

    def _enqueue_plan_task_wake(self, plan: Plan, task: Task) -> None:
        """Enqueue a non-silent wake that reports one task's completion or failure."""
        if plan.chat_id is None:
            return
        total = len(plan.tasks)
        done = sum(1 for t in plan.tasks if t.status == TaskStatus.DONE)
        failed = sum(1 for t in plan.tasks if t.status == TaskStatus.FAILED)
        completed_count = done + failed
        detail = task.result if task.status == TaskStatus.DONE and task.result else task.log
        payload = {
            "plan_id": plan.id,
            "plan_name": plan.name,
            "plan_status": plan.status.value,
            "task_id": task.id,
            "task_name": task.name,
            "task_status": task.status.value,
            "task_detail": ((detail or "").replace("\n", " ").strip())[:500],
            "completed_count": completed_count,
            "total_count": total,
            "retry_after": 5.0,
        }
        self.wake_queue.enqueue(
            WakeEvent(
                id="",
                chat_id=plan.chat_id,
                reason="plan_task_update",
                priority=1,
                scheduled_at=time.time(),
                payload=payload,
                silent=False,
                created_at=time.time(),
                ready=True,
            )
        )

    def _maybe_enqueue_plan_conclusion(self, plan: Plan) -> None:
        """Enqueue a single non-silent wake with the plan's final conclusion."""
        if plan.chat_id is None:
            return
        if plan.status not in (PlanStatus.COMPLETED, PlanStatus.FAILED):
            return
        if plan.id in self._plan_conclusion_enqueued:
            return
        self._plan_conclusion_enqueued.add(plan.id)

        task_lines: list[str] = []
        for t in plan.tasks:
            line = f"- {t.name}: {t.status.value}"
            if t.status == TaskStatus.DONE and t.result:
                line += f" ({t.result[:100].replace(chr(10), ' ').strip()})"
            elif t.status == TaskStatus.FAILED and t.log:
                line += f" ({t.log[:100].replace(chr(10), ' ').strip()})"
            task_lines.append(line)

        payload = {
            "plan_id": plan.id,
            "plan_name": plan.name,
            "plan_status": plan.status.value,
            "tasks_summary": "\n".join(task_lines),
            "retry_after": 5.0,
        }
        self.wake_queue.enqueue(
            WakeEvent(
                id="",
                chat_id=plan.chat_id,
                reason="plan_completed",
                priority=1,
                scheduled_at=time.time(),
                payload=payload,
                silent=False,
                created_at=time.time(),
                ready=True,
            )
        )

    def _build_plan_task_update_message(self, payload: dict[str, Any]) -> str:
        """Format a user message for a single task status update."""
        plan_name = payload.get("plan_name", "unknown")
        plan_status = payload.get("plan_status", "unknown")
        task_name = payload.get("task_name", "unknown")
        task_status = payload.get("task_status", "unknown")
        task_detail = payload.get("task_detail", "")
        completed = payload.get("completed_count", 0)
        total = payload.get("total_count", 0)
        lines = [
            "[system wake: plan task update]",
            "",
            f"Plan: {plan_name}",
            f"Plan status: {plan_status} ({completed}/{total} tasks finished)",
            f"Task: {task_name}",
            f"Task status: {task_status}",
        ]
        if task_detail:
            lines.append(f"Detail: {task_detail}")
        lines.append("")
        lines.append("Briefly report this task's status to the user.")
        return "\n".join(lines)

    def _build_plan_completed_message(self, payload: dict[str, Any]) -> str:
        """Format a user message asking the model to deliver the final conclusion."""
        plan_name = payload.get("plan_name", "unknown")
        plan_status = payload.get("plan_status", "unknown")
        tasks_summary = payload.get("tasks_summary", "")
        lines = [
            "[system wake: plan completed]",
            "",
            f"Plan: {plan_name}",
            f"Final status: {plan_status}",
        ]
        if tasks_summary:
            lines.extend(["", "Task summary:", tasks_summary])
        lines.append("")
        lines.append("Please give a concise final conclusion.")
        return "\n".join(lines)

    def _build_dispatch_continuation(self, dispatch: Dispatch) -> str:
        """Build a continuation anchor for a completed background dispatch."""
        return self.context_builder.build_dispatch_continuation(dispatch)

    def turn_status(self, chat_id: str, wait: float = 0.0) -> dict[str, Any]:
        return self.turn_controller.turn_status(chat_id, wait=wait)

    def stop(self, chat_id: str) -> ChatResult:
        return self.turn_controller.stop(chat_id)

    def restart(self, chat_id: str) -> ChatResult:
        return self.turn_controller.restart(chat_id)

    def record_mesh_message(
        self,
        chat_id: str,
        display_text: str,
        mesh_payload: dict[str, Any],
    ) -> ChatResult:
        """Persist a terminal mesh message (e.g. a DSN) without running a turn.

        Notifies plugins with a non-mesh wake event so the mesh plugin does not
        overwrite `current_mesh` for terminal messages, and appends a transcript
        note so the context is not lost. This deliberately does not enqueue a
        wake because a DSN must not trigger a model turn.
        """
        record = self._active_record(chat_id)

        event = WakeEvent(
            id=f"mesh:{mesh_payload.get('message_id', 'unknown')}",
            chat_id=chat_id,
            reason="mesh_dsn",
            priority=1,
            scheduled_at=time.time(),
            payload={"user_message": display_text, "mesh_meta": mesh_payload},
            silent=True,
            ready=False,
        )
        self._plugins.on_waking(chat_id, record, time.time(), wake_event=event)

        self._memory_manager(chat_id).append_mesh_note(display_text)

        return ChatResult(reply="", notice=None)

    @_locked
    def graceful_service_restart(
        self,
        chat_id: str,
        service: str | None = None,
        reason: str = "",
    ) -> ChatResult:
        """Public API for a graceful service restart (HTTP/Telegram/MCP)."""
        if service is None:
            service = f"{self.config.persona.name}.service"
        now = time.time()
        if now - self._last_service_restart_at < self._service_restart_cooldown_seconds:
            return ChatResult(
                reply=f"A restart for {service} is already scheduled.",
                notice="Please wait for it to complete.",
            )
        self._last_service_restart_at = now

        # Suppress auto-continue for this chat and cancel pending wakes.
        self.suppress_auto_continue(chat_id=chat_id, seconds=300)
        if self.wake_queue is not None:
            try:
                self.wake_queue.cancel(chat_id=chat_id, reason="auto_continue")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to cancel auto-continue wakes for %s: %s", chat_id, exc)

        if self._incidents is not None:
            try:
                self._incidents.record(
                    plugin="self_management",
                    phase="graceful_restart",
                    error=f"User requested restart of {service}: {reason}",
                    action="scheduled",
                    chat_id=chat_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to record restart incident for %s: %s", chat_id, exc)

        self._schedule_systemd_restart(service, delay=5.0, chat_id=chat_id, reason=reason)

        return ChatResult(
            reply=f"Restarting {service}. I'll be back in a moment.",
            notice="The service will restart in a few seconds.",
        )

    @_locked
    def switch_model(self, chat_id: str, model: str) -> ChatResult:
        return self.turn_controller.switch_model(chat_id, model)

    def new_session(self, chat_id: str, model: str | None = None) -> ChatResult:
        return self.turn_controller.new_session(chat_id, model=model)

    def resume_session(self, chat_id: str, session_number: int) -> ChatResult:
        return self.turn_controller.resume_session(chat_id, session_number)

    def branch_session(self, chat_id: str, session_number: int) -> ChatResult:
        return self.turn_controller.branch_session(chat_id, session_number)

    # ---------------------------------------------------------------- plans / tasks

    def plan_create(
        self,
        name: str,
        description: str = "",
        chat_id: str | None = None,
        tasks: list[Task] | None = None,
    ) -> Plan:
        """Create a new plan."""
        return self.plan_manager.create_plan(
            name, description=description, chat_id=chat_id, tasks=tasks or []
        )

    def plan_task_start(self, plan_id: str, task_id: str | None = None) -> Task:
        """Start a ready task in a plan."""
        return self.task_engine.start_task(plan_id, task_id)

    def plan_task_done(
        self,
        plan_id: str,
        task_id: str,
        result: str = "",
        log: str = "",
    ) -> Task:
        """Manually mark a task as done and emit the completion event."""
        existing = self.plan_manager.get_task(plan_id, task_id)
        already_done = existing is not None and existing.status == TaskStatus.DONE
        task = self.plan_manager.complete_task(plan_id, task_id, result=result, log=log)
        if task is None:
            raise ValueError(f"Task {task_id} not found in plan {plan_id}")
        if not already_done:
            self.event_bus.post(
                Event(
                    type="task.completed",
                    payload={
                        "plan_id": plan_id,
                        "task_id": task_id,
                        "result": result,
                        "log": log,
                    },
                )
            )
        return task

    @_locked
    def subagent_start(
        self,
        chat_id: str,
        prompt: str,
        *,
        context: str | None = None,
        model: str | None = None,
        cwd: Path | None = None,
        acp_timeout: float | None = None,
    ) -> ChatResult:
        return self._subagent.subagent_start(
            chat_id,
            prompt,
            context=context,
            model=model,
            cwd=cwd,
            acp_timeout=acp_timeout,
        )

    def _persist_subagent_result(self, dispatch: Dispatch, result: str) -> Path | None:
        return self._subagent._persist_subagent_result(dispatch, result)

    def _complete_subagent_task(self, task: Task) -> None:
        self._subagent._complete_subagent_task(task)

    @staticmethod
    def _extract_summary(text: str, max_chars: int = 240) -> str:
        return RuntimeSubagent._extract_summary(text, max_chars=max_chars)

    @staticmethod
    def _human_duration(seconds: float) -> str:
        return RuntimeSubagent._human_duration(seconds)

    def _subagent_terminal_state(
        self, task: Task
    ) -> tuple[DispatchStatus | None, str | None, bool, bool, bool]:
        return self._subagent._subagent_terminal_state(task)

    def _notify_subagent_timeout(
        self,
        task: Task,
        chat_id: str,
        dispatch_id: str,
        dispatch: Dispatch,
        summary: str,
        is_timeout: bool,
    ) -> None:
        self._subagent._notify_subagent_timeout(task, chat_id, dispatch_id, dispatch, summary, is_timeout)

    def _subagent_status_name(self, task: Task, dispatch: Dispatch | None) -> str:
        """Map a subagent task/dispatch to a simple status string."""
        if task.timed_out or (dispatch is not None and dispatch.status == DispatchStatus.TIMEOUT):
            return "timeout"
        if task.cancelled or (dispatch is not None and dispatch.status == DispatchStatus.CANCELLED):
            return "cancelled"
        if task.status == TaskStatus.RUNNING:
            return "running"
        if dispatch is not None and dispatch.status == DispatchStatus.PENDING and dispatch.finished_at is None:
            return "running"
        if task.status == TaskStatus.DONE:
            return "completed"
        if task.status == TaskStatus.FAILED:
            return "failed"
        if task.status in (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.BLOCKED):
            return "pending"
        return "unknown"

    def _subagent_summary(self, task: Task, dispatch: Dispatch | None) -> str | None:
        """Return the best available summary for a subagent task."""
        if dispatch and dispatch.summary:
            return dispatch.summary
        text = task.log if task.status == TaskStatus.FAILED else task.result
        if text:
            return self._extract_summary(text, max_chars=240)
        return None

    def subagent_status(self, chat_id: str) -> dict[str, Any]:
        return self._subagent.subagent_status(chat_id)

    def plan_list(self) -> list[Plan]:
        """List all plans."""
        return self.plan_manager.list_plans()

    def plan_get(self, plan_id: str) -> Plan | None:
        """Return one plan by id."""
        return self.plan_manager.get_plan(plan_id)

    # ---------------------------------------------------------------- pruning

    def _prune_chat(self, chat_id: str) -> None:
        """Delete archived sessions older than the prune window."""
        self._chat_store._prune_chat(chat_id)

    def _prune_and_compact(self, chat_id: str) -> None:
        self._chat_store._prune_and_compact(chat_id)

    def _prune_all(self) -> None:
        self._chat_store._prune_all()
