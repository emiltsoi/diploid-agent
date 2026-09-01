"""AgentRuntime: service container and non-turn public API."""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from diploid_agent.config import (
    Config,
    ConfigPersistenceError,
    NotificationsConfig,
    PluginConfig,
    TaskConfig,
    TelegramConfig,
    TimerConfig,
    WakerConfig,
)
from diploid_agent.context import ContextBuilder
from diploid_agent.dispatch import Dispatch, DispatchStatus, DispatchStore
from diploid_agent.engine import AgentEngine, TurnRequest, TurnResult, build_engine
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
from diploid_agent.notifier import NoopNotifier, Notifier, TelegramNotifier, WebhookNotifier
from diploid_agent.persona_composer import PersonaPrompt
from diploid_agent.plan.manager import PlanManager
from diploid_agent.plan.models import Plan, PlanStatus, Task, TaskStatus, TaskType
from diploid_agent.plugin_incidents import PluginIncidentStore
from diploid_agent.plugins import PluginManager
from diploid_agent.plugins.contexts import (
    McpCommandContext,
    MemoryTransitionContext,
    PromoteContext,
    RetainContext,
    ShutdownContext,
    SkillCommandContext,
)
from diploid_agent.runtime.event_bus import Event, EventBus
from diploid_agent.runtime.instance import InstanceManager
from diploid_agent.runtime.timer_service import TimerService
from diploid_agent.runtime.turn_controller import TurnController
from diploid_agent.runtime.wake_queue import WakeQueue
from diploid_agent.skills import SkillManager
from diploid_agent.task.engine import TaskEngine
from diploid_agent.transport.base import RuntimeAPI

logger = logging.getLogger(__name__)

# This set is intentionally keyed by file name, not full path. Durable files are
# expected to live at the root of each session directory, and that convention is
# enforced by _copy_session_dir. This is a known limitation/acceptance.
_CHAT_DURABLE_FILES = {
    "chat_transcript.jsonl",
    "chat_MEMORY.md",
    "chat_self_state.md",
    "hindsight-pending-retain.jsonl",
}

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


def _is_telegram_chat_id(chat_id: str) -> bool:
    """Return True if ``chat_id`` looks like a real Telegram chat id."""
    stripped = chat_id.lstrip("-")
    return stripped.isdigit()


class AgentRuntime(RuntimeAPI):
    """Persistent chat runtime backed by Devin ACP."""

    def __init__(self, config: Config):
        self.config = config
        self.sessions_root = Path(config.harness.sessions_root).expanduser()
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.store_path = Path(config.harness.session_store_path).expanduser()
        self._runtime_overrides_path = self.store_path.parent / "runtime-overrides.yaml"
        self._loaded_overrides: dict[str, Any] | None = None
        self.metrics = MetricsCollector()
        self.engine = self._create_engine(metrics=self.metrics)
        self._store: dict[str, ChatState] = {}
        self._lock = threading.RLock()
        self._active_turns: dict[str, ActiveTurn] = {}
        self._active_chat_skills: dict[str, set[str]] = {}
        self._per_chat_metrics: dict[str, dict[str, Any]] = {}
        self._memory_managers: dict[str, MemoryManager] = {}
        self._global_metrics: dict[str, Any] = {
            "turns": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "latency_seconds": 0.0,
        }
        self._recent_turns: deque[dict[str, Any]] = deque(
            maxlen=config.harness.metrics.max_recent_turns
        )
        self.instance_id = f"harness-{uuid.uuid4().hex[:12]}"
        self.instance_started_at = time.time()
        self._router = ModelRouter(config)

        # Load external plugin search paths before PluginManager imports anything.
        for plugin_path in self.config.harness.plugin_paths:
            if plugin_path.exists() and str(plugin_path) not in sys.path:
                sys.path.append(str(plugin_path))

        self._load_store()
        dispatch_store_path = Path(config.harness.dispatch_store_path).expanduser()
        self.dispatch_store = DispatchStore(dispatch_store_path)

        wake_store_path = Path(config.harness.wake_store_path).expanduser()
        self.wake_queue = WakeQueue(wake_store_path)

        plan_root = Path(config.harness.plan.root).expanduser()
        plan_root.mkdir(parents=True, exist_ok=True)
        self._load_runtime_overrides()
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

        self._outbox: deque[tuple[str, ChatResult]] = deque()
        self._outbox_condition = threading.Condition()

        self.instance_manager = InstanceManager(
            self.sessions_root,
            self.instance_id,
            ttl_seconds=config.harness.instance_ttl_seconds,
        )

        if config.harness.session_prune_enabled:
            self._prune_all()
        self._rehydrate_metrics()

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
            self._save_runtime_overrides()
        self._plugin_mcp_server_names: set[str] = set()
        self._register_plugin_mcp_servers()

        # Ingress handlers for pluggable transport protocols (e.g. mesh).
        self._ingress_handlers: dict[str, Any] = {}

        self.mcp = McpManager(config)
        self.skills = SkillManager(
            personas_root=Path(self.config.persona.profile_root).parent,
            shared_root=self.config.harness.skills.shared_root,
            chat_cwd_root=self.sessions_root,
        )

        # Prompt assembly is delegated to a dedicated builder.
        self.context_builder = ContextBuilder(
            self.config,
            self._plugins,
            self._memory_manager,
            self.skills,
            self._active_skill_names,
        )
        self.context_builder.metrics = self._per_chat_metrics

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

    def _create_notifier(self) -> Notifier:
        if self.config.harness.notifications.outbox_delivery:
            # The transport (e.g. Telegram long-poller) will consume the outbox.
            return NoopNotifier()
        if not self.config.harness.notifications.enabled:
            return NoopNotifier()
        if self.config.harness.notifications.webhook_url:
            return WebhookNotifier(self.config.harness.notifications.webhook_url)
        token = self.config.harness.telegram.token
        if token:
            return TelegramNotifier(token, metrics=self.metrics)
        return NoopNotifier()

    @property
    def _outbox_delivery_enabled(self) -> bool:
        return self.config.harness.notifications.outbox_delivery

    def _enqueue_outbox(self, chat_id: str, chat_result: ChatResult) -> None:
        """Put a final ChatResult in the outbox for the transport to deliver."""
        with self._outbox_condition:
            self._outbox.append((chat_id, chat_result))
            self._outbox_condition.notify_all()

    def _safe_notifier_send(self, chat_id: str, text: str, notifier: Notifier | None = None) -> None:
        """Send a notification, swallowing exceptions and logging them."""
        if not text:
            return
        notifier = notifier or self.notifier
        if notifier is None:
            return
        try:
            notifier.send(chat_id, text)
        except Exception:
            logger.exception("Failed to send notification to %s", chat_id)

    def _deliver_chat_result(self, chat_id: str, chat_result: ChatResult) -> None:
        """Send a final ChatResult through the configured delivery channel."""
        if self._outbox_delivery_enabled:
            self._enqueue_outbox(chat_id, chat_result)
            return
        if self.config.harness.notifications.enabled and chat_result.reply:
            self._safe_notifier_send(chat_id, chat_result.reply)

    def outbox_pop(
        self,
        chat_id: str | None = None,
        wait: float = 0.0,
    ) -> ChatResult | None:
        """Return the next ChatResult for a chat, blocking up to ``wait`` seconds."""
        deadline = time.monotonic() + wait if wait > 0 else 0.0
        with self._outbox_condition:
            while True:
                for i, (cid, result) in enumerate(self._outbox):
                    if chat_id is None or cid == chat_id:
                        popped = self._outbox[i]
                        del self._outbox[i]
                        return popped[1]
                if wait <= 0:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._outbox_condition.wait(timeout=remaining)

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
        if not self.store_path.exists():
            return
        for line in self.store_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = SessionRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError):
                continue
            state = self._store.setdefault(record.chat_id, ChatState())
            state.sessions[record.session_number] = record
            state.next_session_number = max(state.next_session_number, record.session_number + 1)

    def _append_record(self, record: SessionRecord) -> None:
        with self._lock:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.store_path, "a") as f:
                f.write(json.dumps(record.to_dict(), default=str) + "\n")

    def _compact_store(self) -> None:
        """Rewrite the store without pruning churn; called after pruning."""
        with self._lock:
            lines = []
            for state in self._store.values():
                for record in state.sessions.values():
                    lines.append(json.dumps(record.to_dict(), default=str) + "\n")
            tmp_path = self.store_path.with_suffix(".jsonl.new")
            tmp_path.write_text("".join(lines))
            tmp_path.replace(self.store_path)

    # ---------------------------------------------------------------- session dirs

    def _chat_dir(self, chat_id: str) -> Path:
        safe = chat_id.replace("/", "_")
        return self.sessions_root / safe

    def _archive_dir(self, chat_id: str, session_number: int) -> Path:
        return self._chat_dir(chat_id) / ".archive" / str(session_number)

    def _durable_file_names(self) -> set[str]:
        names = set(_CHAT_DURABLE_FILES)
        names.update(self._plugins.durable_files())
        return names

    def _copy_session_dir(self, source: Path, target: Path) -> None:
        if source == target:
            return
        durable = self._durable_file_names()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            for item in target.iterdir():
                if item.name in durable or item.name == ".archive":
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        else:
            target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.name in durable or item.name == ".archive":
                continue
            dest = target / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    def _archive_active_session(self, chat_id: str, record: SessionRecord) -> None:
        """Copy the active directory into the archive for `record`."""
        active_dir = self._chat_dir(chat_id)
        archive = self._archive_dir(chat_id, record.session_number)
        if active_dir.exists() and any(active_dir.iterdir()):
            self._copy_session_dir(active_dir, archive)

    def _clear_active_session(self, chat_id: str) -> None:
        active_dir = self._chat_dir(chat_id)
        if not active_dir.exists():
            active_dir.mkdir(parents=True, exist_ok=True)
            return
        durable = self._durable_file_names()
        for item in active_dir.iterdir():
            if item.name in durable or item.name == ".archive":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    # ---------------------------------------------------------------- active state

    def _chat_state(self, chat_id: str) -> ChatState:
        with self._lock:
            return self._store.setdefault(chat_id, ChatState())

    def _active_record(self, chat_id: str) -> SessionRecord | None:
        with self._lock:
            state = self._store.get(chat_id)
            if state is None or not state.sessions:
                return None
            return max(state.sessions.values(), key=lambda r: r.updated_at)

    def _next_session_number(self, chat_id: str) -> int:
        state = self._chat_state(chat_id)
        number = state.next_session_number
        state.next_session_number = number + 1
        return number

    def _generate_label(self, chat_id: str, user_message: str) -> str:
        """Auto-generate a short label from the first user message."""
        return self.context_builder.generate_label(chat_id, user_message)

    # ---------------------------------------------------------------- metrics

    def _rehydrate_metrics(self) -> None:
        """Seed per-chat and global metrics from the on-disk session store."""
        with self._lock:
            for chat_state in self._store.values():
                record = max(chat_state.sessions.values(), key=lambda r: r.updated_at, default=None)
                if not record or not record.cumulative_metrics:
                    continue
                cumulative = record.cumulative_metrics
                self._per_chat_metrics[record.chat_id] = {
                    "turns": cumulative.get("turns", 0),
                    "input_tokens": cumulative.get("input_tokens", 0),
                    "output_tokens": cumulative.get("output_tokens", 0),
                    "total_tokens": cumulative.get("total_tokens", 0),
                    "cached_tokens": cumulative.get("cached_tokens", 0),
                    "latency_seconds": cumulative.get("latency_seconds", 0.0),
                    "cumulative": cumulative,
                    "last_turn": record.last_turn_metrics,
                }
                self._global_metrics["turns"] += cumulative.get("turns", 0)
                self._global_metrics["input_tokens"] += cumulative.get("input_tokens", 0)
                self._global_metrics["output_tokens"] += cumulative.get("output_tokens", 0)
                self._global_metrics["total_tokens"] += cumulative.get("total_tokens", 0)
                self._global_metrics["cached_tokens"] += cumulative.get("cached_tokens", 0)
                self._global_metrics["latency_seconds"] += cumulative.get("latency_seconds", 0.0)

    def _record_turn_metrics(
        self,
        chat_id: str,
        turn_number: int,
        model: str,
        usage: dict[str, Any] | None,
        latency_seconds: float,
    ) -> dict[str, Any]:
        """Record per-turn metrics and update running totals."""
        usage = usage or {}
        turn_metrics = {
            "chat_id": chat_id,
            "turn_number": turn_number,
            "model": model,
            "input_tokens": usage.get("inputTokens") or usage.get("input_tokens", 0),
            "output_tokens": usage.get("outputTokens") or usage.get("output_tokens", 0),
            "total_tokens": usage.get("totalTokens") or usage.get("total_tokens", 0),
            "cached_tokens": usage.get("cachedReadTokens") or usage.get("cached_tokens", 0),
            "latency_seconds": round(latency_seconds, 3),
        }

        with self._lock:
            per_chat = self._per_chat_metrics.setdefault(
                chat_id,
                {
                    "turns": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cached_tokens": 0,
                    "latency_seconds": 0.0,
                    "cumulative": {},
                },
            )
            per_chat["turns"] += 1
            per_chat["input_tokens"] += turn_metrics["input_tokens"]
            per_chat["output_tokens"] += turn_metrics["output_tokens"]
            per_chat["total_tokens"] += turn_metrics["total_tokens"]
            per_chat["cached_tokens"] += turn_metrics["cached_tokens"]
            per_chat["latency_seconds"] += turn_metrics["latency_seconds"]
            per_chat["cumulative"] = {
                "turns": per_chat["turns"],
                "input_tokens": per_chat["input_tokens"],
                "output_tokens": per_chat["output_tokens"],
                "total_tokens": per_chat["total_tokens"],
                "cached_tokens": per_chat["cached_tokens"],
                "latency_seconds": round(per_chat["latency_seconds"], 3),
            }

            self._global_metrics["turns"] += 1
            self._global_metrics["input_tokens"] += turn_metrics["input_tokens"]
            self._global_metrics["output_tokens"] += turn_metrics["output_tokens"]
            self._global_metrics["total_tokens"] += turn_metrics["total_tokens"]
            self._global_metrics["cached_tokens"] += turn_metrics["cached_tokens"]
            self._global_metrics["latency_seconds"] += turn_metrics["latency_seconds"]

            self._recent_turns.append(turn_metrics)

        self.metrics.inc("turns_total")
        self.metrics.inc("tokens_total", value=turn_metrics["input_tokens"], kind="input")
        self.metrics.inc("tokens_total", value=turn_metrics["output_tokens"], kind="output")
        self.metrics.inc("tokens_total", value=turn_metrics["total_tokens"], kind="total")
        self.metrics.inc("tokens_total", value=turn_metrics["cached_tokens"], kind="cached")
        self.metrics.inc("turn_latency_seconds_total", value=turn_metrics["latency_seconds"])

        return dict(turn_metrics)

    def _metrics_context_for_prompt(self, chat_id: str, compact: bool = False) -> str | None:
        """Return a metrics notice for injection into the LLM prompt."""
        return self.context_builder.metrics_context_for_prompt(chat_id, compact=compact)

    @_locked
    def mcp_list(self, chat_id: str) -> str:
        lines = ["Configured MCP servers:"]
        for server in self.mcp.list_servers():
            status = "disabled" if server["disabled"] else "enabled"
            lines.append(
                f"- {server['name']} ({status}): {server['command']} {' '.join(server['args'])}"
            )
        return "\n".join(lines) if len(lines) > 1 else "No MCP servers configured."

    @_locked
    def mcp_enable(self, chat_id: str, name: str) -> str:
        record = self._active_record(chat_id)
        if record is None:
            return "No active session. Start one with /new first."
        ctx = self._plugins.before_mcp_enabled(
            chat_id,
            McpCommandContext(chat_id=chat_id, server_name=name, enabled=True, record=record),
        )
        name = ctx.server_name
        names = set(record.enabled_mcp_servers or self.mcp.default_enabled_names())
        names.add(name)
        record.enabled_mcp_servers = sorted(names)
        self._append_record(record)
        self._plugins.after_mcp_enabled(
            chat_id,
            McpCommandContext(chat_id=chat_id, server_name=name, enabled=True, record=record),
        )
        return f"Enabled MCP server {name}. New sessions will use it."

    @_locked
    def mcp_disable(self, chat_id: str, name: str) -> str:
        record = self._active_record(chat_id)
        if record is None:
            return "No active session. Start one with /new first."
        ctx = self._plugins.before_mcp_disabled(
            chat_id,
            McpCommandContext(chat_id=chat_id, server_name=name, enabled=False, record=record),
        )
        name = ctx.server_name
        names = set(record.enabled_mcp_servers or self.mcp.default_enabled_names())
        names.discard(name)
        record.enabled_mcp_servers = sorted(names)
        self._append_record(record)
        self._plugins.after_mcp_disabled(
            chat_id,
            McpCommandContext(chat_id=chat_id, server_name=name, enabled=False, record=record),
        )
        return f"Disabled MCP server {name}."

    @_locked
    def skill_list(self, chat_id: str) -> str:
        skills = self.skills.list_skills(chat_id)
        enabled = self._active_skill_names(chat_id)
        if not skills:
            return "No skills available."
        lines = ["Available skills:"]
        for skill in skills:
            state = "enabled" if skill.name in enabled else "disabled"
            lines.append(f"- /{skill.name} ({state}) — {skill.description or 'no description'}")
        return "\n".join(lines)

    @_locked
    def skill_enable(self, chat_id: str, name: str) -> str:
        record = self._active_record(chat_id)
        if record is None:
            return "No active session. Start one with /new first."
        ctx = self._plugins.before_skill_enabled(
            chat_id,
            SkillCommandContext(chat_id=chat_id, skill_name=name, enabled=True, record=record),
        )
        name = ctx.skill_name
        enabled = set(record.enabled_skills or self._default_active_skills())
        enabled.add(name)
        record.enabled_skills = sorted(enabled)
        record.disabled_skills = sorted(set(record.disabled_skills or []) - {name})
        self._append_record(record)
        self._plugins.after_skill_enabled(
            chat_id,
            SkillCommandContext(chat_id=chat_id, skill_name=name, enabled=True, record=record),
        )
        return f"Enabled skill /{name}."

    @_locked
    def skill_disable(self, chat_id: str, name: str) -> str:
        record = self._active_record(chat_id)
        if record is None:
            return "No active session. Start one with /new first."
        ctx = self._plugins.before_skill_disabled(
            chat_id,
            SkillCommandContext(chat_id=chat_id, skill_name=name, enabled=False, record=record),
        )
        name = ctx.skill_name
        enabled = set(record.enabled_skills or self._default_active_skills())
        enabled.discard(name)
        record.enabled_skills = sorted(enabled)
        record.disabled_skills = sorted(set(record.disabled_skills or []) | {name})
        self._append_record(record)
        self._plugins.after_skill_disabled(
            chat_id,
            SkillCommandContext(chat_id=chat_id, skill_name=name, enabled=False, record=record),
        )
        return f"Disabled skill /{name}."

    @_locked
    def skill_create(self, chat_id: str, name: str, content: str) -> str:
        if not self.config.harness.skills.allow_chat_creation:
            return "Chat-scoped skill creation is disabled."
        self.skills.create_chat_skill(chat_id, name, content)
        return f"Created chat skill /{name}. It will be available after /new."

    def get_metrics(self, chat_id: str | None = None) -> dict[str, Any]:
        """Return cumulative metrics for a chat or globally."""
        with self._lock:
            if chat_id is None:
                return {
                    "global": dict(self._global_metrics),
                    "recent_turns": list(self._recent_turns),
                }
            per_chat = self._per_chat_metrics.get(chat_id, {})
            return {
                "chat_id": chat_id,
                "cumulative": per_chat.get("cumulative", {}),
                "last_turn": per_chat.get("last_turn"),
            }

    def get_prometheus_metrics(self) -> str:
        """Return metrics in Prometheus exposition format."""
        return self.metrics.render()

    def health(self) -> dict[str, Any]:
        """Return the current health of the runtime and its dependencies."""
        components: dict[str, Any] = {}

        acp_healthy = False
        try:
            acp_healthy = bool(self.engine.health())
        except Exception as exc:  # noqa: BLE001
            logger.debug("ACP health check failed: %s", exc)
        components["acp"] = {
            "status": "ok" if acp_healthy else "error",
            "healthy": acp_healthy,
        }

        hindsight_healthy = True
        if self.config.harness.memory.backend == "hindsight":
            hindsight_healthy = self._hindsight_health()
        components["hindsight"] = {
            "status": "ok" if hindsight_healthy else "error",
            "healthy": hindsight_healthy,
        }

        telegram_healthy = False
        try:
            telegram_healthy = bool(self.notifier.health())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Telegram health check failed: %s", exc)
        components["telegram"] = {
            "status": "ok" if telegram_healthy else "error",
            "healthy": telegram_healthy,
        }

        plugin_health = self._plugins.plugin_health("0")
        plugins_healthy = all(p["healthy"] for p in plugin_health)
        components["plugins"] = {
            "status": "ok" if plugins_healthy else "error",
            "healthy": plugins_healthy,
            "details": plugin_health,
        }

        overall = "ok" if all(c["healthy"] for c in components.values()) else "degraded"
        return {
            "status": overall,
            "uptime_seconds": round(time.time() - self.instance_started_at, 3),
            "components": components,
        }

    def _hindsight_health(self) -> bool:
        """Probe the Hindsight backend health endpoint."""
        import urllib.parse

        base = self.config.harness.memory.hindsight.base_url
        if not base:
            return True
        url = urllib.parse.urljoin(base, "/health")
        try:
            import httpx

            resp = httpx.get(url, timeout=5.0)
            return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001
            logger.debug("Hindsight health check failed: %s", exc)
            return False

    # ---------------------------------------------------------------- helpers

    def _active_mcp_server_names(self, chat_id: str) -> list[str]:
        record = self._active_record(chat_id)
        if record and record.enabled_mcp_servers is not None:
            return record.enabled_mcp_servers
        return sorted(
            set(self.mcp.default_enabled_names()) | set(self._plugins.default_mcp_names())
        )

    def _active_mcp_servers(self, chat_id: str) -> list[dict[str, Any]]:
        return self.mcp.enabled_servers(chat_id, self._active_mcp_server_names(chat_id))

    def _default_active_skills(self) -> set[str]:
        """Return skills that should be active for a brand-new chat."""
        if self.config.harness.skills.default_lazy:
            return set()
        return set(self.config.harness.skills.default_enabled) | set(
            self._plugins.default_skill_names()
        )

    def _active_skill_names(self, chat_id: str) -> set[str]:
        record = self._active_record(chat_id)
        if record and record.enabled_skills is not None:
            base = set(record.enabled_skills)
        else:
            base = self._default_active_skills()
        return base | self._active_chat_skills.get(chat_id, set())

    @_locked
    def match_and_activate_skills(self, chat_id: str, user_message: str) -> set[str]:
        """Match user message against skill triggers for this turn.

        Matched skills are only active for the current turn.  Persistent
        enabling is still tracked in ``record.enabled_skills``.
        """
        record = self._active_record(chat_id)
        all_skills = {s.name for s in self.skills.list_skills(chat_id)}
        disabled = set(record.disabled_skills or []) if record else set()
        matched = self.skills.match_skills(
            user_message,
            chat_id,
            enabled=all_skills - disabled,
        )
        self._active_chat_skills[chat_id] = matched
        return self._active_skill_names(chat_id)

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
        """Append plugin MCP server configs to the harness config before McpManager sees it."""
        active_plugin_servers = self._plugins.mcp_server_configs()
        active_names = {s.name for s in active_plugin_servers}
        stale = self._plugin_mcp_server_names - active_names

        kept: list[Any] = []
        replaced: set[str] = set()
        for server in self.config.harness.mcp.servers:
            if server.name in stale:
                # This server was provided by a plugin that is now disabled or removed.
                continue
            if server.name in active_names:
                # Replace in place so updates to args/env are picked up.
                for ps in active_plugin_servers:
                    if ps.name == server.name:
                        kept.append(ps)
                        replaced.add(ps.name)
                        break
                else:
                    kept.append(server)
            else:
                # Static or otherwise non-plugin server; preserve it.
                kept.append(server)

        # Append any brand-new plugin servers.
        for ps in active_plugin_servers:
            if ps.name not in replaced:
                kept.append(ps)

        self.config.harness.mcp.servers = kept
        self._plugin_mcp_server_names = active_names

    def plugin_event(
        self,
        chat_id: str,
        plugin: str,
        *,
        event: str | None = None,
        raw_args: str | None = None,
        **params: Any,
    ) -> ChatResult:
        """Dispatch an event to a state plugin and wrap the reply in a ChatResult."""
        reply = self._plugins.event(chat_id, plugin, event=event, raw_args=raw_args, **params)
        return ChatResult(reply=reply)

    @_locked
    def plugin_list(self, chat_id: str) -> list[dict[str, Any]]:
        return self._plugins.list_plugin_status(chat_id)

    @_locked
    def plugin_set_enabled(self, chat_id: str, name: str, enabled: bool) -> ChatResult:
        return ChatResult(reply=self._plugins.set_plugin_enabled(chat_id, name, enabled))

    @_locked
    def plugin_reload(self, chat_id: str, name: str) -> ChatResult:
        return ChatResult(reply=self._plugins.reload_plugin(chat_id, name))

    @_locked
    def plugin_add(self, config: PluginConfig) -> ChatResult:
        result = self._plugins.add_plugin(config)
        self.config.harness.plugins = self._plugins._plugins
        self._register_plugin_mcp_servers()
        self.context_builder.plugin_manager = self._plugins
        self._save_runtime_overrides()
        return ChatResult(reply=result)

    @_locked
    def plugin_remove(self, name: str) -> ChatResult:
        result = self._plugins.remove_plugin(name)
        self.config.harness.plugins = self._plugins._plugins
        self._register_plugin_mcp_servers()
        self.context_builder.plugin_manager = self._plugins
        self._save_runtime_overrides()
        return ChatResult(reply=result)

    @_locked
    def plugin_toggle(self, name: str, enabled: bool, chat_id: str | None = None) -> ChatResult:
        if chat_id is not None:
            result = self._plugins.set_plugin_enabled(chat_id, name, enabled)
        else:
            result = self._plugins.toggle_plugin(name, enabled)
            self.config.harness.plugins = self._plugins._plugins
            self._register_plugin_mcp_servers()
            self.context_builder.plugin_manager = self._plugins
            self._save_runtime_overrides()
        return ChatResult(reply=result)

    @_locked
    def plugin_rollback(self, steps: int = 1) -> ChatResult:
        result = self._plugins.rollback(steps)
        self.config.harness.plugins = self._plugins._plugins
        self._register_plugin_mcp_servers()
        self.context_builder.plugin_manager = self._plugins
        self._save_runtime_overrides()
        return ChatResult(reply=result)

    @_locked
    def plugin_sandbox(self, module: str, plugin: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run a candidate plugin module through start/stop in a subprocess."""
        import json as _json
        import subprocess

        data: dict[str, Any] = {"name": "sandbox", "module": module, **(plugin or {})}
        cfg = PluginConfig(**data)
        cmd = [
            sys.executable,
            "-m",
            "diploid_agent.plugin_sandbox",
            "--module",
            module,
            "--name",
            cfg.name,
            "--chat-id",
            "0",
            "--prompt-slot",
            cfg.prompt_slot,
        ]
        if cfg.state_file:
            cmd.extend(["--state-file", cfg.state_file])
        if cfg.config:
            cmd.extend(["--config-json", _json.dumps(cfg.config, ensure_ascii=False)])
        for p in self.config.harness.plugin_paths:
            if p.exists():
                cmd.extend(["--plugin-path", str(p)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        try:
            output = _json.loads(result.stdout.splitlines()[-1])
        except (IndexError, _json.JSONDecodeError) as exc:
            output = {"ok": False, "error": f"Invalid sandbox output: {result.stdout!r} ({exc})"}
        if not output.get("ok") and self._incidents is not None:
            self._incidents.record(
                plugin=cfg.name,
                phase="sandbox",
                error=output.get("error", "unknown"),
                action="rejected",
            )
        return output

    @_locked
    def plugin_create(
        self,
        name: str,
        module: str | None = None,
        prompt_slot: str = "self_state",
        state_file: str | None = None,
        mcp_server: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Scaffold a new plugin module on disk, sandbox it, and return a ready config."""
        target_module = module or name
        if not target_module.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Unsafe plugin module name: {target_module}")
        if target_module.count(".") or target_module.startswith("/"):
            raise ValueError(f"Plugin module name must be a bare package name: {target_module}")

        plugin_root = self.config.harness.plugin_paths[0].expanduser()
        plugin_root.mkdir(parents=True, exist_ok=True)
        if str(plugin_root) not in sys.path:
            sys.path.append(str(plugin_root))

        plugin_dir = plugin_root / target_module
        if plugin_dir.exists():
            raise ValueError(f"Plugin directory already exists: {plugin_dir}")
        plugin_dir.mkdir(parents=True)

        init_path = plugin_dir / "__init__.py"
        init_path.write_text(
            f'"""{target_module} plugin for diploid-agent."""\n\n'
            f"from __future__ import annotations\n\n"
            f"from typing import Any\n\n"
            f"from diploid_agent.config import PluginConfig\n"
            f"from diploid_agent.plugins.base import StatePlugin\n\n\n"
            f"class Plugin(StatePlugin):\n"
            f'    """A minimal state plugin."""\n\n'
            f"    def __init__(\n"
            f"        self,\n"
            f"        config: PluginConfig,\n"
            f"        chat_id: str,\n"
            f"        sessions_root: Any,\n"
            f"        runtime: Any = None,\n"
            f"    ) -> None:\n"
            f"        super().__init__(config, chat_id, sessions_root, runtime=runtime)\n\n"
            f"    def prompt_block(self, max_chars: int | None = None) -> str | None:\n"
            f"        return None\n",
            encoding="utf-8",
        )

        plugin_config: dict[str, Any] = {
            "name": name,
            "enabled": True,
            "module": target_module,
            "prompt_slot": prompt_slot,
            "prompt_order": 100,
            "max_prompt_chars": 0,
        }
        if state_file:
            plugin_config["state_file"] = state_file
        if mcp_server:
            plugin_config["mcp_server"] = mcp_server
        if config:
            plugin_config["config"] = config

        sandbox_result = self.plugin_sandbox(target_module, plugin_config)
        if not sandbox_result.get("ok"):
            # Don't leave a broken scaffold behind.
            try:
                shutil.rmtree(plugin_dir)
            except OSError:
                pass
            error = sandbox_result.get("error", "unknown")
            raise ValueError(f"Sandbox failed for {target_module}: {error}")

        return plugin_config

    def incidents(self) -> list[dict[str, Any]]:
        return self._incidents.recent()

    def incidents_for_plugin(self, name: str) -> list[dict[str, Any]]:
        return self._incidents.for_plugin(name)

    @_locked
    def record_incident(
        self,
        plugin: str,
        phase: str,
        error: str,
        action: str = "",
        chat_id: str = "",
    ) -> ChatResult:
        self._incidents.record(
            plugin=plugin,
            phase=phase,
            error=error,
            action=action,
            chat_id=chat_id,
        )
        return ChatResult(reply="incident recorded")

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
        self._send_restart_notices()

    def _send_restart_notices(self) -> None:
        """Notify recently active chats that the service has restarted.

        The notice is sent through a direct notifier (not the outbox) when
        outbox delivery is enabled, because the transport's DeliveryWorker may
        not be running yet at startup.
        """
        if not self.config.harness.notifications.enabled:
            return

        recent_cutoff = time.time() - 86400.0
        chat_ids: list[str] = []
        with self._lock:
            for chat_id, state in self._store.items():
                if not _is_telegram_chat_id(chat_id):
                    continue
                if not state.sessions:
                    continue
                latest = max(state.sessions.values(), key=lambda r: r.updated_at)
                if latest.updated_at >= recent_cutoff:
                    chat_ids.append(chat_id)

        if not chat_ids:
            return

        logger.info("Sending restart notice to %d recently active chat(s)", len(chat_ids))
        text = "System: service was restarted. You can resume the conversation at any time."
        notifier = self._create_direct_notifier()
        if isinstance(notifier, NoopNotifier):
            return

        for chat_id in chat_ids:
            self._safe_notifier_send(str(chat_id), text, notifier=notifier)

    def _create_direct_notifier(self) -> Notifier:
        """Create a notifier that bypasses the outbox if possible."""
        if self.config.harness.notifications.webhook_url:
            return WebhookNotifier(self.config.harness.notifications.webhook_url)
        token = self.config.harness.telegram.token
        if token:
            return TelegramNotifier(token, metrics=self.metrics)
        return NoopNotifier()

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

    def get_status(self) -> RuntimeStatus:
        """Return the current runtime daemon status."""
        now = time.time()
        plans = self.plan_manager.list_plans()
        active_plans = [p.name for p in plans if p.status == PlanStatus.ACTIVE]
        return RuntimeStatus(
            instance_id=self.instance_id,
            started_at=self.instance_started_at,
            uptime_seconds=now - self.instance_started_at,
            event_bus_running=self.event_bus.running,
            timer_running=self.timer_service.running,
            task_engine_active=self.task_engine.is_running(),
            plan_count=len(plans),
            pending_wake_count=self.wake_queue.due_count(now=now),
            active_chat_count=len(self._active_turns),
            plan_active=bool(active_plans),
            active_plans=active_plans,
        )

    def get_task_config(self) -> TaskConfig:
        """Return the current live task configuration."""
        return self.config.harness.task

    def _update_config_section(
        self,
        current: Any,
        new: Any,
        *,
        success: str,
        post: Callable[[], None] | None = None,
        error: str = "Config updated in memory but persistence failed",
    ) -> str:
        """Apply a partial update to a config section and persist overrides."""
        with self._lock:
            for field in new.model_fields_set:
                setattr(current, field, getattr(new, field))
            if post is not None:
                post()
            if not self._save_runtime_overrides():
                raise ConfigPersistenceError(error)
            return success

    def update_task_config(self, task_config: TaskConfig) -> str:
        """Update the live task configuration, resize the worker pool, and persist.

        Only the fields that were explicitly set on ``task_config`` are mutated,
        so partial updates (e.g., just ``workers``) do not clobber other live
        settings like ``acp_model`` or ``enabled_types``. The update is applied
        in-place on the existing ``TaskConfig`` so the ``TaskEngine`` and
        ``AgentRuntime`` share the same object.

        Raises:
            ConfigPersistenceError: If the in-memory update succeeds but the
                runtime-overrides file cannot be written.
        """
        return self._update_config_section(
            self.config.harness.task,
            task_config,
            success="Task config updated",
            post=self.task_engine.reconfigure,
            error="Task config updated in memory but persistence failed",
        )

    def _load_runtime_overrides(self) -> None:
        """Load persisted runtime config overrides."""
        if not self._runtime_overrides_path.exists():
            return
        try:
            data = yaml.safe_load(self._runtime_overrides_path.read_text()) or {}
        except (OSError, ValueError, yaml.YAMLError):
            logger.exception(
                "Failed to load runtime overrides from %s", self._runtime_overrides_path
            )
            return
        if not isinstance(data, dict):
            logger.warning(
                "Runtime overrides file %s is not a dict; skipping", self._runtime_overrides_path
            )
            return
        self._loaded_overrides = data
        self._load_override("task", TaskConfig)
        self._load_override("waker", WakerConfig)
        self._load_override("timer", TimerConfig)
        self._load_override("notifications", NotificationsConfig)
        self._load_override("telegram", TelegramConfig)
        self._load_plugins_override(data.get("plugins"))

    def _load_override(self, key: str, model_cls: type) -> None:
        """Load one top-level override section if it is a valid mapping."""
        data = self._loaded_overrides
        if data is None:
            return
        section = data.get(key)
        if section is None:
            return
        if not isinstance(section, dict):
            logger.warning("Runtime overrides '%s' entry is not a dict; skipping", key)
            return
        try:
            existing = getattr(self.config.harness, key)
            if isinstance(existing, model_cls):
                # Merge the override values into the existing section so fields
                # that are missing from the runtime-overrides file keep the
                # defaults from harness.yaml. Re-validate to skip malformed
                # overrides just like a fresh model construction would.
                merged = {**existing.model_dump(), **section}
                setattr(self.config.harness, key, model_cls(**merged))
            else:
                setattr(self.config.harness, key, model_cls(**section))
        except (ValueError, TypeError):
            logger.exception("Invalid %s override in %s", key, self._runtime_overrides_path)

    def _load_plugins_override(self, section: Any) -> None:
        """Load the plugins override list, if it is a valid list of mappings."""
        if section is None:
            return
        if not isinstance(section, list):
            logger.warning("Runtime overrides 'plugins' entry is not a list; skipping")
            return
        plugins: list[PluginConfig] = []
        for item in section:
            if not isinstance(item, dict):
                logger.warning("Runtime overrides 'plugins' item is not a dict; skipping")
                continue
            try:
                plugins.append(PluginConfig(**item))
            except (ValueError, TypeError):
                logger.exception("Invalid plugin override in %s", self._runtime_overrides_path)
        if plugins:
            self.config.harness.plugins = plugins

    def _save_runtime_overrides(self) -> bool:
        """Persist runtime config overrides to disk atomically."""
        with self._lock:
            try:
                self._runtime_overrides_path.parent.mkdir(parents=True, exist_ok=True)
                overrides = {
                    "task": self.config.harness.task.model_dump(exclude_none=True, mode="json"),
                    "waker": self.config.harness.waker.model_dump(exclude_none=True, mode="json"),
                    "timer": self.config.harness.timer.model_dump(exclude_none=True, mode="json"),
                    "notifications": self.config.harness.notifications.model_dump(
                        exclude_none=True, mode="json"
                    ),
                    "telegram": self.config.harness.telegram.model_dump(
                        exclude_none=True, mode="json"
                    ),
                    "plugins": [
                        p.model_dump(exclude_none=True, mode="json")
                        for p in self.config.harness.plugins
                    ],
                }
                tmp_path = self._runtime_overrides_path.with_suffix(".yaml.tmp")
                tmp_path.write_text(
                    yaml.safe_dump(overrides, sort_keys=False, default_flow_style=False)
                )
                os.replace(tmp_path, self._runtime_overrides_path)
                return True
            except OSError:
                logger.exception(
                    "Failed to save runtime overrides to %s", self._runtime_overrides_path
                )
                return False

    def get_notifications_config(self) -> NotificationsConfig:
        """Return the current live notifications configuration."""
        return self.config.harness.notifications

    def update_notifications_config(self, notifications_config: NotificationsConfig) -> str:
        """Update the live notifications configuration and persist.

        Only explicitly set fields are mutated, so partial updates do not
        clobber other live settings. The notifier is recreated so changes take
        effect immediately.

        Raises:
            ConfigPersistenceError: If the in-memory update succeeds but the
                runtime-overrides file cannot be written.
        """
        return self._update_config_section(
            self.config.harness.notifications,
            notifications_config,
            success="Notifications config updated",
            post=lambda: setattr(self, "notifier", self._create_notifier()),
            error="Notifications config updated in memory but persistence failed",
        )

    def get_timer_config(self) -> TimerConfig:
        """Return the current live timer configuration."""
        return self.config.harness.timer

    def update_timer_config(self, timer_config: TimerConfig) -> str:
        """Update the live timer configuration and persist.

        Only explicitly set fields are mutated, so partial updates do not
        clobber other live settings.

        Raises:
            ConfigPersistenceError: If the in-memory update succeeds but the
                runtime-overrides file cannot be written.
        """
        return self._update_config_section(
            self.config.harness.timer,
            timer_config,
            success="Timer config updated",
            error="Timer config updated in memory but persistence failed",
        )

    def get_waker_config(self) -> WakerConfig:
        """Return the current live waker configuration."""
        return self.config.harness.waker

    def update_waker_config(self, waker_config: WakerConfig) -> str:
        """Update the live waker configuration and persist.

        Only explicitly set fields are mutated, so partial updates do not
        clobber other live settings.

        Raises:
            ConfigPersistenceError: If the in-memory update succeeds but the
                runtime-overrides file cannot be written.
        """
        return self._update_config_section(
            self.config.harness.waker,
            waker_config,
            success="Waker config updated",
            error="Waker config updated in memory but persistence failed",
        )

    def get_telegram_config(self) -> TelegramConfig:
        """Return the current live Telegram configuration."""
        return self.config.harness.telegram

    def update_telegram_config(self, telegram_config: TelegramConfig) -> str:
        """Update the live Telegram configuration and persist.

        Only explicitly set fields are mutated, so partial updates do not
        clobber other live settings. The notifier is recreated so changes to
        the token or enabled flag take effect immediately.

        Raises:
            ConfigPersistenceError: If the in-memory update succeeds but the
                runtime-overrides file cannot be written.
        """
        return self._update_config_section(
            self.config.harness.telegram,
            telegram_config,
            success="Telegram config updated",
            post=lambda: setattr(self, "notifier", self._create_notifier()),
            error="Telegram config updated in memory but persistence failed",
        )

    def get_plugins_config(self) -> list[PluginConfig]:
        """Return the current live plugin list."""
        return list(self.config.harness.plugins)

    def update_plugins_config(self, plugins: list[PluginConfig]) -> str:
        """Update the live plugin list and persist.

        Plugins are matched by name and only fields that were explicitly set in
        the request are mutated. New plugins are appended. The PluginManager is
        reconfigured with the merged list.

        Raises:
            ConfigPersistenceError: If the in-memory update succeeds but the
                runtime-overrides file cannot be written.
        """
        with self._lock:
            by_name: dict[str, PluginConfig] = {p.name: p for p in self.config.harness.plugins}
            for plugin in plugins:
                if plugin.name in by_name:
                    existing = by_name[plugin.name]
                    for field in plugin.model_fields_set:
                        setattr(existing, field, getattr(plugin, field))
                else:
                    by_name[plugin.name] = plugin
            merged = list(by_name.values())
            self.config.harness.plugins = merged
            self._plugins.reconfigure(merged)
            self._register_plugin_mcp_servers()
            self.context_builder.plugin_manager = self._plugins
            if not self._save_runtime_overrides():
                raise ConfigPersistenceError(
                    "Plugins config updated in memory but persistence failed"
                )
            return "Plugins config updated"

    def get_config(self) -> dict[str, Any]:
        """Return the current live runtime configuration (excluding secrets)."""
        with self._lock:
            data = self.config.model_dump(mode="json", exclude={"secrets"})
            # Redact nested credentials that can be loaded from environment variables.
            telegram = data.get("harness", {}).get("telegram")
            if isinstance(telegram, dict):
                telegram["token"] = "***"
            hindsight = data.get("harness", {}).get("memory", {}).get("hindsight")
            if isinstance(hindsight, dict):
                hindsight["api_key"] = "***"
            return data

    def update_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a partial runtime configuration update.

        Supported top-level keys: ``telegram`` and ``plugins``. Other keys are
        ignored. The updated live configuration is returned.

        Raises:
            ConfigPersistenceError: If the in-memory update succeeds but the
                runtime-overrides file cannot be written.
        """
        with self._lock:
            if "telegram" in patch:
                self.update_telegram_config(TelegramConfig(**patch["telegram"]))
            if "plugins" in patch:
                self.update_plugins_config([PluginConfig(**p) for p in patch["plugins"]])
            return self.get_config()

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
        return self.context_builder.build_system_notice(persona, recall, chat_status)

    def _trim_reply_quote_to(self, quote: str, limit: int) -> str:
        """Trim a reply-to quote to a given budget, with a truncation marker."""
        return self.context_builder.trim_reply_quote_to(quote, limit)

    def _trim_reply_quote(self, quote: str) -> str:
        """Trim a reply-to quote to the configured budget, with a truncation marker."""
        return self.context_builder.trim_reply_quote(quote)

    def _telegram_message_registry_path(self, chat_id: str) -> Path:
        return self._chat_dir(chat_id) / "telegram_messages.jsonl"

    def _load_telegram_message_registry(self, chat_id: str) -> dict[int, dict[str, Any]]:
        path = self._telegram_message_registry_path(chat_id)
        if not path.exists():
            return {}
        entries: dict[int, dict[str, Any]] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            message_id = entry.get("message_id")
            if message_id is not None:
                entries[message_id] = entry
        return entries

    def _format_user_message(
        self,
        user_message: str,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        chat_id: str | None = None,
    ) -> str:
        """Wrap the user message with a labeled reply-to reference if present."""
        return self.context_builder.format_user_message(
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
        record = self._active_record(chat_id)
        pctx = self.context_builder.build_first(
            chat_id,
            user_message,
            record,
            model=model,
            reply_to=reply_to,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
            continuation_anchor=continuation_anchor,
        )
        return pctx.prompt, pctx.notice, pctx.memory_flags

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
        record = self._active_record(chat_id)
        pctx = self.context_builder.build_follow_up(
            chat_id,
            user_message,
            record,
            reply_to=reply_to,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
            continuation_anchor=continuation_anchor,
        )
        return pctx.prompt

    def _model(self, record: SessionRecord | None) -> str:
        return record.model if record else self.config.engine.model

    def resolve_model(
        self,
        chat_id: str,
        user_message: str,
        record: SessionRecord | None = None,
    ) -> ModelRoute:
        """Resolve the model for a user message, checking budget and lane rules."""
        cumulative_tokens = 0
        if record and record.cumulative_metrics:
            cumulative_tokens = record.cumulative_metrics.get("total_tokens", 0)
        else:
            with self._lock:
                per_chat = self._per_chat_metrics.get(chat_id, {})
                cumulative_tokens = per_chat.get("total_tokens", 0)
        return self._router.resolve(user_message, cumulative_tokens)

    def routing_context(
        self,
        chat_id: str,
        record: SessionRecord | None = None,
    ) -> dict[str, Any]:
        """Return the current routing/budget context for a chat."""
        cumulative_tokens = 0
        if record and record.cumulative_metrics:
            cumulative_tokens = record.cumulative_metrics.get("total_tokens", 0)
        else:
            with self._lock:
                per_chat = self._per_chat_metrics.get(chat_id, {})
                cumulative_tokens = per_chat.get("total_tokens", 0)
        return self._router.budget_status(cumulative_tokens)

    @staticmethod
    def _partial_notice(result: TurnResult, continue_word: str = "Continue") -> str:
        """Return a user-facing notice for an interrupted/partial ACP turn."""
        if result.timed_out or result.stop_reason == "timeout":
            return (
                f"The agent reached the time limit before completing its reply. A partial result is shown. "
                f"Reply `{continue_word}` to keep going, or tell me what to change."
            )
        if result.cancelled or result.stop_reason == "cancelled":
            return (
                f"The agent was stopped before completing its reply. A partial result is shown. "
                f"Reply `{continue_word}` to keep going, or tell me what to change."
            )
        if result.partial:
            return (
                f"The agent reached the time limit before completing its reply. A partial result is shown. "
                f"Reply `{continue_word}` to keep going, or tell me what to change."
            )
        return ""

    def is_continuation_message(self, user_message: str) -> bool:
        """Return True if the user message is a continuation trigger."""
        return self.context_builder.is_continuation_message(user_message)

    def _continuation_anchor(self, record: SessionRecord | None, user_message: str) -> str | None:
        """Return a prompt anchor when resuming an interrupted turn."""
        return self.context_builder.continuation_anchor(record, user_message)

    def _turn_number(self, record: SessionRecord | None) -> int:
        return record.turn_number if record else 0

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
        cwd = self._chat_dir(chat_id)
        now = time.time()
        return SessionRecord(
            chat_id=chat_id,
            session_number=session_number,
            session_id=session_id,
            model=model,
            persona=self.config.persona.name,
            cwd=str(cwd),
            created_at=now,
            updated_at=now,
            turn_number=0,
            label=label,
            parent=parent,
            persona_memory_exceeded=memory_flags.get("persona_memory_exceeded", False),
            chat_memory_exceeded=memory_flags.get("chat_memory_exceeded", False),
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
        cwd = self._chat_dir(chat_id)
        cwd.mkdir(parents=True, exist_ok=True)
        self.skills.sync_to_chat(chat_id, cwd, skill_names or self._active_skill_names(chat_id))
        request = TurnRequest(
            prompt=prompt,
            cwd=cwd,
            model=model,
            mcp_servers=mcp_servers or self._active_mcp_servers(chat_id),
            soft_timeout=self.config.engine.soft_timeout,
        )
        result = self.engine.prompt(request, on_chunk=on_chunk, on_update=on_update)
        if result.session_id is None:
            from diploid_agent.acp_client import AcpError

            raise AcpError(
                "acp.create_session",
                {"message": "Engine returned a result without a session id"},
            )
        return result, result.session_id

    def _check_chat_memory_transition(self, chat_id: str, record: SessionRecord) -> str | None:
        """Return a system notice if the chat memory just exceeded its cap."""
        cap = self.config.harness.memory.max_chat_memory_chars
        if not cap:
            return None

        mgr = self._memory_manager(chat_id)
        path = mgr.chat_memory_path
        if not path or not path.exists():
            return None

        total = len(path.read_text())
        previous = record.chat_memory_exceeded
        exceeded = total > cap
        record.chat_memory_exceeded = exceeded

        if previous == exceeded:
            return None

        default_notice: str | None = None
        if not previous and exceeded:
            default_notice = (
                f"Chat memory ({path}) has grown beyond the context budget "
                f"({total} > {cap} characters). Older content is still saved "
                f"but is not loaded. You can ask me to read and prune it."
            )

        ctx = self._plugins.on_chat_memory_transition(
            chat_id,
            MemoryTransitionContext(
                chat_id=chat_id,
                record=record,
                kind="chat",
                path=path,
                total=total,
                cap=cap,
                notice=default_notice,
                suppress_default=False,
            ),
        )
        if ctx.suppress_default:
            if ctx.notice and ctx.notice != default_notice:
                return ctx.notice
            return None
        return ctx.notice or default_notice

    def _check_persona_memory_transition(self, record: SessionRecord) -> str | None:
        """Return a system notice if the persona memory just exceeded its cap."""
        cap = self.config.harness.memory.max_persona_memory_chars
        if not cap:
            return None

        path = self.config.persona.profile_root / self.config.persona.memory_filename
        if not path.exists():
            return None

        total = len(path.read_text())
        previous = record.persona_memory_exceeded
        exceeded = total > cap
        record.persona_memory_exceeded = exceeded

        if previous == exceeded:
            return None

        default_notice: str | None = None
        if not previous and exceeded:
            default_notice = (
                f"Persona memory ({path}) has grown beyond the context budget "
                f"({total} > {cap} characters). Older content is still saved "
                f"but is not loaded. You can ask me to read and prune it."
            )

        chat_id = record.chat_id
        ctx = self._plugins.on_persona_memory_transition(
            chat_id,
            MemoryTransitionContext(
                chat_id=chat_id,
                record=record,
                kind="persona",
                path=path,
                total=total,
                cap=cap,
                notice=default_notice,
                suppress_default=False,
            ),
        )
        if ctx.suppress_default:
            if ctx.notice and ctx.notice != default_notice:
                return ctx.notice
            return None
        return ctx.notice or default_notice

    # ---------------------------------------------------------------- public API

    def get_model(self, chat_id: str) -> str:
        """Return the model currently used for a chat, or the default."""
        with self._lock:
            return self._model(self._active_record(chat_id))

    def _context_usage(self, record: SessionRecord) -> dict[str, Any]:
        """Return context-window and prompt-budget usage for a chat record."""
        context_window: int | None = None
        try:
            context_window_fn = getattr(self.engine, "model_context_window", None)
            if context_window_fn is not None:
                context_window = context_window_fn(record.model)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to resolve context window for %s: %s", record.model, exc)

        last_turn = record.last_turn_metrics or {}
        cumulative = record.cumulative_metrics or {}

        def _enrich(turn: dict[str, Any]) -> dict[str, Any]:
            enriched = dict(turn)
            if context_window:
                input_tokens = turn.get("input_tokens", 0) or 0
                total_tokens = turn.get("total_tokens", 0) or 0
                enriched["input_percent"] = round(input_tokens / context_window * 100, 2)
                enriched["total_percent"] = round(total_tokens / context_window * 100, 2)
                enriched["available_tokens"] = max(0, context_window - input_tokens)
            return enriched

        return {
            "model": record.model,
            "context_window": context_window,
            "last_turn": _enrich(last_turn),
            "cumulative": cumulative,
            "memory_budgets": {
                "max_chat_memory_chars": self.config.harness.memory.max_chat_memory_chars,
                "max_persona_memory_chars": self.config.harness.memory.max_persona_memory_chars,
                "max_short_term_chars": self.config.harness.memory.max_short_term_chars,
                "max_reply_quote_chars": self.config.harness.memory.max_reply_quote_chars,
                "hindsight_max_recall_tokens": self.config.harness.memory.hindsight.max_recall_tokens,
            },
            "memory_exceeded": {
                "chat_memory_exceeded": record.chat_memory_exceeded,
                "persona_memory_exceeded": record.persona_memory_exceeded,
            },
        }

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
        """Start a background ACP subagent and return a dispatch id.

        The subagent runs in a fresh AcpEngine, so it survives the parent turn
        being stopped or killed. When it completes, the harness starts a new
        turn for the chat via the existing dispatch/continue flow.
        """
        record = self._active_record(chat_id)
        if record is None:
            return ChatResult(reply="No active session for this chat.")

        use_model = model or self._model(record)
        started_at = time.time()
        dispatch = self.dispatch_store.add(
            chat_id,
            record.session_id,
            context=context or prompt[:200],
            dispatch_type="subagent",
            started_at=started_at,
        )
        task = Task(
            name="subagent",
            type=TaskType.SUBAGENT,
            prompt=prompt,
            chat_id=chat_id,
            acp_model=use_model,
            acp_timeout=acp_timeout,
            mcp_servers=self.mcp.enabled_servers(chat_id, self._active_mcp_server_names(chat_id)),
            dispatch_id=dispatch.id,
            cwd=cwd or self._chat_dir(chat_id),
        )
        plan = self.plan_manager.create_plan(
            f"subagent-{dispatch.id[:8]}",
            description="Background subagent task",
            chat_id=chat_id,
            tasks=[task],
        )

        wake_id = f"wake-{dispatch.id}"
        self.wake_queue.enqueue(
            WakeEvent(
                id=wake_id,
                chat_id=chat_id,
                reason="dispatch",
                priority=1,
                scheduled_at=time.time(),
                created_at=time.time(),
                silent=False,
                payload={"dispatch_id": dispatch.id, "notify": True, "retry_after": 5.0},
                ready=False,
            )
        )

        self.task_engine.start_task(plan.id, task.id)
        return ChatResult(
            reply="Subagent started. I'll report back when it finishes.",
            dispatch_id=dispatch.id,
        )

    def _persist_subagent_result(self, dispatch: Dispatch, result: str) -> Path | None:
        """Write the full subagent result to the chat session dir and update the dispatch.

        Returns the path to the written file, or ``None`` on write failure.
        """
        chat_id = dispatch.chat_id
        chat_dir = self._chat_dir(chat_id)
        result_dir = chat_dir / "subagent-results"
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"subagent-{dispatch.id}.md"
        try:
            result_path.write_text(result, encoding="utf-8")
        except Exception:
            logger.exception("Failed to write subagent result to %s", result_path)
            return None

        summary = self._extract_summary(result, max_chars=240)
        self.dispatch_store.set_result(
            dispatch.id,
            result,
            summary=summary,
            finished_at=time.time(),
            full_result_path=str(result_path),
        )
        return result_path

    def _complete_subagent_task(self, task: Task) -> None:
        """Store the subagent result, persist the full output, and make the wake ready.

        The dispatch is kept PENDING for normal completions so `continue_turn`
        can consume it. For timeout/cancelled subagents the status is updated to
        TIMEOUT/CANCELLED, but `continue_turn` and `wake` still allow those
        states to proceed. The full result is written to the chat session dir.
        """
        if task.dispatch_id is None or task.chat_id is None:
            return
        dispatch = self.dispatch_store.get(task.dispatch_id)
        if dispatch is None:
            return

        result = task.result or task.log or "(no result)"
        if task.status == TaskStatus.FAILED and not result:
            result = "(subagent failed)"

        status, stop_reason, is_timeout, is_cancelled, is_partial = self._subagent_terminal_state(
            task
        )
        self._persist_subagent_result(dispatch, result)

        # Re-read the dispatch after _persist_subagent_result so summary/full_result_path are fresh.
        dispatch = self.dispatch_store.get(task.dispatch_id) or dispatch
        summary = dispatch.summary or self._extract_summary(result, max_chars=240)

        self.dispatch_store.set_result(
            task.dispatch_id,
            result,
            summary=summary,
            finished_at=time.time(),
            status=status,
            stop_reason=stop_reason,
            cancelled=is_cancelled,
            partial=is_partial,
            timed_out=is_timeout,
            full_result_path=dispatch.full_result_path,
        )

        if is_timeout or is_cancelled:
            dispatch = self.dispatch_store.get(task.dispatch_id) or dispatch
            self._notify_subagent_timeout(
                task, task.chat_id, task.dispatch_id, dispatch, summary, is_timeout
            )

        wake_id = f"wake-{task.dispatch_id}"
        try:
            wake = self.wake_queue.get(wake_id)
            if wake is not None and wake.payload is not None:
                wake.payload["result"] = result
            self.wake_queue.ready(wake_id, now=time.time())
        except Exception:
            logger.exception("Failed to mark subagent wake %s ready", wake_id)

    @staticmethod
    def _extract_summary(text: str, max_chars: int = 240) -> str:
        """Return a short, one-line summary of a subagent result.

        Prefers the first markdown heading if present, otherwise the first
        block of text up to ``max_chars``.
        """
        if not text:
            return ""
        text = text.strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        first = lines[0]
        if first.startswith("#"):
            return first.lstrip("#").strip()[:max_chars]
        return text[:max_chars]

    @staticmethod
    def _human_duration(seconds: float) -> str:
        """Return a compact, human-readable duration."""
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"{seconds}s"
        minutes, secs = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m {secs}s"

    def _subagent_terminal_state(
        self, task: Task
    ) -> tuple[DispatchStatus | None, str | None, bool, bool, bool]:
        """Map a finished subagent task to its dispatch status and reason flags.

        Returns ``(dispatch_status, stop_reason, is_timeout, is_cancelled, is_partial)``.
        ``dispatch_status`` is ``None`` for normal completions so the dispatch
        stays PENDING until ``continue_turn`` completes it. ``FAILED`` is set for
        genuine failures so the continuation prompt can report them.
        """
        log = (task.log or "").lower()
        if (
            task.timed_out
            or task.stop_reason == "timeout"
            or (task.status == TaskStatus.FAILED and "timeout" in log)
        ):
            return (
                DispatchStatus.TIMEOUT,
                "timeout",
                True,
                False,
                task.partial or True,
            )
        if (
            task.cancelled
            or task.stop_reason == "cancelled"
            or (task.status == TaskStatus.FAILED and "cancel" in log)
        ):
            return (
                DispatchStatus.CANCELLED,
                "cancelled",
                False,
                True,
                task.partial or True,
            )
        if task.status == TaskStatus.FAILED:
            return (
                DispatchStatus.FAILED,
                "failed",
                False,
                False,
                task.partial or False,
            )
        return (
            None,
            None,
            False,
            False,
            task.partial or False,
        )

    def _notify_subagent_timeout(
        self,
        task: Task,
        chat_id: str,
        dispatch_id: str,
        dispatch: Dispatch,
        summary: str,
        is_timeout: bool,
    ) -> None:
        """Send a proactive user-visible notification for a timeout/cancelled subagent."""
        start = task.started_at if task.started_at is not None else dispatch.started_at
        finished = task.completed_at if task.completed_at is not None else dispatch.finished_at
        if start is None or finished is None:
            duration = "unknown"
        else:
            duration = self._human_duration(finished - start)
        reason = "timed out" if is_timeout else "was cancelled"
        text = f"Subagent {dispatch_id} {reason} after {duration}. Partial summary: {summary}"
        chat_result = ChatResult(
            reply=text,
            dispatch_id=dispatch_id,
            session_id=dispatch.session_id if dispatch else None,
        )
        if self._outbox_delivery_enabled:
            self._enqueue_outbox(chat_id, chat_result)
            return
        self._safe_notifier_send(chat_id, text)

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
        """Return the status of all background subagents for a chat."""
        plans = self.plan_manager.list_plans(chat_id=chat_id)
        subagents: list[dict[str, Any]] = []
        for plan in plans:
            for task in plan.tasks:
                if task.type != TaskType.SUBAGENT:
                    continue
                dispatch = self.dispatch_store.get(task.dispatch_id) if task.dispatch_id else None
                status = self._subagent_status_name(task, dispatch)
                summary = self._subagent_summary(task, dispatch)
                started_at = task.started_at or (dispatch.started_at if dispatch else None)
                finished_at = task.completed_at or (dispatch.finished_at if dispatch else None)
                subagents.append(
                    {
                        "plan_id": plan.id,
                        "task_id": task.id,
                        "dispatch_id": task.dispatch_id,
                        "name": plan.name,
                        "status": status,
                        "continued": dispatch.status == DispatchStatus.COMPLETED if dispatch else False,
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "summary": summary,
                        "prompt_snippet": (task.prompt or "")[:200],
                    }
                )
        return {"chat_id": chat_id, "subagents": subagents}

    def plan_list(self) -> list[Plan]:
        """List all plans."""
        return self.plan_manager.list_plans()

    def plan_get(self, plan_id: str) -> Plan | None:
        """Return one plan by id."""
        return self.plan_manager.get_plan(plan_id)

    # ---------------------------------------------------------------- pruning

    def _prune_chat(self, chat_id: str) -> None:
        """Delete archived sessions older than the prune window."""
        if not self.config.harness.session_prune_enabled:
            return
        state = self._chat_state(chat_id)
        active = self._active_record(chat_id)
        cutoff = time.time() - (self.config.harness.session_prune_days * 86400)
        to_remove: list[int] = []
        for number, record in state.sessions.items():
            if active and number == active.session_number:
                continue
            if record.updated_at >= cutoff:
                continue
            to_remove.append(number)
        for number in to_remove:
            archive = self._archive_dir(chat_id, number)
            if archive.exists():
                shutil.rmtree(archive)
            del state.sessions[number]

    def _prune_and_compact(self, chat_id: str) -> None:
        if self.config.harness.session_prune_enabled:
            self._prune_chat(chat_id)
            self._compact_store()

    def _prune_all(self) -> None:
        if not self.config.harness.session_prune_enabled:
            return
        for chat_id in list(self._store.keys()):
            self._prune_chat(chat_id)
        self._compact_store()
