"""AgentRuntime: service container and non-turn public API."""

from __future__ import annotations

import logging
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from diploid_agent.acp_client import AcpLifecycleLog
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
from diploid_agent.dispatch import Dispatch, DispatchStore
from diploid_agent.engine import AgentEngine, build_engine
from diploid_agent.engine.router import ModelRouter
from diploid_agent.locking import locked
from diploid_agent.mcp import McpManager
from diploid_agent.memory import MemoryManager
from diploid_agent.metrics import MetricsCollector
from diploid_agent.models import (
    ActiveTurn,
    ChatResult,
    ChatState,
    RuntimeStatus,
    SessionRecord,
)
from diploid_agent.plan.manager import PlanManager
from diploid_agent.plan.models import Plan, Task
from diploid_agent.plugin_incidents import PluginIncidentStore
from diploid_agent.plugins import PluginManager
from diploid_agent.runtime.actions import RuntimeActions
from diploid_agent.runtime.auto_continue import RuntimeAutoContinue
from diploid_agent.runtime.config_manager import RuntimeConfigManager
from diploid_agent.runtime.cron_service import CronService
from diploid_agent.runtime.event_bus import EventBus
from diploid_agent.runtime.ingress import RuntimeIngress
from diploid_agent.runtime.instance import InstanceManager
from diploid_agent.runtime.lifecycle import RuntimeLifecycle
from diploid_agent.runtime.mcp_skills import RuntimeMcpSkills
from diploid_agent.runtime.metrics import RuntimeMetrics
from diploid_agent.runtime.outbox import RuntimeOutbox
from diploid_agent.runtime.planning import RuntimePlanning
from diploid_agent.runtime.plugins import RuntimePlugins
from diploid_agent.runtime.prompts import RuntimePrompts
from diploid_agent.runtime.restart import RuntimeRestart
from diploid_agent.runtime.state import RuntimeState
from diploid_agent.runtime.store import ChatSessionStore
from diploid_agent.runtime.subagent import RuntimeSubagent
from diploid_agent.runtime.timer_service import TimerService
from diploid_agent.runtime.typing import RuntimeTyping
from diploid_agent.runtime.wake_queue import WakeQueue
from diploid_agent.skills import SkillManager
from diploid_agent.task.engine import TaskEngine
from diploid_agent.text import human_duration
from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.turn import TurnController

if TYPE_CHECKING:
    from diploid_agent.notifier import Notifier
    from diploid_agent.transport.ingress import IngressHandler

logger = logging.getLogger(__name__)


class AgentRuntime(RuntimeAPI):
    """Persistent chat runtime backed by Devin ACP.

    Two surfaces live on this class: the ``RuntimeAPI`` methods transports
    call (``process``, ``continue_turn``, ``outbox_pop``, ...) and a
    delegation layer used by ``TurnController`` and sibling runtime
    components (components are constructed here, so callers reach them
    through it rather than importing each other). Members prefixed ``_``
    are runtime-internal; the session-store mirrors (``active_record``,
    ``append_record``, ``chat_dir``, ...) and the named component delegates
    (``schedule_draining_restart``, ``float_mesh_to_telegram``) are the
    public half of that seam.
    """

    def __init__(self, config: Config):
        self.config = config
        self._init_core()
        self._init_stores()
        self._init_services()
        self._init_plugins()
        self._init_components()
        self.notifier = self._create_notifier()

    def _init_core(self) -> None:
        """Identity, engine, lock, and shared mutable state."""
        config = self.config
        self.sessions_root = Path(config.harness.sessions_root).expanduser()
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.store_path = Path(config.harness.session_store_path).expanduser()
        self.lifecycle_log = AcpLifecycleLog(self.store_path.parent / "acp-lifecycle.jsonl")
        self.metrics = MetricsCollector()
        self.engine = self._create_engine(metrics=self.metrics)
        self._lock = threading.RLock()
        self._state = RuntimeState()
        self.instance_id = f"harness-{uuid.uuid4().hex[:12]}"
        self.instance_started_at = time.time()
        # Shared dicts injected into components by reference.
        self._active_turns: dict[str, ActiveTurn] = {}
        # Chats with a session-mutating op in flight (switch/new/resume/branch).
        # Unlike an ActiveTurn these hold no per-chat lock — the runtime lock's
        # `_call_unlocked` windows would otherwise let a turn or second op slip
        # in mid-flight and corrupt the session record.
        self._session_ops: set[str] = set()
        self._active_chat_skills: dict[str, set[str]] = {}
        self._memory_managers: dict[str, MemoryManager] = {}
        self._plan_conclusion_enqueued: set[str] = set()
        self._router = ModelRouter(config)

    def _init_stores(self) -> None:
        """Persistence and early config: the config manager is built here so
        runtime overrides load before plugins/services read ``config.harness``;
        its service deps therefore arrive as late-bound ``*_fn`` callables."""
        config = self.config
        self._config_manager = RuntimeConfigManager(
            config=config,
            lock=self._lock,
            active_turns=self._active_turns,
            instance_id=self.instance_id,
            instance_started_at=self.instance_started_at,
            plan_manager_fn=lambda: self.plan_manager,
            event_bus_fn=lambda: self.event_bus,
            timer_service_fn=lambda: self.timer_service,
            task_engine_fn=lambda: self.task_engine,
            wake_queue_fn=lambda: self.wake_queue,
            plugins_fn=lambda: self._plugins,
            runtime_plugins_fn=lambda: self._runtime_plugins,
            context_builder_fn=lambda: self.context_builder,
            recreate_notifier_fn=lambda: setattr(self, "notifier", self._create_notifier()),
            save_overrides_fn=lambda: self._save_runtime_overrides(),
        )
        self._chat_store = ChatSessionStore(
            sessions_root=self.sessions_root,
            store_path=self.store_path,
            lock=self._lock,
            config=config,
            plugins_fn=lambda: self._plugins,
            context_builder_fn=lambda: self.context_builder,
        )
        self._store = self._chat_store.states
        self._runtime_metrics = RuntimeMetrics(
            metrics=self.metrics,
            store=self._store,
            lock=self._lock,
            config=config,
            engine_fn=lambda: self.engine,
            instance_started_at=self.instance_started_at,
            plugins_fn=lambda: self._plugins,
            context_builder_fn=lambda: self.context_builder,
            notifier_fn=lambda: self.notifier,
        )

        # Load external plugin search paths before PluginManager imports anything.
        for plugin_path in self.config.harness.plugin_paths:
            if plugin_path.exists() and str(plugin_path) not in sys.path:
                sys.path.append(str(plugin_path))

        self._chat_store.load_store()
        dispatch_store_path = Path(config.harness.dispatch_store_path).expanduser()
        self.dispatch_store = DispatchStore(dispatch_store_path)

        wake_store_path = Path(config.harness.wake_store_path).expanduser()
        self.wake_queue = WakeQueue(wake_store_path)

        self._config_manager._load_runtime_overrides()

    def _init_services(self) -> None:
        """Background services: event bus, plans/tasks, timers, outbox."""
        config = self.config
        plan_root = Path(config.harness.plan.root).expanduser()
        plan_root.mkdir(parents=True, exist_ok=True)
        self.event_bus = EventBus()
        self.event_bus.start()
        self.plan_manager = PlanManager(plan_root)
        self._typing = RuntimeTyping(notifier_fn=lambda: self.notifier)
        self.task_engine = TaskEngine(
            self.plan_manager,
            self.event_bus,
            engine=self.engine,
            config=self.config,
            task_config=config.harness.task,
            on_task_start=self._typing.on_task_started,
            on_task_done=self._typing.on_task_done,
            on_service_restart=self._on_service_restart,
        )

        self.timer_service = TimerService(
            self.wake_queue,
            self.event_bus,
            config=self.config.harness.timer,
        )

        self.cron_service = CronService(
            config=config,
            plan_manager=self.plan_manager,
            task_engine=self.task_engine,
            event_bus=self.event_bus,
            sessions_root=self.sessions_root,
            wake_queue=self.wake_queue,
        )

        self._outbox = RuntimeOutbox(
            config=config,
            metrics=self.metrics,
            store=self._store,
            lock=self._lock,
            notifier_fn=lambda: self.notifier,
        )

        self.instance_manager = InstanceManager(
            self.sessions_root,
            self.instance_id,
            ttl_seconds=config.harness.instance_ttl_seconds,
        )

        if config.harness.session_prune_enabled:
            self.prune_all()
        self._runtime_metrics._rehydrate_metrics()

        # Durable record of plugin incidents (sandbox, lifecycle, health, watchdog).
        self._incidents = PluginIncidentStore(self.store_path.parent / "plugin-incidents.jsonl")

        self._auto_continue = RuntimeAutoContinue()

    def _init_plugins(self) -> None:
        """Plugin stack: PluginManager, restart supervisor, MCP/skill wiring.

        Plugins can declare MCP servers and skills; they are added before
        ``McpManager`` sees the config.
        """
        config = self.config
        plugins = list(self.config.harness.plugins)
        self._plugins = PluginManager(
            plugins=plugins,
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

        self._restart = RuntimeRestart(
            config=self.config,
            state=self._state,
            lock=self._lock,
            wake_queue=self.wake_queue,
            incidents=self._incidents,
            plugins=self._plugins,
            chat_store=self._chat_store,
            active_turns=self._active_turns,
            session_ops=self._session_ops,
            store=self._store,
            instance_id=self.instance_id,
            instance_started_at=self.instance_started_at,
            suppress_auto_continue_fn=lambda *a, **k: self.suppress_auto_continue(*a, **k),
            unit_exists_fn=lambda s: self._unit_exists(s),
            memory_manager=self._memory_manager,
            notify_fn=self._notify_agent_restart,
        )

        # Ingress handlers for pluggable transport protocols (e.g. mesh).
        self._ingress_handlers: dict[str, IngressHandler] = {}

        self.mcp = McpManager(config)
        self.skills = SkillManager(
            personas_root=Path(self.config.persona.profile_root).parent,
            shared_root=self.config.harness.skills.shared_root,
            chat_cwd_root=self.sessions_root,
            active_persona=self.config.persona.name,
            persona_profile_root=self.config.persona.profile_root,
        )
        self._mcp_skills = RuntimeMcpSkills(
            mcp=self.mcp,
            skills=self.skills,
            plugins=self._plugins,
            chat_store=self._chat_store,
            active_chat_skills=self._active_chat_skills,
            lock=self._lock,
            config=config,
        )
        self._runtime_plugins = RuntimePlugins(
            plugins=self._plugins,
            incidents=self._incidents,
            config_manager=self._config_manager,
            lock=self._lock,
            config=config,
            chat_store=self._chat_store,
            lifecycle_log=self.lifecycle_log,
            context_builder_fn=lambda: self.context_builder,
        )
        self._runtime_plugins._register_plugin_mcp_servers()

    def _init_components(self) -> None:
        """Turn-facing components: context builder, prompts, planning,
        subagent, turn controller, actions, ingress, lifecycle."""
        config = self.config
        # Prompt assembly is delegated to a dedicated builder.
        self.context_builder = ContextBuilder(
            self.config,
            self._plugins,
            self._memory_manager,
            self.skills,
            self._mcp_skills._active_skill_names,
            context_window_fn=self._engine_context_window,
            lifecycle_log=self.lifecycle_log,
            chat_store=self._chat_store,
        )
        self.context_builder.metrics = self._runtime_metrics._per_chat_metrics

        self._prompts = RuntimePrompts(
            config=config,
            lock=self._lock,
            chat_store=self._chat_store,
            context_builder=self.context_builder,
            mcp_skills=self._mcp_skills,
            plugins=self._plugins,
            skills=self.skills,
            engine_fn=lambda: self.engine,
            router=self._router,
            runtime_metrics=self._runtime_metrics,
            memory_manager=self._memory_manager,
        )
        self._planning = RuntimePlanning(
            wake_queue=self.wake_queue,
            plan_conclusion_enqueued=self._plan_conclusion_enqueued,
            context_builder=self.context_builder,
        )
        self._subagent = RuntimeSubagent(
            wake_queue=self.wake_queue,
            plan_manager=self.plan_manager,
            chat_store=self._chat_store,
            task_engine=self.task_engine,
            prompts=self._prompts,
            outbox=self._outbox,
            mcp=self.mcp,
            dispatch_store=self.dispatch_store,
            mcp_skills=self._mcp_skills,
            lock=self._lock,
        )
        self.turn_controller = TurnController(self)
        self._actions = RuntimeActions(
            state=self._state,
            config=config,
            lock=self._lock,
            chat_store=self._chat_store,
            lifecycle_log=self.lifecycle_log,
            memory_manager=self._memory_manager,
            runtime_metrics=self._runtime_metrics,
            prompts=self._prompts,
            outbox=self._outbox,
            plugins=self._plugins,
            subagent=self._subagent,
            restart=self._restart,
            incidents=self._incidents,
            wake_queue=self.wake_queue,
            plan_manager=self.plan_manager,
            task_engine=self.task_engine,
            event_bus=self.event_bus,
            turn_controller=self.turn_controller,
            instance_id=self.instance_id,
            engine_fn=lambda: self.engine,
            call_unlocked_fn=self._call_unlocked,
            suppress_auto_continue_fn=lambda *a, **k: self.suppress_auto_continue(*a, **k),
            acp_client_fn=lambda: getattr(self, "acp_client", None),
        )
        self._ingress = RuntimeIngress(
            config=config,
            lock=self._lock,
            instance_manager=self.instance_manager,
            wake_queue=self.wake_queue,
            dispatch_store=self.dispatch_store,
            turn_controller=self.turn_controller,
            planning=self._planning,
            ingress_handlers=self._ingress_handlers,
        )
        self._lifecycle = RuntimeLifecycle(
            state=self._state,
            config=config,
            lock=self._lock,
            event_bus=self.event_bus,
            wake_queue=self.wake_queue,
            instance_manager=self.instance_manager,
            task_engine=self.task_engine,
            timer_service=self.timer_service,
            cron_service=self.cron_service,
            typing=self._typing,
            restart=self._restart,
            outbox=self._outbox,
            plugins=self._plugins,
            chat_store=self._chat_store,
            memory_managers=self._memory_managers,
            store=self._store,
            ingress_handlers=self._ingress_handlers,
            instance_id=self.instance_id,
            instance_started_at=self.instance_started_at,
            plan_manager=self.plan_manager,
            planning=self._planning,
            subagent=self._subagent,
            runtime_api=self,
        )

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
            lifecycle_log=self.lifecycle_log,
        )

    def _engine_context_window(self, model: str) -> int | None:
        """Context window for ``model`` from the *current* engine.

        Resolved per call because ``runtime.engine`` is rebindable and some
        test engines don't implement ``model_context_window``.
        """
        fn = getattr(self.engine, "model_context_window", None)
        return fn(model) if fn is not None else None

    @property
    def client(self) -> AgentEngine:
        """Backward-compatible alias for the engine."""
        return self.engine

    @client.setter
    def client(self, value: AgentEngine) -> None:
        self.engine = value

    # ------------------------------------------------------- shared state
    # Aliases over ``self._state`` (RuntimeState). Tests and turn code write
    # ``runtime._started`` / ``runtime._last_service_restart_at`` and call
    # ``runtime._restart_draining.set()`` directly; the aliases keep those
    # seams working while components share the box.

    @property
    def _started(self) -> bool:
        return self._state.started

    @_started.setter
    def _started(self, value: bool) -> None:
        self._state.started = value

    @property
    def _restart_draining(self) -> threading.Event:
        return self._state.restart_draining

    @_restart_draining.setter
    def _restart_draining(self, value: threading.Event) -> None:
        self._state.restart_draining = value

    @property
    def _last_service_restart_at(self) -> float:
        return self._state.last_service_restart_at

    @_last_service_restart_at.setter
    def _last_service_restart_at(self, value: float) -> None:
        self._state.last_service_restart_at = value

    @property
    def _service_restart_cooldown_seconds(self) -> float:
        return self._state.service_restart_cooldown_seconds

    @_service_restart_cooldown_seconds.setter
    def _service_restart_cooldown_seconds(self, value: float) -> None:
        self._state.service_restart_cooldown_seconds = value

    def _on_service_restart(self, service: str, reason: str) -> str:
        """Handle a service restart request from the ACP subprocess."""
        return self._restart._on_service_restart(service, reason)

    def _notify_agent_restart(self, chat_id: str, text: str) -> None:
        """Enqueue an operator notice for an agent-initiated restart."""
        try:
            self._enqueue_outbox(chat_id, ChatResult(reply=text))
        except Exception as exc:
            logger.warning("Failed to notify chat %s of agent restart", chat_id, exc_info=exc)

    def _unit_exists(self, service: str) -> bool:
        """Best-effort check that a user unit exists before draining for it."""
        return self._restart._systemd_unit_exists(service)

    def schedule_draining_restart(
        self,
        service: str,
        chat_id: str | None,
        reason: str,
        drain_cap: float = 120.0,
    ) -> bool:
        """Begin the restart drain and schedule the real systemd restart."""
        return self._restart.schedule_draining_restart(
            service, chat_id, reason, drain_cap=drain_cap
        )

    def _arm_restart_watchdog(
        self,
        service: str,
        due_in: float,
        chat_id: str | None,
        margin: float = 60.0,
    ) -> None:
        """Self-heal when a scheduled restart never fires."""
        self._restart._arm_restart_watchdog(service, due_in, chat_id, margin=margin)

    def _wait_for_active_turns(self, timeout: float) -> bool:
        """Block until every ActiveTurn finishes or ``timeout`` expires."""
        return self._restart._wait_for_active_turns(timeout)

    def _flush_plugins_for_restart(self) -> None:
        """Run shutdown/sleeping hooks on every chat so plugin state persists."""
        self._restart._flush_plugins_for_restart()

    def _schedule_systemd_restart(
        self,
        service: str,
        delay: float,
        chat_id: str | None,
        reason: str,
    ) -> None:
        """Run a short-delayed systemd-run that restarts the named service."""
        self._restart._schedule_systemd_restart(service, delay, chat_id, reason)

    def suppress_auto_continue(self, chat_id: str | None = None, seconds: float = 300.0) -> None:
        """Suppress auto-continue for a chat or globally for a number of seconds."""
        self._auto_continue.suppress(chat_id=chat_id, seconds=seconds)

    def is_auto_continue_suppressed(self, chat_id: str) -> bool:
        """Return True if auto-continue should be suppressed for this chat."""
        return self._auto_continue.is_suppressed(chat_id)

    def is_continuation_message(self, user_message: str) -> bool:
        """Return True if the user message is a continuation trigger."""
        return self._prompts.is_continuation_message(user_message)

    def _create_notifier(self):
        """Create the runtime's configured notifier."""
        return self._outbox._create_notifier()

    @property
    def _outbox_delivery_enabled(self) -> bool:
        return self._outbox._outbox_delivery_enabled

    def _enqueue_outbox(self, chat_id: str, chat_result: ChatResult) -> None:
        """Put a final ChatResult in the outbox for the transport to deliver."""
        self._outbox._enqueue_outbox(chat_id, chat_result)

    def _safe_notifier_send(
        self, chat_id: str, text: str, notifier: Notifier | None = None
    ) -> None:
        """Send a notification, swallowing exceptions and logging them."""
        self._outbox._safe_notifier_send(chat_id, text, notifier=notifier)

    def _deliver_chat_result(self, chat_id: str, chat_result: ChatResult) -> None:
        """Send a final ChatResult through the configured delivery channel."""
        self._outbox._deliver_chat_result(chat_id, chat_result)

    def float_mesh_to_telegram(
        self,
        chat_id: str,
        *,
        sender: str,
        recipient: str,
        body: str,
        action: str,
        reply: str,
        msg_id: str,
    ) -> None:
        """Mirror a sent mesh message to Telegram as a system notice."""
        self._outbox.float_mesh_to_telegram(
            chat_id,
            sender=sender,
            recipient=recipient,
            body=body,
            action=action,
            reply=reply,
            msg_id=msg_id,
        )

    def outbox_pop(
        self,
        chat_id: str | None = None,
        wait: float = 0.0,
        return_chat_id: bool = False,
    ) -> ChatResult | tuple[str, ChatResult] | None:
        """Return the next ChatResult for a chat, blocking up to ``wait`` seconds."""
        return self._outbox.outbox_pop(chat_id, wait=wait, return_chat_id=return_chat_id)

    def _call_unlocked(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Release the RLock while running a long call, then reacquire.

        Release *every* level, not just one: callers can hold the lock
        multiple times (``@_locked`` methods invoking other ``@_locked``
        methods stack acquisitions on the same RLock).  Leaving even one
        level held during an engine call would block ``_on_chunk`` /
        ``_on_update``, which the ACP transport dispatches from its reader
        path -- a held runtime lock starves the stdout reader and wedges the
        ACP child on a full pipe.
        """
        released = 0
        while self._lock._is_owned():
            try:
                self._lock.release()
                released += 1
            except RuntimeError:
                break
        try:
            return fn(*args, **kwargs)
        finally:
            for _ in range(released):
                self._lock.acquire()

    def call_engine_unlocked(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Public wrapper that releases the runtime RLock around a call.

        Plugins can use this from gate hooks (e.g. ``before_turn``) to call
        the engine without blocking the runtime.
        """
        return self._call_unlocked(fn, *args, **kwargs)

    # ---------------------------------------------------------------- load/save

    def load_store(self) -> None:
        self._chat_store.load_store()

    def append_record(self, record: SessionRecord) -> None:
        self._chat_store.append_record(record)

    def compact_store(self) -> None:
        self._chat_store.compact_store()

    # ---------------------------------------------------------------- session dirs

    def chat_dir(self, chat_id: str) -> Path:
        return self._chat_store.chat_dir(chat_id)

    def _snapshot_plugin_states(self, chat_id: str) -> None:
        """Snapshot durable plugin and body state files before a transport restart."""
        self._runtime_plugins._snapshot_plugin_states(chat_id)

    def _restore_plugin_states(self, chat_id: str) -> None:
        """Restore durable plugin and body state files after a transport wake."""
        self._runtime_plugins._restore_plugin_states(chat_id)

    def archive_dir(self, chat_id: str, session_number: int) -> Path:
        return self._chat_store.archive_dir(chat_id, session_number)

    def durable_file_names(self) -> set[str]:
        return self._chat_store.durable_file_names()

    def copy_session_dir(self, source: Path, target: Path) -> None:
        self._chat_store.copy_session_dir(source, target)

    def archive_active_session(self, chat_id: str, record: SessionRecord) -> None:
        """Copy the active directory into the archive for `record`."""
        self._chat_store.archive_active_session(chat_id, record)

    def clear_active_session(self, chat_id: str) -> None:
        self._chat_store.clear_active_session(chat_id)

    # ---------------------------------------------------------------- active state

    def chat_state(self, chat_id: str) -> ChatState:
        return self._chat_store.chat_state(chat_id)

    def active_record(self, chat_id: str) -> SessionRecord | None:
        return self._chat_store.active_record(chat_id)

    def next_session_number(self, chat_id: str) -> int:
        return self._chat_store.next_session_number(chat_id)

    def generate_label(self, chat_id: str, user_message: str) -> str:
        """Auto-generate a short label from the first user message."""
        return self._chat_store.generate_label(chat_id, user_message)

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
        health = self._runtime_metrics.health()
        health["pending_restart"] = self._restart.pending_restart()
        return health

    def _hindsight_health(self) -> bool:
        """Probe the Hindsight backend health endpoint."""
        return self._runtime_metrics._hindsight_health()

    # ---------------------------------------------------------------- helpers

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

    def _record_restart_memory(self, chat_id: str, reason: str | None = None) -> None:
        """Record a brief ACP restart observation for memory_recall."""
        self._restart._record_restart_memory(chat_id, reason=reason)

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

    def plugin_toggle(self, name: str, enabled: bool, chat_id: str | None = None) -> ChatResult:
        return self._runtime_plugins.plugin_toggle(name, enabled, chat_id=chat_id)

    def plugin_rollback(self, steps: int = 1) -> ChatResult:
        return self._runtime_plugins.plugin_rollback(steps)

    def plugin_sandbox(self, module: str, plugin: dict[str, Any] | None = None) -> dict[str, Any]:
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
        self._lifecycle.start()

    def _send_restart_notices(self) -> None:
        """Notify recently active chats that the service has restarted."""
        self._lifecycle.send_restart_notices()

    def _create_direct_notifier(self) -> Any:
        """Create a notifier that bypasses the outbox if possible."""
        return self._lifecycle.create_direct_notifier()

    def shutdown(self, drain_timeout: float = 120.0) -> None:
        """Drain active turns, notify plugins, and stop background workers."""
        self._lifecycle.shutdown(drain_timeout)

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

    # ---------------------------------------------------------------- public API

    def status(self, chat_id: str) -> dict[str, Any]:
        """Return the harness-recorded status for a chat."""
        return self._actions.status(chat_id)

    def list_sessions(self, chat_id: str) -> dict[str, Any]:
        """Return all non-pruned sessions for a chat, with the active one marked."""
        return self._actions.list_sessions(chat_id)

    def memory(self, chat_id: str) -> str:
        """Return the per-chat memory content."""
        return self._actions.memory(chat_id)

    def summarize(self, chat_id: str) -> ChatResult:
        """Trigger a manual summarization for a chat."""
        return self._actions.summarize(chat_id)

    def recall(
        self,
        chat_id: str,
        query: str,
        tags: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Recall relevant memories for a query."""
        return self._actions.recall(chat_id, query=query, tags=tags, max_tokens=max_tokens)

    def retain(
        self,
        chat_id: str,
        content: str,
        tags: list[str] | None = None,
        context: str | None = None,
    ) -> ChatResult:
        """Retain an observation in the chat's memory."""
        return self._actions.retain(chat_id, content=content, tags=tags, context=context)

    def promote(self, chat_id: str, fact: str) -> ChatResult:
        """Promote a fact to the persona's memory."""
        return self._actions.promote(chat_id, fact)

    def record_system_note(self, chat_id: str, text: str) -> None:
        """Append a system note to the chat's transcript."""
        try:
            self._memory_manager(chat_id).append_mesh_note(text)
        except Exception as exc:
            logger.warning(
                "Failed to record system note for %s",
                chat_id,
                exc_info=exc,
            )

    def list_models(self) -> list[str]:
        """Return the list of models the ACP server accepts."""
        return self._actions.list_models()

    def register_ingress_handler(self, protocol: str, handler: IngressHandler) -> None:
        """Register a protocol-specific inbound HTTP handler."""
        self._ingress.register_ingress_handler(protocol, handler)

    async def handle_ingress(self, protocol: str, request: Any) -> Any:
        """Dispatch an inbound HTTP request to the registered handler."""
        return await self._ingress.handle_ingress(protocol, request)

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
        return self._ingress.process(
            chat_id,
            user_message,
            model=model,
            reply_to=reply_to,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
            notify=notify,
        )

    def dispatch(
        self,
        chat_id: str,
        context: str | None = None,
    ) -> ChatResult:
        return self._ingress.dispatch(chat_id, context=context)

    def continue_turn(self, dispatch_id: str, result: str) -> ChatResult:
        return self._ingress.continue_turn(dispatch_id, result)

    def wake(
        self,
        chat_id: str,
        event_id: str | None = None,
        reason: str | None = None,
        silent: bool | None = None,
    ) -> ChatResult:
        return self._ingress.wake(chat_id, event_id=event_id, reason=reason, silent=silent)

    def _enqueue_plan_task_wake(self, plan: Plan, task: Task) -> None:
        """Enqueue a non-silent wake that reports one task's completion or failure."""
        return self._planning._enqueue_plan_task_wake(plan, task)

    def _maybe_enqueue_plan_conclusion(self, plan: Plan) -> None:
        """Enqueue a single non-silent wake with the plan's final conclusion."""
        return self._planning._maybe_enqueue_plan_conclusion(plan)

    def _build_plan_task_update_message(self, payload: dict[str, Any]) -> str:
        """Format a user message for a single task status update."""
        return self._planning._build_plan_task_update_message(payload)

    def _build_plan_completed_message(self, payload: dict[str, Any]) -> str:
        """Format a user message asking the model to deliver the final conclusion."""
        return self._planning._build_plan_completed_message(payload)

    def _build_dispatch_continuation(self, dispatch: Dispatch) -> str:
        """Build a continuation anchor for a completed background dispatch."""
        return self._planning._build_dispatch_continuation(dispatch)

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
        """Persist a terminal mesh message (e.g. a DSN) without running a turn."""
        return self._actions.record_mesh_message(chat_id, display_text, mesh_payload)

    def update_mesh_chat_map(
        self,
        chat_map: dict[str, str] | None = None,
        chat_mapping: Literal["per_sender", "single", "session"] | None = None,
        fallback_chat_id: str | None = None,
    ) -> dict:
        """Update the live mesh chat mapping and persist runtime overrides."""
        return self._config_manager.update_mesh_chat_map(
            chat_map=chat_map,
            chat_mapping=chat_mapping,
            fallback_chat_id=fallback_chat_id,
        )

    def graceful_service_restart(
        self,
        chat_id: str,
        service: str | None = None,
        reason: str = "",
    ) -> ChatResult:
        """Public API for a graceful service restart (HTTP/Telegram/MCP)."""
        return self._actions.graceful_service_restart(chat_id, service=service, reason=reason)

    def switch_model(self, chat_id: str, model: str, *, in_place: bool = False) -> ChatResult:
        """Switch the model for a chat."""
        return self._actions.switch_model(chat_id, model, in_place=in_place)

    def new_session(self, chat_id: str, model: str | None = None) -> ChatResult:
        """Start a fresh ACP session for a chat."""
        return self._actions.new_session(chat_id, model=model)

    def resume_session(self, chat_id: str, session_number: int) -> ChatResult:
        """Resume a previous session for a chat."""
        return self._actions.resume_session(chat_id, session_number)

    def branch_session(self, chat_id: str, session_number: int) -> ChatResult:
        """Branch a previous session for a chat."""
        return self._actions.branch_session(chat_id, session_number)

    # ---------------------------------------------------------------- plans / tasks

    def plan_create(
        self,
        name: str,
        description: str = "",
        chat_id: str | None = None,
        tasks: list[Task] | None = None,
    ) -> Plan:
        """Create a new plan."""
        return self._actions.plan_create(
            name, description=description, chat_id=chat_id, tasks=tasks
        )

    def plan_task_start(self, plan_id: str, task_id: str | None = None) -> Task:
        """Start a ready task in a plan."""
        return self._actions.plan_task_start(plan_id, task_id)

    def plan_task_done(
        self,
        plan_id: str,
        task_id: str,
        result: str = "",
        log: str = "",
    ) -> Task:
        """Manually mark a task as done and emit the completion event."""
        return self._actions.plan_task_done(plan_id, task_id, result=result, log=log)

    @locked
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
    def _human_duration(seconds: float) -> str:
        return human_duration(seconds)

    def _notify_subagent_timeout(
        self,
        task: Task,
        chat_id: str,
        dispatch_id: str,
        dispatch: Dispatch,
        summary: str,
        is_timeout: bool,
    ) -> None:
        self._subagent._notify_subagent_timeout(
            task, chat_id, dispatch_id, dispatch, summary, is_timeout
        )

    def subagent_status(self, chat_id: str) -> dict[str, Any]:
        return self._subagent.subagent_status(chat_id)

    def plan_list(self) -> list[Plan]:
        """List all plans."""
        return self.plan_manager.list_plans()

    def plan_get(self, plan_id: str) -> Plan | None:
        """Return one plan by id."""
        return self.plan_manager.get_plan(plan_id)

    # ---------------------------------------------------------------- pruning

    def prune_chat(self, chat_id: str) -> None:
        """Delete archived sessions older than the prune window."""
        self._chat_store.prune_chat(chat_id)

    def prune_and_compact(self, chat_id: str) -> None:
        self._chat_store.prune_and_compact(chat_id)

    def prune_all(self) -> None:
        self._chat_store.prune_all()
