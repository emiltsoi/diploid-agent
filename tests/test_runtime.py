"""Tests for the AgentRuntime / TurnController split.

These tests verify that the runtime package exposes the same behavior as the
legacy ConversationHarness by delegating turn logic to TurnController while
keeping the service container and non-turn API on AgentRuntime.
"""

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    NotificationsConfig,
    PersonaConfig,
    PlanConfig,
    Secrets,
)
from diploid_agent.dispatch import DispatchStatus
from diploid_agent.engine.base import AgentEngine, TurnRequest, TurnResult
from diploid_agent.engine.fake import FakeAgentEngine
from diploid_agent.harness import ConversationHarness
from diploid_agent.models import WakeEvent
from diploid_agent.notifier import NoopNotifier
from diploid_agent.plan.models import Task, TaskStatus, TaskType
from diploid_agent.runtime import AgentRuntime, TurnController
from diploid_agent.transport.base import RuntimeAPI


def _make_config(tmp_path: Path) -> Config:
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=fixture_root,
            fleet_root=tmp_path / "fleet",
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            plan=PlanConfig(root=tmp_path / "plans"),
            memory={"backend": "file"},  # type: ignore[arg-type]
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


def _make_config_with_outbox(tmp_path: Path) -> Config:
    config = _make_config(tmp_path)
    config.harness.notifications = NotificationsConfig(enabled=True, outbox_delivery=True)
    return config


def test_agent_runtime_implements_runtime_api(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    assert isinstance(runtime, RuntimeAPI)


def test_subagent_start_creates_dispatch_and_plan(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.start()
    fake = FakeAgentEngine()
    fake.replies = ["parent reply"]
    runtime.engine = fake
    runtime.task_engine.engine = fake

    # Create an active session.
    result = runtime.process("chat-1", "hello")
    assert result.reply == "parent reply"

    fake.replies = ["subagent result"]
    result = runtime.subagent_start("chat-1", "do the thing")
    assert "subagent started" in result.reply.lower()
    assert result.dispatch_id is not None

    # The dispatch and wake should exist.
    dispatch = runtime.dispatch_store.get(result.dispatch_id)
    assert dispatch is not None
    assert dispatch.status == DispatchStatus.PENDING
    wake = runtime.wake_queue.get(f"wake-{result.dispatch_id}")
    assert wake is not None
    assert wake.reason == "dispatch"
    assert not wake.ready

    # Wait for the task to complete and the wake to become ready.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        wake = runtime.wake_queue.get(f"wake-{result.dispatch_id}")
        if wake and wake.ready:
            break
        time.sleep(0.05)
    assert wake.ready

    # Simulate the TimerService waking the chat.
    result = runtime.wake("chat-1", event_id=wake.id)
    assert result.reply == "ok"
    assert result.turn_number is not None

    dispatch = runtime.dispatch_store.get(result.dispatch_id or dispatch.id)
    assert dispatch.status == DispatchStatus.COMPLETED
    assert "subagent result" in (dispatch.result or "")

    runtime.shutdown()


def test_subagent_status_reports_running_and_completed(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.start()
    fake = FakeAgentEngine()
    fake.replies = ["parent reply"]
    runtime.engine = fake
    runtime.task_engine.engine = fake

    # Create an active session.
    result = runtime.process("chat-1", "hello")
    assert result.reply == "parent reply"

    fake.replies = ["# Summary\n\nsubagent result"]
    result = runtime.subagent_start("chat-1", "do the thing")
    assert "subagent started" in result.reply.lower()
    assert result.dispatch_id is not None
    dispatch_id = result.dispatch_id

    # The status should show a running subagent.
    status = runtime.subagent_status("chat-1")
    assert status["chat_id"] == "chat-1"
    assert len(status["subagents"]) == 1
    assert status["subagents"][0]["status"] == "running"
    assert status["subagents"][0]["dispatch_id"] == dispatch_id
    assert status["subagents"][0]["prompt_snippet"] == "do the thing"
    assert not status["subagents"][0]["continued"]

    # Wait for the task to complete and the dispatch to record summary.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        dispatch = runtime.dispatch_store.get(dispatch_id)
        if dispatch and dispatch.finished_at is not None:
            break
        time.sleep(0.05)
    assert dispatch is not None
    assert dispatch.finished_at is not None
    assert dispatch.summary == "Summary"

    status = runtime.subagent_status("chat-1")
    assert status["subagents"][0]["status"] == "completed"
    assert status["subagents"][0]["summary"] == "Summary"
    assert not status["subagents"][0]["continued"]

    # Simulate the TimerService waking the chat and completing the dispatch.
    wake = runtime.wake_queue.get(f"wake-{dispatch_id}")
    assert wake is not None
    result = runtime.wake("chat-1", event_id=wake.id)
    assert result.turn_number is not None

    status = runtime.subagent_status("chat-1")
    assert status["subagents"][0]["status"] == "completed"
    assert status["subagents"][0]["continued"]

    # Chat status should include background tasks.
    chat_status = runtime.status("chat-1")
    assert "background_tasks" in chat_status
    assert chat_status["background_tasks"]["subagents"][0]["dispatch_id"] == dispatch_id

    runtime.shutdown()


def test_agent_runtime_has_engine_and_turn_controller(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    assert runtime.engine is not None
    assert isinstance(runtime.turn_controller, TurnController)


def test_agent_runtime_client_property_aliased_engine(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    assert runtime.client is runtime.engine
    fake = object()
    runtime.client = fake  # type: ignore[assignment]
    assert runtime.engine is fake


def test_agent_runtime_non_turn_public_api(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    for name in (
        "mcp_list",
        "mcp_enable",
        "mcp_disable",
        "skill_list",
        "skill_enable",
        "skill_disable",
        "skill_create",
        "get_metrics",
        "list_models",
        "get_model",
        "status",
        "subagent_status",
        "list_sessions",
        "memory",
        "summarize",
        "recall",
        "retain",
        "promote",
    ):
        assert hasattr(runtime, name)
        assert callable(getattr(runtime, name))


def test_turn_controller_delegates_to_runtime_for_services(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    tc = runtime.turn_controller
    assert tc.runtime is runtime
    assert tc.runtime.config is runtime.config
    assert tc.runtime.engine is runtime.engine
    assert tc.runtime._lock is runtime._lock


def test_runtime_status_plan_active(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    status = runtime.get_status()
    assert status.plan_active is False
    assert status.active_plans == []


def test_conversation_harness_delegates_to_runtime(tmp_path: Path) -> None:
    harness = ConversationHarness(_make_config(tmp_path))
    assert isinstance(harness.runtime, AgentRuntime)
    assert isinstance(harness.turn_controller, TurnController)

    # RuntimeAPI turn methods should be present and delegate to the controller.
    for name in (
        "process",
        "continue_turn",
        "stop",
        "turn_status",
        "dispatch",
        "switch_model",
        "new_session",
        "resume_session",
        "branch_session",
    ):
        assert hasattr(harness, name)
        assert getattr(harness, name).__func__ is getattr(harness.runtime, name).__func__

    # Private helpers that live on the runtime are accessible via __getattr__.
    for name in (
        "is_continuation_message",
        "_build_first_prompt",
        "_memory_manager",
        "_active_record",
        "_chat_dir",
        "_telegram_message_registry_path",
    ):
        assert hasattr(harness, name)


class _ChunkingEngine(AgentEngine):
    """Fake engine that emits a chunk from a background thread.

    If the TurnController holds the runtime RLock while the engine prompt
    runs, the on_chunk callback will deadlock. This only completes when the
    lock is correctly released during the engine call.
    """

    def prompt(
        self,
        request: TurnRequest,
        *,
        session_id: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> TurnResult:
        def _emit() -> None:
            time.sleep(0.05)
            if on_chunk is not None:
                on_chunk("chunk")

        thread = threading.Thread(target=_emit)
        thread.start()
        thread.join()
        return TurnResult(reply="ok", session_id="s-1")

    def cancel(self, session_id: str) -> None:
        pass

    def list_models(self) -> list[str]:
        return ["swe-1-7"]

    def session_alive(self, session_id: str) -> bool:
        return True

    def close(self) -> None:
        pass

    def restart(self, reason: str | None = None, chat_id: str | None = None) -> None:
        pass

    def restart_transport(self, reason: str | None = None, chat_id: str | None = None) -> None:
        pass

    def is_stale_session_error(self, exc: BaseException) -> bool:
        return False

    def model_context_window(self, model: str) -> int | None:
        return None


def test_process_releases_runtime_lock_for_streaming_chunks(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = _ChunkingEngine()

    start = time.perf_counter()
    result = runtime.process("chat-1", "hello")
    elapsed = time.perf_counter() - start

    assert result.reply == "ok"
    assert elapsed < 1.0


def test_process_queues_message_when_chat_is_busy(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = _ChunkingEngine()

    runtime.instance_manager.acquire("chat-1")
    results: list[Any] = []

    def _process() -> None:
        results.append(runtime.process("chat-1", "hello"))

    worker = threading.Thread(target=_process)
    worker.start()
    time.sleep(0.05)
    runtime.instance_manager.release("chat-1")
    worker.join(timeout=5.0)

    assert results
    result = results[0]
    assert result.reply == "I'll get back to you in a moment."
    assert result.notice == "This chat is busy; your message was queued."

    pending = runtime.wake_queue.pending(chat_id="chat-1")
    assert len(pending) == 1
    assert pending[0].reason == "user_request"
    assert pending[0].priority == 10
    assert pending[0].payload["user_message"] == "hello"
    assert pending[0].payload.get("retry_after") == 2.0


def test_agent_runtime_restarts_transport(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    called: list[bool] = []

    class _RestartEngine(_ChunkingEngine):
        def restart(self, reason: str | None = None, chat_id: str | None = None) -> None:
            called.append(True)

    runtime.engine = _RestartEngine()

    result = runtime.restart("chat-1")
    assert called
    assert "restarted" in result.reply.lower()


def test_graceful_service_restart_schedules_systemd_run(tmp_path: Path, monkeypatch) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    popen_calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kwargs: Any) -> Any:
        popen_calls.append(cmd)

        class _Fake:
            pass

        return _Fake()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    result = runtime.graceful_service_restart("chat-1", "vesper.service", reason="test")
    assert "restarting" in result.reply.lower()
    assert len(popen_calls) == 1
    assert popen_calls[0][0] == "systemd-run"
    assert "vesper.service" in popen_calls[0]


def test_auto_continue_suppression(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    assert runtime.is_auto_continue_suppressed("chat-1") is False
    runtime.suppress_auto_continue("chat-1", seconds=10)
    assert runtime.is_auto_continue_suppressed("chat-1") is True
    runtime.suppress_auto_continue(seconds=10)
    assert runtime.is_auto_continue_suppressed("chat-2") is True


def _enqueue_dispatch_wake(runtime: AgentRuntime, dispatch_id: str) -> None:
    wake_id = f"wake-{dispatch_id}"
    runtime.wake_queue.enqueue(
        WakeEvent(
            id=wake_id,
            chat_id="chat-1",
            reason="dispatch",
            priority=1,
            scheduled_at=time.time(),
            created_at=time.time(),
            silent=False,
            payload={"dispatch_id": dispatch_id},
            ready=False,
        )
    )


def test_complete_subagent_task_persists_full_result(tmp_path: Path) -> None:
    """_complete_subagent_task writes the full result and leaves the dispatch ready."""
    runtime = AgentRuntime(_make_config(tmp_path))
    try:
        dispatch = runtime.dispatch_store.add("chat-1", "session-1", context="test")
        _enqueue_dispatch_wake(runtime, dispatch.id)

        task = Task(
            name="subagent",
            type=TaskType.SUBAGENT,
            chat_id="chat-1",
            dispatch_id=dispatch.id,
            status=TaskStatus.DONE,
            result="# Summary\n\nfull subagent result",
            started_at=time.time() - 10,
            completed_at=time.time(),
        )
        runtime._complete_subagent_task(task)

        dispatch = runtime.dispatch_store.get(dispatch.id)
        assert dispatch is not None
        assert dispatch.status == DispatchStatus.PENDING
        assert dispatch.result == "# Summary\n\nfull subagent result"
        assert dispatch.summary == "Summary"
        assert dispatch.full_result_path is not None
        result_file = Path(dispatch.full_result_path)
        assert result_file.exists()
        assert result_file.read_text() == "# Summary\n\nfull subagent result"

        wake = runtime.wake_queue.get(f"wake-{dispatch.id}")
        assert wake is not None
        assert wake.ready
    finally:
        runtime.shutdown()


def test_complete_subagent_task_timeout_notifies_and_marks_status(tmp_path: Path) -> None:
    """A timed-out subagent is marked TIMEOUT, persisted, and the user is notified."""
    runtime = AgentRuntime(_make_config(tmp_path))

    class _FakeNotifier(NoopNotifier):
        def __init__(self) -> None:
            self.sent: list[tuple[str, str]] = []

        def send(self, chat_id: str, text: str, *, reply_to_message_id: int | None = None) -> None:
            self.sent.append((chat_id, text))

    fake = _FakeNotifier()
    runtime.notifier = fake  # type: ignore[assignment]

    try:
        dispatch = runtime.dispatch_store.add("chat-1", "session-1", context="timeout test")
        _enqueue_dispatch_wake(runtime, dispatch.id)

        task = Task(
            name="subagent",
            type=TaskType.SUBAGENT,
            chat_id="chat-1",
            dispatch_id=dispatch.id,
            status=TaskStatus.DONE,
            result="partial result",
            log="Subagent stopped early: timed out",
            stop_reason="timeout",
            timed_out=True,
            partial=True,
            started_at=time.time() - 60,
            completed_at=time.time(),
        )
        runtime._complete_subagent_task(task)

        dispatch = runtime.dispatch_store.get(dispatch.id)
        assert dispatch is not None
        assert dispatch.status == DispatchStatus.TIMEOUT
        assert dispatch.stop_reason == "timeout"
        assert dispatch.timed_out is True
        assert dispatch.partial is True
        assert dispatch.full_result_path is not None
        assert "partial result" in Path(dispatch.full_result_path).read_text()

        assert len(fake.sent) == 1
        assert fake.sent[0][0] == "chat-1"
        assert "timed out" in fake.sent[0][1]
        assert "Partial summary:" in fake.sent[0][1]

        wake = runtime.wake_queue.get(f"wake-{dispatch.id}")
        assert wake is not None
        assert wake.ready
        assert runtime._subagent_status_name(task, dispatch) == "timeout"
    finally:
        runtime.shutdown()


def test_outbox_pop_returns_enqueued_result(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config_with_outbox(tmp_path))
    runtime.start()
    fake = FakeAgentEngine()
    fake.replies = ["parent reply"]
    runtime.engine = fake
    runtime.task_engine.engine = fake

    result = runtime.process("chat-1", "hello")
    assert result.reply == "parent reply"

    popped = runtime.outbox_pop("chat-1", wait=2.0)
    assert popped is not None
    assert popped.reply == "parent reply"
    assert popped.turn_number is not None

    runtime.shutdown()


def test_subagent_timeout_with_outbox_enqueues_notification(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config_with_outbox(tmp_path))

    try:
        dispatch = runtime.dispatch_store.add("chat-1", "session-1", context="timeout test")
        _enqueue_dispatch_wake(runtime, dispatch.id)

        task = Task(
            name="subagent",
            type=TaskType.SUBAGENT,
            chat_id="chat-1",
            dispatch_id=dispatch.id,
            status=TaskStatus.DONE,
            result="partial result",
            log="Subagent stopped early: timed out",
            stop_reason="timeout",
            timed_out=True,
            partial=True,
            started_at=time.time() - 60,
            completed_at=time.time(),
        )
        runtime._complete_subagent_task(task)

        notification = runtime.outbox_pop("chat-1", wait=2.0)
        assert notification is not None
        assert "timed out" in notification.reply
        assert "Partial summary" in notification.reply
        assert notification.dispatch_id == dispatch.id

        wake = runtime.wake_queue.get(f"wake-{dispatch.id}")
        assert wake is not None
        assert wake.ready
    finally:
        runtime.shutdown()


def test_dispatch_failed_wake_continues_and_updates_status(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config_with_outbox(tmp_path))
    runtime.start()
    fake = FakeAgentEngine()
    fake.replies = ["parent reply", "failed subagent note"]
    runtime.engine = fake
    runtime.task_engine.engine = fake

    runtime.process("chat-1", "hello")

    active = runtime._active_record("chat-1")
    assert active is not None
    dispatch = runtime.dispatch_store.add("chat-1", active.session_id)
    _enqueue_dispatch_wake(runtime, dispatch.id)

    task = Task(
        name="subagent",
        type=TaskType.SUBAGENT,
        chat_id="chat-1",
        dispatch_id=dispatch.id,
        status=TaskStatus.FAILED,
        result="",
        log="Subagent failed",
        partial=False,
        started_at=time.time() - 30,
        completed_at=time.time(),
    )
    runtime._complete_subagent_task(task)

    dispatch = runtime.dispatch_store.get(dispatch.id)
    assert dispatch is not None
    assert dispatch.status == DispatchStatus.FAILED

    wake = runtime.wake_queue.get(f"wake-{dispatch.id}")
    assert wake is not None
    assert wake.ready

    result = runtime.wake("chat-1", event_id=wake.id)
    assert result.turn_number is not None
    assert "failed" in result.reply.lower() or "subagent" in result.reply.lower()

    runtime.shutdown()
