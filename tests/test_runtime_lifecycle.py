"""Tests for AgentRuntime daemon lifecycle."""

from __future__ import annotations

import time
from pathlib import Path

from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    PersonaConfig,
    PlanConfig,
    Secrets,
    TimerConfig,
)
from diploid_agent.models import WakeEvent
from diploid_agent.plan.models import Task
from diploid_agent.runtime.agent_runtime import AgentRuntime


def _fixture_root() -> Path:
    return Path(__file__).parent / "fixtures" / "test-pilot"


def _make_config(tmp_path: Path) -> Config:
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=_fixture_root(),
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            plan=PlanConfig(root=tmp_path / "plans"),
            memory={"backend": "file"},  # type: ignore[arg-type]
            timer=TimerConfig(enabled=True, interval_seconds=0.1),
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


def test_runtime_start_is_idempotent(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.start()
    assert runtime.timer_service.running is True
    assert runtime.event_bus.running is True

    runtime.start()
    assert runtime.timer_service.running is True

    runtime.shutdown()


def test_runtime_shutdown_stops_services(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.start()
    assert runtime.timer_service.running is True
    runtime.shutdown()
    assert runtime.timer_service.running is False
    assert runtime.event_bus.running is False


def test_runtime_get_status_reports_state(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.start()
    try:
        status = runtime.get_status()
        assert status.instance_id == runtime.instance_id
        assert status.uptime_seconds >= 0.0
        assert status.event_bus_running is True
        assert status.timer_running is True
        assert status.plan_count == 0
        assert status.active_chat_count == 0
    finally:
        runtime.shutdown()


class FakeEngine:
    def prompt(self, *a, **k):
        from diploid_agent.engine import TurnResult

        return TurnResult(reply="woken", session_id="s1")

    def list_models(self):
        return ["m1"]

    def restart(self):
        pass

    def is_stale_session_error(self, exc):
        return False

    def close(self):
        pass


def test_timer_event_handler_wakes_chat(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()

    event = WakeEvent(
        id="",
        chat_id="chat-1",
        reason="timer",
        priority=1,
        scheduled_at=time.time() - 1,
        payload={},
        silent=True,
        created_at=time.time(),
        ready=True,
    )
    enqueued = runtime.wake_queue.enqueue(event)

    runtime.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if runtime.wake_queue.get(enqueued.id) is None:
                break
            time.sleep(0.05)
    finally:
        runtime.shutdown()

    assert runtime.wake_queue.get(enqueued.id) is None


def test_task_completed_handler_marks_plan_done(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.start()
    try:
        plan = runtime.plan_create("lifecycle-plan", tasks=[Task(name="t", command="")])
        task = plan.tasks[0]

        from diploid_agent.runtime.event_bus import Event

        runtime.event_bus.post(
            Event(
                type="task.completed",
                payload={
                    "plan_id": plan.id,
                    "task_id": task.id,
                    "result": "done-via-event",
                    "log": "",
                },
            )
        )

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            updated = runtime.plan_manager.get_task(plan.id, task.id)
            if updated is not None and updated.result == "done-via-event":
                break
            time.sleep(0.05)

        updated = runtime.plan_manager.get_task(plan.id, task.id)
        assert updated is not None
        assert updated.result == "done-via-event"
    finally:
        runtime.shutdown()
