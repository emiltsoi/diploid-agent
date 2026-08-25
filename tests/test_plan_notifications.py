from __future__ import annotations

import time
from pathlib import Path

import pytest

from acp_fleet_harness.config import (
    Config,
    DevinConfig,
    HarnessConfig,
    PersonaConfig,
    PlanConfig,
    Secrets,
    TimerConfig,
)
from acp_fleet_harness.models import WakeEvent
from acp_fleet_harness.plan.models import Task
from acp_fleet_harness.runtime.agent_runtime import AgentRuntime
from acp_fleet_harness.runtime.event_bus import Event


def _fixture_root() -> Path:
    return Path(__file__).parent / "fixtures" / "test-pilot"


def _make_config(tmp_path: Path) -> Config:
    return Config(
        devin=DevinConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=_fixture_root(),
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            plan=PlanConfig(root=tmp_path / "plans"),
            memory={"backend": "file"},  # type: ignore[arg-type]
            timer=TimerConfig(enabled=False, interval_seconds=0.1),
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


class FakeEngine:
    def prompt(self, *a, **k):
        from acp_fleet_harness.engine import TurnResult

        return TurnResult(reply="noted", session_id="s1")

    def list_models(self):
        return ["m1"]

    def restart(self):
        pass

    def is_stale_session_error(self, exc):
        return False

    def close(self):
        pass


class CaptureNotifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def send(self, chat_id: str, text: str, *, reply_to_message_id: int | None = None) -> None:
        self.messages.append((chat_id, text))

    def typing(self, chat_id: str) -> None:
        return None


@pytest.fixture
def runtime(tmp_path: Path):
    r = AgentRuntime(_make_config(tmp_path))
    r.engine = FakeEngine()
    r.notifier = CaptureNotifier()
    r.start()
    yield r
    r.shutdown()


def _wait_for_due_wakes(
    runtime: AgentRuntime,
    chat_id: str,
    expected: int = 1,
    reason: str | None = None,
    timeout: float = 2.0,
) -> None:
    """Wait until at least ``expected`` due wakes for ``chat_id`` are visible."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        wakes = [
            e
            for e in runtime.wake_queue.pending(chat_id=chat_id, now=time.time() + 1)
            if e.ready and (reason is None or e.reason == reason)
        ]
        if len(wakes) >= expected:
            return
        time.sleep(0.05)
    raise AssertionError(f"did not see {expected} due wakes for {chat_id}")


def _pop_wakes(runtime: AgentRuntime, chat_id: str, reason: str | None = None):
    due = runtime.wake_queue.pop_due(now=time.time() + 1)
    return [e for e in due if e.chat_id == chat_id and (reason is None or e.reason == reason)]


def test_task_completion_enqueues_plan_task_update_and_final_wake(runtime):
    plan = runtime.plan_create(
        "deploy",
        chat_id="chat-1",
        tasks=[Task(name="lint", command="echo ok", chat_id="chat-1")],
    )

    runtime.event_bus.post(
        Event(
            type="task.completed",
            payload={
                "plan_id": plan.id,
                "task_id": plan.tasks[0].id,
                "result": "ok",
                "log": "",
            },
        )
    )

    _wait_for_due_wakes(runtime, "chat-1", expected=2)
    wakes = _pop_wakes(runtime, "chat-1")
    assert len(wakes) == 2
    assert wakes[0].reason == "plan_task_update"
    assert wakes[1].reason == "plan_completed"


def test_task_failure_enqueues_plan_task_update_and_final_wake(runtime):
    plan = runtime.plan_create(
        "deploy",
        chat_id="chat-1",
        tasks=[Task(name="lint", command="exit 1", chat_id="chat-1")],
    )

    runtime.event_bus.post(
        Event(
            type="task.failed",
            payload={
                "plan_id": plan.id,
                "task_id": plan.tasks[0].id,
                "log": "it broke",
            },
        )
    )

    _wait_for_due_wakes(runtime, "chat-1", expected=2)
    wakes = _pop_wakes(runtime, "chat-1")
    assert len(wakes) == 2
    assert wakes[0].reason == "plan_task_update"
    assert wakes[0].payload["task_status"] == "failed"
    assert wakes[1].reason == "plan_completed"


def test_plan_without_chat_id_skips_wakes(runtime):
    plan = runtime.plan_create(
        "deploy",
        tasks=[Task(name="lint", command="echo ok")],
    )

    runtime.event_bus.post(
        Event(
            type="task.completed",
            payload={
                "plan_id": plan.id,
                "task_id": plan.tasks[0].id,
                "result": "ok",
                "log": "",
            },
        )
    )

    time.sleep(0.2)
    assert runtime.wake_queue.due_count(now=time.time() + 1) == 0


def test_plan_task_update_message_format(runtime):
    payload = {
        "plan_name": "deploy",
        "plan_status": "active",
        "task_name": "lint",
        "task_status": "done",
        "task_detail": "all clean",
        "completed_count": 1,
        "total_count": 3,
    }
    msg = runtime._build_plan_task_update_message(payload)
    assert "[system wake: plan task update]" in msg
    assert "Plan: deploy" in msg
    assert "Task: lint" in msg
    assert "Task status: done" in msg
    assert "Detail: all clean" in msg
    assert "1/3" in msg


def test_plan_completed_message_format(runtime):
    payload = {
        "plan_name": "deploy",
        "plan_status": "completed",
        "tasks_summary": "- lint: done\n- test: done",
    }
    msg = runtime._build_plan_completed_message(payload)
    assert "[system wake: plan completed]" in msg
    assert "Plan: deploy" in msg
    assert "Final status: completed" in msg
    assert "lint: done" in msg
    assert "Please give a concise final conclusion" in msg


def test_plan_conclusion_not_duplicated(runtime):
    plan = runtime.plan_create(
        "deploy",
        chat_id="chat-1",
        tasks=[
            Task(name="a", command="echo a", chat_id="chat-1"),
            Task(name="b", command="echo b", chat_id="chat-1"),
        ],
    )

    for t in plan.tasks:
        runtime.event_bus.post(
            Event(
                type="task.completed",
                payload={
                    "plan_id": plan.id,
                    "task_id": t.id,
                    "result": "ok",
                    "log": "",
                },
            )
        )

    _wait_for_due_wakes(runtime, "chat-1", expected=1, reason="plan_completed")
    final_wakes = _pop_wakes(runtime, "chat-1", reason="plan_completed")
    assert len(final_wakes) == 1

    # A duplicate or late event should not create another conclusion wake.
    runtime.event_bus.post(
        Event(
            type="task.completed",
            payload={
                "plan_id": plan.id,
                "task_id": plan.tasks[0].id,
                "result": "ok",
                "log": "",
            },
        )
    )
    time.sleep(0.2)
    final_wakes = _pop_wakes(runtime, "chat-1", reason="plan_completed")
    assert len(final_wakes) == 0


def test_plan_wake_runs_and_notifies(runtime):
    payload = {
        "plan_name": "deploy",
        "plan_status": "active",
        "task_name": "lint",
        "task_status": "done",
        "task_detail": "all clean",
        "completed_count": 1,
        "total_count": 3,
    }
    event = runtime.wake_queue.enqueue(
        WakeEvent(
            id="",
            chat_id="chat-1",
            reason="plan_task_update",
            priority=1,
            scheduled_at=time.time(),
            payload=payload,
            silent=False,
            created_at=time.time(),
            ready=True,
        )
    )

    result = runtime.wake("chat-1", event_id=event.id)
    assert result.reply == "noted"
    assert len(runtime.notifier.messages) == 1
    assert runtime.notifier.messages[0][0] == "chat-1"


def test_task_engine_full_flow_enqueues_wakes(tmp_path: Path):
    config = _make_config(tmp_path)
    runtime = AgentRuntime(config)
    runtime.engine = FakeEngine()
    runtime.notifier = CaptureNotifier()
    runtime.start()
    try:
        plan = runtime.plan_create(
            "deploy",
            chat_id="chat-1",
            tasks=[Task(name="lint", command="echo ok", chat_id="chat-1")],
        )
        runtime.plan_task_start(plan.id)

        _wait_for_due_wakes(runtime, "chat-1", expected=2)
        wakes = _pop_wakes(runtime, "chat-1")
        assert len(wakes) == 2
        assert wakes[0].reason == "plan_task_update"
        assert wakes[1].reason == "plan_completed"

        # Run the actual status turn and the conclusion turn.
        for w in wakes:
            result = runtime.wake("chat-1", event_id=w.id)
            assert result.reply == "noted"

        assert len(runtime.notifier.messages) == 2
    finally:
        runtime.shutdown()
