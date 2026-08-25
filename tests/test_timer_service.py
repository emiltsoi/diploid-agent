"""Tests for the TimerService wake queue consumer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from devin_fleet_harness.config import (
    Config,
    DevinConfig,
    HarnessConfig,
    PersonaConfig,
    PlanConfig,
    Secrets,
    TimerConfig,
)
from devin_fleet_harness.models import WakeEvent
from devin_fleet_harness.runtime.agent_runtime import AgentRuntime
from devin_fleet_harness.runtime.event_bus import Event, EventBus
from devin_fleet_harness.runtime.timer_service import TimerService
from devin_fleet_harness.runtime.wake_queue import WakeQueue


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
            timer=TimerConfig(enabled=True, interval_seconds=0.05, lease_seconds=1.0),
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


@dataclass
class FakeRuntime:
    wake_queue: WakeQueue
    event_bus: EventBus


def test_timer_service_posts_timer_fired_event(tmp_path: Path) -> None:
    bus = EventBus()
    bus.start()
    wake_path = tmp_path / "wake.jsonl"
    runtime = FakeRuntime(wake_queue=WakeQueue(wake_path), event_bus=bus)
    config = TimerConfig(enabled=True, interval_seconds=0.05, lease_seconds=60.0)
    service = TimerService(runtime, config)

    captured: list[Event] = []
    bus.subscribe(lambda event: captured.append(event))

    event = WakeEvent(
        id="w1",
        chat_id="chat-1",
        reason="timer",
        priority=1,
        scheduled_at=time.time() - 1,
        created_at=time.time(),
        ready=True,
    )
    runtime.wake_queue.enqueue(event)

    service.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if captured:
                break
            time.sleep(0.05)
    finally:
        service.stop()
        bus.stop()

    assert len(captured) == 1
    assert captured[0].type == "timer.fired"
    assert captured[0].payload["event_id"] == "w1"
    assert captured[0].payload["chat_id"] == "chat-1"


class FakeEngine:
    def prompt(self, *a, **k):
        from devin_fleet_harness.engine import TurnResult

        return TurnResult(reply="woken", session_id="s1")

    def list_models(self):
        return ["m1"]

    def restart(self):
        pass

    def is_stale_session_error(self, exc):
        return False

    def close(self):
        pass


def test_timer_service_wakes_due_event(tmp_path: Path) -> None:
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
