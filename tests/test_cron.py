"""Tests for the declarative cron scheduler (Wave A)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from diploid_agent.config import (
    Config,
    CronCallSpec,
    CronConfig,
    CronJobSpec,
    CronScheduleSpec,
    DiploidConfig,
    HarnessConfig,
    PersonaConfig,
    PlanConfig,
    PluginConfig,
    Secrets,
    TaskConfig,
)
from diploid_agent.models import WakeEvent
from diploid_agent.plan.manager import PlanManager
from diploid_agent.plan.models import TaskStatus
from diploid_agent.runtime.cron_service import CronService
from diploid_agent.runtime.cron_state import CronStateStore
from diploid_agent.runtime.event_bus import Event, EventBus
from diploid_agent.runtime.wake_queue import WakeQueue
from diploid_agent.task.engine import TaskEngine
from diploid_agent.transport.http import create_app


def _fixture_root() -> Path:
    return Path(__file__).parent / "fixtures" / "test-pilot"


class FakeEngine:
    def prompt(self, *a, **k):
        from diploid_agent.engine import TurnResult

        return TurnResult(reply="phantom says ok", session_id="s1")

    def list_models(self):
        return ["m1"]

    def restart(self):
        pass

    def is_stale_session_error(self, exc):
        return False

    def close(self):
        pass


def _make_config(
    tmp_path: Path,
    *,
    cron_enabled: bool = True,
    persona_crons: dict | None = None,
    global_crons: dict | None = None,
    **cron_kwargs,
) -> Config:
    persona_root = tmp_path / "persona"
    persona_root.mkdir(parents=True, exist_ok=True)
    for name in ("SOUL.md", "AGENTS.md", "MEMORY.md"):
        (persona_root / name).write_text((_fixture_root() / name).read_text())
    if persona_crons is not None:
        (persona_root / "crons.yaml").write_text(yaml.safe_dump(persona_crons))
    global_file = tmp_path / "config" / "crons.yaml"
    if global_crons is not None:
        global_file.parent.mkdir(parents=True, exist_ok=True)
        global_file.write_text(yaml.safe_dump(global_crons))
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(name="test-pilot", profile_root=persona_root),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            plan=PlanConfig(root=tmp_path / "plans"),
            memory={"backend": "file"},  # type: ignore[arg-type]
            cron=CronConfig(
                global_file=global_file,
                **{"min_interval_seconds": 0.0, **cron_kwargs},
            ),
            plugins=[
                PluginConfig(
                    name="authorship",
                    enabled=True,
                    module="diploid_agent.plugins.authorship",
                    config={"cron_enabled": cron_enabled},
                )
            ],
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


def _make_service(config: Config, tmp_path: Path) -> CronService:
    plan_manager = PlanManager(tmp_path / "plans")
    bus = EventBus()
    engine = TaskEngine(
        plan_manager,
        bus,
        engine=FakeEngine(),
        config=config,
        task_config=TaskConfig(shell_timeout=5.0),
    )
    return CronService(
        config=config,
        plan_manager=plan_manager,
        task_engine=engine,
        event_bus=bus,
        sessions_root=config.harness.sessions_root,
        wake_queue=WakeQueue(tmp_path / "wake_queue.jsonl"),
    )


def _script_job(**overrides) -> dict:
    job = {
        "id": "tidy",
        "schedule": {"every_seconds": 1.0},
        "call": {"type": "script", "command": "echo hi"},
        "chat_id": "chat-1",
    }
    job.update(overrides)
    return job


# ---------------------------------------------------------------- validation


def test_schedule_requires_exactly_one_field() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        CronScheduleSpec()
    with pytest.raises(ValueError, match="exactly one"):
        CronScheduleSpec(cron="* * * * *", every_seconds=60)


def test_at_daily_format() -> None:
    assert CronScheduleSpec(at_daily="06:30").at_daily == "06:30"
    with pytest.raises(ValueError):
        CronScheduleSpec(at_daily="6:70")
    with pytest.raises(ValueError):
        CronScheduleSpec(at_daily="25:00")


def test_call_required_fields() -> None:
    with pytest.raises(ValueError, match="command"):
        CronCallSpec(type="script")
    with pytest.raises(ValueError, match="prompt"):
        CronCallSpec(type="llm")


def test_turn_delivery_accepted() -> None:
    spec = CronJobSpec(
        id="job",
        schedule={"every_seconds": 60},
        call={"type": "script", "command": "true"},
        delivery="turn",
    )
    assert spec.delivery == "turn"


def test_job_id_must_be_slug() -> None:
    with pytest.raises(ValueError):
        CronJobSpec(
            id="Bad ID!",
            schedule={"every_seconds": 60},
            call={"type": "script", "command": "true"},
        )


# ------------------------------------------------------------------ service


def test_due_script_job_materializes(tmp_path: Path) -> None:
    config = _make_config(tmp_path, persona_crons={"jobs": [_script_job()]})
    svc = _make_service(config, tmp_path)
    svc._tick()  # seeds next_due forward
    state = svc._state.get("tidy")
    assert state is not None and state.next_due_at and state.next_due_at > time.time()

    state.next_due_at = time.time() - 1
    svc._state.update(state)
    svc._tick()  # fires
    state = svc._state.get("tidy")
    assert state.running_task_id is not None
    plan = svc._plan_manager.get_plan(state.running_plan_id)
    assert plan is not None and plan.name == "__cron__"
    task = plan.tasks[-1]
    assert task.type.value == "shell" and task.command == "echo hi"


def test_min_interval_floor_drops_job(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        min_interval_seconds=300.0,
        persona_crons={"jobs": [_script_job(schedule={"every_seconds": 60})]},
    )
    svc = _make_service(config, tmp_path)
    assert "tidy" not in svc._jobs
    assert any("min_interval" in w for w in svc._warnings)


def test_fast_cron_expression_dropped(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        min_interval_seconds=300.0,
        persona_crons={"jobs": [_script_job(schedule={"cron": "* * * * *"})]},
    )
    svc = _make_service(config, tmp_path)
    assert "tidy" not in svc._jobs


def test_persona_file_gated_by_toggle(tmp_path: Path) -> None:
    config = _make_config(tmp_path, cron_enabled=False, persona_crons={"jobs": [_script_job()]})
    svc = _make_service(config, tmp_path)
    assert svc._jobs == {}
    assert any("cron_enabled" in w for w in svc._warnings)


def test_global_file_ungated(tmp_path: Path) -> None:
    config = _make_config(tmp_path, cron_enabled=False, global_crons={"jobs": [_script_job()]})
    svc = _make_service(config, tmp_path)
    assert "tidy" in svc._jobs


def test_per_persona_job_cap(tmp_path: Path) -> None:
    jobs = [_script_job(id=f"job-{i}", schedule={"every_seconds": 10 + i}) for i in range(5)]
    config = _make_config(tmp_path, max_jobs_per_persona=2, persona_crons={"jobs": jobs})
    svc = _make_service(config, tmp_path)
    assert len(svc._jobs) == 2
    assert any("cap" in w for w in svc._warnings)


def test_duplicate_id_conflict(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        global_crons={"jobs": [_script_job()]},
        persona_crons={"jobs": [_script_job()]},
    )
    svc = _make_service(config, tmp_path)
    assert len(svc._jobs) == 1
    assert any("conflict" in w for w in svc._warnings)


def test_parse_error_keeps_last_good(tmp_path: Path) -> None:
    config = _make_config(tmp_path, persona_crons={"jobs": [_script_job()]})
    svc = _make_service(config, tmp_path)
    assert "tidy" in svc._jobs
    crons_path = tmp_path / "persona" / "crons.yaml"
    crons_path.write_text("{{{{not yaml: [")
    # Force mtime change.
    import os

    os.utime(crons_path, (time.time() + 5, time.time() + 5))
    svc._tick()
    assert "tidy" in svc._jobs  # last-good kept
    assert any("parse error" in w for w in svc._warnings)


def test_overlap_skip_and_queue(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        persona_crons={
            "jobs": [
                _script_job(id="skipper", overlap="skip"),
                _script_job(id="queuer", overlap="queue"),
            ]
        },
    )
    svc = _make_service(config, tmp_path)
    svc._tick()
    for job_id in ("skipper", "queuer"):
        state = svc._state.get(job_id)
        state.next_due_at = time.time() - 1
        state.running_task_id = "task-still-running"
        state.running_plan_id = "plan-x"
        svc._state.update(state)
    # Block reconcile: make the fake task appear RUNNING.
    plan = svc._plan_manager.create_plan(name="p", chat_id="chat-1")
    from diploid_agent.plan.models import Task

    running = svc._plan_manager.add_task(plan.id, Task(name="r"))
    assert running is not None
    svc._plan_manager.start_task(plan.id, running.id)
    for job_id in ("skipper", "queuer"):
        state = svc._state.get(job_id)
        state.running_task_id = running.id
        state.running_plan_id = plan.id
        svc._state.update(state)
    svc._tick()
    assert svc._state.get("skipper").last_status == "skipped"
    assert svc._state.get("queuer").queued_due is True


def test_catchup_once_fires_once_on_boot(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        persona_crons={"jobs": [_script_job(catchup="once")]},
    )
    svc = _make_service(config, tmp_path)
    # Simulate downtime: persisted next_due in the past before first tick.
    state = svc._state.ensure("tidy")
    state.next_due_at = time.time() - 600
    svc._state.update(state)
    svc._tick()  # boot pass + due fire
    state = svc._state.get("tidy")
    assert state.running_task_id is not None or state.last_status == "catchup"


def test_catchup_skip_advances(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        persona_crons={"jobs": [_script_job(catchup="skip")]},
    )
    svc = _make_service(config, tmp_path)
    state = svc._state.ensure("tidy")
    state.next_due_at = time.time() - 600
    svc._state.update(state)
    svc._tick()
    state = svc._state.get("tidy")
    assert state.next_due_at > time.time()
    assert state.running_task_id is None


def test_consecutive_failures_auto_disable(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        persona_crons={"jobs": [_script_job(max_consecutive_failures=2)]},
    )
    svc = _make_service(config, tmp_path)
    state = svc._state.ensure("tidy")

    from diploid_agent.plan.models import Task

    for _ in range(2):
        plan = svc._plan_manager.create_plan(name="p", chat_id="chat-1")
        task = svc._plan_manager.add_task(plan.id, Task(name="cron:tidy"))
        svc._plan_manager.start_task(plan.id, task.id)
        svc._plan_manager.fail_task(plan.id, task.id, log="boom")
        state.running_task_id = task.id
        state.running_plan_id = plan.id
        svc._state.update(state)
        svc._event_bus.post(
            Event(
                type="task.failed",
                payload={"plan_id": plan.id, "task_id": task.id},
            )
        )
        svc._on_event(Event(type="task.failed", payload={"plan_id": plan.id, "task_id": task.id}))
        state = svc._state.get("tidy")

    assert state.disabled is True
    assert state.last_status == "disabled"
    last_file = tmp_path / "sessions" / "chat-1" / "cron" / "tidy.last"
    assert last_file.exists()
    assert json.loads(last_file.read_text())["status"] == "disabled"


def test_llm_job_composes_phantom(tmp_path: Path) -> None:
    job = _script_job(
        id="reflect",
        call={"type": "llm", "prompt": "Tidy the notes.", "model": "m-x"},
    )
    config = _make_config(tmp_path, persona_crons={"jobs": [job]})
    svc = _make_service(config, tmp_path)
    svc._tick()
    state = svc._state.get("reflect")
    state.next_due_at = time.time() - 1
    svc._state.update(state)
    svc._tick()
    plan = svc._plan_manager.get_plan(svc._state.get("reflect").running_plan_id)
    task = plan.tasks[-1]
    assert task.type.value == "acp"
    assert task.mcp_servers == []  # phantom: no mesh, no self-wake tools
    assert task.acp_model == "m-x"
    assert "# test-pilot — background job" in task.prompt
    assert "Tidy the notes." in task.prompt
    assert "SOUL" in task.prompt
    assert "Output contract" in task.prompt


def test_silent_delivery_writes_last_files(tmp_path: Path) -> None:
    config = _make_config(tmp_path, persona_crons={"jobs": [_script_job()]})
    svc = _make_service(config, tmp_path)
    svc._tick()
    state = svc._state.get("tidy")
    state.next_due_at = time.time() - 1
    svc._state.update(state)
    svc._tick()
    state = svc._state.get("tidy")
    task = svc._plan_manager.get_task(state.running_plan_id, state.running_task_id)
    # Wait for the worker to finish the (fast) echo command.
    deadline = time.time() + 5
    while time.time() < deadline:
        task = svc._plan_manager.get_task(state.running_plan_id, state.running_task_id)
        if task.status in (TaskStatus.DONE, TaskStatus.FAILED):
            break
        time.sleep(0.05)
    svc._on_event(
        Event(
            type="task.completed",
            payload={"plan_id": state.running_plan_id, "task_id": task.id},
        )
    )
    last_file = tmp_path / "sessions" / "chat-1" / "cron" / "tidy.last"
    assert last_file.exists()
    data = json.loads(last_file.read_text())
    assert data["status"] == "ok"
    assert data["delivery"] == "silent"  # digest slot filters on this field
    assert (tmp_path / "sessions" / "chat-1" / "cron" / "tidy.log").exists()


def test_get_cron_route(tmp_path: Path) -> None:
    config = _make_config(tmp_path, persona_crons={"jobs": [_script_job()]})
    from diploid_agent.runtime.agent_runtime import AgentRuntime

    runtime = AgentRuntime(config)
    with TestClient(create_app(config, runtime)) as client:
        resp = client.get("/cron")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert [j["id"] for j in body["jobs"]] == ["tidy"]
    assert body["jobs"][0]["call_type"] == "script"


def test_state_store_roundtrip(tmp_path: Path) -> None:
    store = CronStateStore(tmp_path / "cron_state.jsonl")
    state = store.ensure("job-a", source_file="x")
    state.next_due_at = 123.0
    store.update(state)
    store2 = CronStateStore(tmp_path / "cron_state.jsonl")
    assert store2.get("job-a").next_due_at == 123.0


# ------------------------------------------------------------- review fixes


def test_finalize_is_idempotent(tmp_path: Path) -> None:
    """A racing event-bus + reconcile finalize must not double-count."""
    config = _make_config(tmp_path, persona_crons={"jobs": [_script_job()]})
    svc = _make_service(config, tmp_path)
    plan = svc._plan_manager.create_plan(name="p", chat_id="chat-1")
    from diploid_agent.plan.models import Task

    task = svc._plan_manager.add_task(plan.id, Task(name="cron:tidy"))
    svc._plan_manager.start_task(plan.id, task.id)
    svc._plan_manager.fail_task(plan.id, task.id, log="boom")
    state = svc._state.ensure("tidy")
    state.running_task_id = task.id
    state.running_plan_id = plan.id
    svc._state.update(state)
    task = svc._plan_manager.get_task(plan.id, task.id)
    svc._finalize(state, task)
    assert svc._state.get("tidy").consecutive_failures == 1
    # Second finalize of the same task (the caller's stale state still shows
    # it running) must return early instead of counting the failure twice.
    svc._finalize(state, task)
    assert svc._state.get("tidy").consecutive_failures == 1


def test_start_failure_marks_task_failed(tmp_path: Path, monkeypatch) -> None:
    """A failed start must not leave an orphaned READY task in the plan."""
    config = _make_config(tmp_path, persona_crons={"jobs": [_script_job()]})
    svc = _make_service(config, tmp_path)
    svc._tick()
    state = svc._state.get("tidy")
    state.next_due_at = time.time() - 1
    svc._state.update(state)

    def _boom(plan_id: str, task_id: str):
        raise ValueError("nope")

    monkeypatch.setattr(svc._task_engine, "start_task", _boom)
    svc._tick()
    state = svc._state.get("tidy")
    assert state.last_status == "failed"
    assert state.running_task_id is None
    task = svc._plan_manager.get_task(svc._cron_plans["chat-1"], state.last_task_id)
    assert task.status == TaskStatus.FAILED


def test_cron_gap_samples_hidden_dense_gap() -> None:
    """The first upcoming gap is ~50min; a 10-min gap hides one fire later."""
    lt = time.localtime()
    base = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 10, 0, 0, 0, 0, -1))
    # Midday hours keep the sampled gaps DST-immune.
    assert CronService._cron_gap("5,55 12,13 * * *", base) == 600


def test_dense_later_cron_dropped(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        min_interval_seconds=1200.0,
        persona_crons={"jobs": [_script_job(schedule={"cron": "5,55 12,13 * * *"})]},
    )
    svc = _make_service(config, tmp_path)
    assert "tidy" not in svc._jobs
    assert any("min_interval" in w for w in svc._warnings)


def test_sparse_cron_survives_sampling(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        min_interval_seconds=1200.0,
        persona_crons={"jobs": [_script_job(schedule={"cron": "30 3 * * *"})]},
    )
    svc = _make_service(config, tmp_path)
    assert "tidy" in svc._jobs


def test_at_daily_lands_on_wall_clock(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        persona_crons={"jobs": [_script_job(schedule={"at_daily": "10:30"})]},
    )
    svc = _make_service(config, tmp_path)
    spec = svc._jobs["tidy"].spec
    base = time.time()
    nxt = svc._next_due(spec, base)
    lt = time.localtime(nxt)
    assert (lt.tm_hour, lt.tm_min) == (10, 30)
    assert base < nxt <= base + 90000  # a 25h DST day still lands "tomorrow"


def test_missing_chat_id_drops_job(tmp_path: Path) -> None:
    job = _script_job()
    job["chat_id"] = None
    config = _make_config(tmp_path, persona_crons={"jobs": [job]})
    config.harness.mesh.fallback_chat_id = ""  # no fallback configured
    svc = _make_service(config, tmp_path)
    assert "tidy" not in svc._jobs
    assert any("chat_id" in w for w in svc._warnings)


def test_reads_do_not_rewrite_state_file(tmp_path: Path) -> None:
    store = CronStateStore(tmp_path / "cron_state.jsonl")
    store.ensure("job-a")
    saved: list[int] = []
    orig_save = store._save

    def _spy() -> None:
        saved.append(1)
        orig_save()

    store._save = _spy  # type: ignore[method-assign]
    store.get("job-a")
    store.all()
    store.ensure("job-a")  # existing — pure read
    assert not saved
    store.ensure("job-b")  # new — writes once
    assert len(saved) == 1


def test_catchup_task_description_tagged(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        persona_crons={"jobs": [_script_job(catchup="once")]},
    )
    svc = _make_service(config, tmp_path)
    state = svc._state.ensure("tidy")
    state.next_due_at = time.time() - 600
    svc._state.update(state)
    svc._tick()
    state = svc._state.get("tidy")
    assert state.last_task_id is not None
    task = svc._plan_manager.get_task(state.running_plan_id or "", state.last_task_id)
    assert task is not None and "catchup" in task.description


# ------------------------------------------------------------------ Wave B


def _run_to_done(svc: CronService, job_id: str = "tidy"):
    """Force-fire a job and route its task completion through the cron handler."""
    svc._tick()
    state = svc._state.get(job_id)
    state.next_due_at = time.time() - 1
    svc._state.update(state)
    svc._tick()
    state = svc._state.get(job_id)
    assert state.running_task_id is not None
    deadline = time.time() + 5
    task = None
    while time.time() < deadline:
        task = svc._plan_manager.get_task(state.running_plan_id, state.running_task_id)
        if task.status in (TaskStatus.DONE, TaskStatus.FAILED):
            break
        time.sleep(0.05)
    assert task is not None and task.status == TaskStatus.DONE
    svc._on_event(
        Event(
            type="task.completed",
            payload={"plan_id": state.running_plan_id, "task_id": task.id},
        )
    )
    return task


def _last_file(tmp_path: Path, job_id: str = "tidy") -> dict:
    return json.loads((tmp_path / "sessions" / "chat-1" / "cron" / f"{job_id}.last").read_text())


def test_turn_delivery_enqueues_wake(tmp_path: Path) -> None:
    config = _make_config(tmp_path, persona_crons={"jobs": [_script_job(delivery="turn")]})
    svc = _make_service(config, tmp_path)
    _run_to_done(svc)
    events = [e for e in svc._wake_queue.pending(chat_id="chat-1") if e.reason == "cron:tidy"]
    assert len(events) == 1
    ev = events[0]
    assert ev.silent is False
    assert ev.payload["notify"] is True
    assert "[cron: tidy finished — ok]" in ev.payload["user_message"]
    data = _last_file(tmp_path)
    assert data["delivery_result"] == "turn"
    assert svc._state.get("tidy").turn_count == 1


def test_turn_delivery_pending_cap_degrades(tmp_path: Path) -> None:
    config = _make_config(tmp_path, persona_crons={"jobs": [_script_job(delivery="turn")]})
    svc = _make_service(config, tmp_path)
    max_pending = config.harness.timer.self_wake_max_pending
    for _ in range(max_pending):
        svc._wake_queue.enqueue(
            WakeEvent(
                id="",
                chat_id="chat-1",
                reason="self_wake",
                priority=1,
                scheduled_at=time.time(),
                payload={},
                created_at=time.time() - 3600,
                ready=True,
            )
        )
    _run_to_done(svc)
    events = [e for e in svc._wake_queue.pending(chat_id="chat-1") if e.reason == "cron:tidy"]
    assert not events
    assert _last_file(tmp_path)["delivery_result"].startswith("turn_suppressed")


def test_turn_delivery_min_interval_defers(tmp_path: Path) -> None:
    config = _make_config(tmp_path, persona_crons={"jobs": [_script_job(delivery="turn")]})
    svc = _make_service(config, tmp_path)
    svc._wake_queue.enqueue(
        WakeEvent(
            id="",
            chat_id="chat-1",
            reason="self_wake",
            priority=1,
            scheduled_at=time.time(),
            payload={},
            created_at=time.time(),
            ready=True,
        )
    )
    _run_to_done(svc)
    events = [e for e in svc._wake_queue.pending(chat_id="chat-1") if e.reason == "cron:tidy"]
    assert len(events) == 1
    min_interval = config.harness.timer.self_wake_min_interval_seconds
    assert events[0].scheduled_at > time.time() + min_interval - 30
    assert _last_file(tmp_path)["delivery_result"].startswith("turn_deferred")


def test_turn_delivery_defers_past_pending_fire(tmp_path: Path) -> None:
    """A deferred/imminent pool fire pushes the next delivery past *it* —
    spacing fires, not arms."""
    config = _make_config(tmp_path, persona_crons={"jobs": [_script_job(delivery="turn")]})
    svc = _make_service(config, tmp_path)
    other = svc._wake_queue.enqueue(
        WakeEvent(
            id="",
            chat_id="chat-1",
            reason="self_wake",
            priority=1,
            scheduled_at=time.time() + 120,
            payload={},
            created_at=time.time() - 3600,
            ready=True,
        )
    )
    _run_to_done(svc)
    events = [e for e in svc._wake_queue.pending(chat_id="chat-1") if e.reason == "cron:tidy"]
    assert len(events) == 1
    min_interval = config.harness.timer.self_wake_min_interval_seconds
    assert events[0].scheduled_at >= other.scheduled_at + min_interval - 5
    assert _last_file(tmp_path)["delivery_result"].startswith("turn_deferred")


def test_turn_delivery_far_future_arm_does_not_defer(tmp_path: Path) -> None:
    """An arm outside the collision window doesn't push a ready result out."""
    config = _make_config(tmp_path, persona_crons={"jobs": [_script_job(delivery="turn")]})
    svc = _make_service(config, tmp_path)
    svc._wake_queue.enqueue(
        WakeEvent(
            id="",
            chat_id="chat-1",
            reason="self_wake",
            priority=1,
            scheduled_at=time.time() + 7200,
            payload={},
            created_at=time.time() - 3600,
            ready=True,
        )
    )
    _run_to_done(svc)
    events = [e for e in svc._wake_queue.pending(chat_id="chat-1") if e.reason == "cron:tidy"]
    assert len(events) == 1
    assert events[0].scheduled_at <= time.time() + 5
    assert _last_file(tmp_path)["delivery_result"] == "turn"


def test_turn_delivery_daily_cap(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        persona_crons={"jobs": [_script_job(delivery="turn")]},
        turn_delivery_max_per_day=1,
    )
    svc = _make_service(config, tmp_path)
    _run_to_done(svc)
    _run_to_done(svc)
    events = [e for e in svc._wake_queue.pending(chat_id="chat-1") if e.reason == "cron:tidy"]
    assert len(events) == 1
    assert _last_file(tmp_path)["delivery_result"] == "turn_suppressed: daily cap reached"


def test_run_now_fires_and_keeps_schedule(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        persona_crons={"jobs": [_script_job(schedule={"every_seconds": 3600})]},
    )
    svc = _make_service(config, tmp_path)
    svc._tick()
    due_before = svc._state.get("tidy").next_due_at
    out = svc.run_now("tidy")
    assert out["status"] == "started"
    state = svc._state.get("tidy")
    assert state.running_task_id is not None
    assert state.next_due_at == due_before  # schedule not consumed
    with pytest.raises(RuntimeError, match="already running"):
        svc.run_now("tidy")
    with pytest.raises(KeyError):
        svc.run_now("ghost")


def test_manual_run_success_reenables(tmp_path: Path) -> None:
    config = _make_config(tmp_path, persona_crons={"jobs": [_script_job()]})
    svc = _make_service(config, tmp_path)
    state = svc._state.ensure("tidy")
    state.disabled = True
    state.consecutive_failures = 3
    svc._state.update(state)
    svc._tick()
    assert svc._state.get("tidy").running_task_id is None  # disabled: no fire
    assert svc.run_now("tidy")["status"] == "started"
    state = svc._state.get("tidy")
    deadline = time.time() + 5
    while time.time() < deadline:
        task = svc._plan_manager.get_task(state.running_plan_id, state.running_task_id)
        if task.status in (TaskStatus.DONE, TaskStatus.FAILED):
            break
        time.sleep(0.05)
    svc._on_event(
        Event(
            type="task.completed",
            payload={"plan_id": state.running_plan_id, "task_id": state.running_task_id},
        )
    )
    state = svc._state.get("tidy")
    assert state.disabled is False
    assert state.consecutive_failures == 0


def test_run_now_route(tmp_path: Path) -> None:
    job = _script_job(call={"type": "script", "command": "sleep 2"})
    config = _make_config(tmp_path, persona_crons={"jobs": [job]})
    from diploid_agent.runtime.agent_runtime import AgentRuntime

    runtime = AgentRuntime(config)
    with TestClient(create_app(config, runtime)) as client:
        assert client.post("/cron/ghost/run").status_code == 404
        resp = client.post("/cron/tidy/run")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "tidy" and body["status"] == "started"
        # The in-flight run is a conflict.
        assert client.post("/cron/tidy/run").status_code == 409


def test_hot_edit_schedule_reseeds_on_finalize(tmp_path: Path) -> None:
    crons = tmp_path / "persona" / "crons.yaml"
    job = _script_job(
        schedule={"every_seconds": 60},
        call={"type": "script", "command": "sleep 2"},
    )
    config = _make_config(tmp_path, persona_crons={"jobs": [job]})
    svc = _make_service(config, tmp_path)
    svc._tick()
    state = svc._state.get("tidy")
    state.next_due_at = time.time() - 1
    svc._state.update(state)
    svc._tick()
    state = svc._state.get("tidy")
    assert state.running_task_id is not None
    # Hot-edit the schedule while the run is in flight.
    crons.write_text(yaml.safe_dump({"jobs": [_script_job(schedule={"every_seconds": 3600})]}))
    svc._tick()  # reload: drift seen, in-flight slot kept
    state = svc._state.get("tidy")
    assert state.running_task_id is not None
    # Finish and finalize — the new schedule owns the next fire.
    deadline = time.time() + 5
    task = None
    while time.time() < deadline:
        task = svc._plan_manager.get_task(state.running_plan_id, state.running_task_id)
        if task.status in (TaskStatus.DONE, TaskStatus.FAILED):
            break
        time.sleep(0.05)
    svc._on_event(
        Event(
            type="task.completed",
            payload={"plan_id": state.running_plan_id, "task_id": task.id},
        )
    )
    state = svc._state.get("tidy")
    assert state.next_due_at > time.time() + 3000


def test_hot_edit_idle_job_reseeds_next_due(tmp_path: Path) -> None:
    crons = tmp_path / "persona" / "crons.yaml"
    config = _make_config(
        tmp_path,
        persona_crons={"jobs": [_script_job(schedule={"every_seconds": 3600})]},
    )
    svc = _make_service(config, tmp_path)
    svc._tick()
    before = svc._state.get("tidy").next_due_at
    assert before > time.time() + 3000
    crons.write_text(yaml.safe_dump({"jobs": [_script_job(schedule={"every_seconds": 60})]}))
    svc._tick()
    after = svc._state.get("tidy").next_due_at
    assert after < before
    assert after <= time.time() + 60


def test_removed_job_midrun_still_delivers(tmp_path: Path) -> None:
    crons = tmp_path / "persona" / "crons.yaml"
    job = _script_job(delivery="digest", call={"type": "script", "command": "sleep 2"})
    config = _make_config(tmp_path, persona_crons={"jobs": [job]})
    svc = _make_service(config, tmp_path)
    svc._tick()
    state = svc._state.get("tidy")
    state.next_due_at = time.time() - 1
    svc._state.update(state)
    svc._tick()
    state = svc._state.get("tidy")
    plan_id, task_id = state.running_plan_id, state.running_task_id
    # Delete the job from the file while the run is in flight.
    crons.write_text(yaml.safe_dump({"jobs": []}))
    svc._tick()
    assert "tidy" not in svc._jobs
    assert svc._state.get("tidy") is not None  # kept while running
    deadline = time.time() + 5
    task = None
    while time.time() < deadline:
        task = svc._plan_manager.get_task(plan_id, task_id)
        if task.status in (TaskStatus.DONE, TaskStatus.FAILED):
            break
        time.sleep(0.05)
    svc._on_event(Event(type="task.completed", payload={"plan_id": plan_id, "task_id": task.id}))
    # Result files still land under the firing spec; the row is then dropped.
    assert _last_file(tmp_path)["job_id"] == "tidy"
    assert svc._state.get("tidy") is None
