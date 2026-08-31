"""Tests for TaskEngine and WorkerPool."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from diploid_agent.config import TaskConfig
from diploid_agent.plan.manager import PlanManager
from diploid_agent.plan.models import Task, TaskStatus, TaskType
from diploid_agent.runtime.event_bus import Event, EventBus
from diploid_agent.task.engine import TaskEngine
from diploid_agent.task.worker import WorkerPool


def _fixture_engine(tmp_path: Path, workers: int = 2, shell_timeout: float = 5.0):
    mgr = PlanManager(tmp_path)
    bus = EventBus()
    bus.start()
    engine = TaskEngine(
        mgr,
        bus,
        max_workers=workers,
        shell_timeout=shell_timeout,
    )
    return engine, mgr, bus


def _wait_for_task(
    mgr: PlanManager,
    plan_id: str,
    task_id: str,
    timeout: float = 2.0,
) -> Task:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = mgr.get_task(plan_id, task_id)
        if task is not None and task.status in (TaskStatus.DONE, TaskStatus.FAILED):
            return task
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} did not finish in time")


def test_task_engine_calls_start_and_done_callbacks(tmp_path: Path) -> None:
    engine, mgr, bus = _fixture_engine(tmp_path)
    calls: list[tuple[str, ...]] = []

    def on_start(plan_id: str, task: Task) -> None:
        calls.append(("start", plan_id, task.name))

    def on_done(plan_id: str, task: Task, outcome: tuple[str, str, int]) -> None:
        calls.append(("done", plan_id, task.name, outcome[0], outcome[1], outcome[2]))

    engine._on_task_start = on_start
    engine._on_task_done = on_done

    plan = mgr.create_plan("callback", tasks=[Task(name="echo", command="echo hi")])
    task = engine.start_task(plan.id)
    finished = _wait_for_task(mgr, plan.id, task.id)

    assert finished.status == TaskStatus.DONE
    assert any(c[0] == "start" and c[2] == "echo" for c in calls)
    assert any(c[0] == "done" and c[2] == "echo" and c[5] == 0 for c in calls)
    engine.shutdown()
    bus.stop()


def test_shell_task_executes_and_completes(tmp_path: Path) -> None:
    engine, mgr, bus = _fixture_engine(tmp_path)
    plan = mgr.create_plan("shell", tasks=[Task(name="echo", command="echo hello")])

    task = engine.start_task(plan.id)
    finished = _wait_for_task(mgr, plan.id, task.id)

    assert finished.status == TaskStatus.DONE
    assert "hello" in (finished.result or "")
    engine.shutdown()
    bus.stop()


def test_shell_failure_marks_task_failed(tmp_path: Path) -> None:
    engine, mgr, bus = _fixture_engine(tmp_path)
    plan = mgr.create_plan("fails", tasks=[Task(name="fails", command="exit 7")])

    task = engine.start_task(plan.id)
    finished = _wait_for_task(mgr, plan.id, task.id)

    assert finished.status == TaskStatus.FAILED
    engine.shutdown()
    bus.stop()


def test_noop_task(tmp_path: Path) -> None:
    engine, mgr, bus = _fixture_engine(tmp_path)
    plan = mgr.create_plan("noop", tasks=[Task(name="noop", type=TaskType.NOOP)])

    task = engine.start_task(plan.id)
    finished = _wait_for_task(mgr, plan.id, task.id)

    assert finished.status == TaskStatus.DONE
    assert (finished.result or "").strip() == "noop"
    engine.shutdown()
    bus.stop()


def test_task_chain_starts_dependent_tasks(tmp_path: Path) -> None:
    engine, mgr, bus = _fixture_engine(tmp_path)
    t1 = Task(name="first", command="echo first", cwd=tmp_path)
    t2 = Task(name="second", command="echo second", depends_on=[t1.id], cwd=tmp_path)
    plan = mgr.create_plan("chain", tasks=[t1, t2])

    engine.start_task(plan.id, t1.id)
    finished = _wait_for_task(mgr, plan.id, t2.id, timeout=5.0)

    assert finished.status == TaskStatus.DONE
    assert "second" in (finished.result or "")
    engine.shutdown()
    bus.stop()


def test_event_bus_emits_completion(tmp_path: Path) -> None:
    engine, mgr, bus = _fixture_engine(tmp_path)
    captured: list[Event] = []

    def on_event(event: Event) -> None:
        if event.type == "task.completed":
            captured.append(event)

    bus.subscribe(on_event)
    plan = mgr.create_plan("emit", tasks=[Task(name="echo", command="echo event")])
    engine.start_task(plan.id)

    _wait_for_task(mgr, plan.id, plan.tasks[0].id)

    # Give the bus thread time to deliver.
    for _ in range(40):
        if captured:
            break
        time.sleep(0.05)

    assert len(captured) == 1
    assert captured[0].payload["plan_id"] == plan.id
    assert captured[0].payload["task_id"] == plan.tasks[0].id
    engine.shutdown()
    bus.stop()


def test_start_unknown_task_raises(tmp_path: Path) -> None:
    engine, mgr, bus = _fixture_engine(tmp_path)
    plan = mgr.create_plan("empty")

    with pytest.raises(ValueError, match="No ready tasks"):
        engine.start_task(plan.id)

    engine.shutdown()
    bus.stop()


def _acp_fixture_engine(tmp_path: Path, workers: int = 2, shell_timeout: float = 5.0):
    from diploid_agent.config import Config, DiploidConfig, HarnessConfig, PersonaConfig
    from diploid_agent.engine.fake import FakeAgentEngine

    config = Config(
        diploid=DiploidConfig(bin="/bin/echo", model="test-model"),
        persona=PersonaConfig(name="test", profile_root=tmp_path),
        harness=HarnessConfig(sessions_root=tmp_path / "sessions"),
    )
    mgr = PlanManager(tmp_path / "plans")
    bus = EventBus()
    bus.start()
    fake = FakeAgentEngine()
    engine = TaskEngine(
        mgr,
        bus,
        engine=fake,
        model="test-model",
        acp_timeout=5.0,
        config=config,
        max_workers=workers,
        shell_timeout=shell_timeout,
    )
    return engine, mgr, bus, fake


def test_acp_task_runs_engine_and_completes(tmp_path: Path) -> None:
    engine, mgr, bus, fake = _acp_fixture_engine(tmp_path)
    fake.replies = ["plan result"]
    plan = mgr.create_plan(
        "acp",
        tasks=[Task(name="ask", type=TaskType.ACP, prompt="do the thing")],
    )

    task = engine.start_task(plan.id)
    finished = _wait_for_task(mgr, plan.id, task.id)

    assert finished.status == TaskStatus.DONE
    assert (finished.result or "").strip() == "plan result"
    engine.shutdown()
    bus.stop()


def test_acp_task_falls_back_to_command(tmp_path: Path) -> None:
    engine, mgr, bus, fake = _acp_fixture_engine(tmp_path)
    fake.replies = ["fallback result"]
    plan = mgr.create_plan(
        "acp",
        tasks=[Task(name="ask", type=TaskType.ACP, command="the command")],
    )

    task = engine.start_task(plan.id)
    finished = _wait_for_task(mgr, plan.id, task.id)

    assert finished.status == TaskStatus.DONE
    assert "fallback result" in (finished.result or "")
    engine.shutdown()
    bus.stop()


def test_acp_task_uses_task_acp_model(tmp_path: Path) -> None:
    engine, mgr, bus, fake = _acp_fixture_engine(tmp_path)
    fake.replies = ["default", "overridden"]

    default_plan = mgr.create_plan(
        "acp-default",
        tasks=[Task(name="default", type=TaskType.ACP, prompt="default model")],
    )
    task = engine.start_task(default_plan.id)
    finished = _wait_for_task(mgr, default_plan.id, task.id)
    assert finished.status == TaskStatus.DONE
    assert fake.call_log[-1][1].model == "test-model"

    override_plan = mgr.create_plan(
        "acp-override",
        tasks=[Task(name="override", type=TaskType.ACP, prompt="task model", acp_model="glm-5-2")],
    )
    task = engine.start_task(override_plan.id)
    finished = _wait_for_task(mgr, override_plan.id, task.id)
    assert finished.status == TaskStatus.DONE
    assert fake.call_log[-1][1].model == "glm-5-2"

    engine.shutdown()
    bus.stop()


def test_acp_task_without_engine_fails(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    bus = EventBus()
    bus.start()
    engine = TaskEngine(mgr, bus, max_workers=2, shell_timeout=5.0)
    plan = mgr.create_plan(
        "acp",
        tasks=[Task(name="ask", type=TaskType.ACP, command="test")],
    )

    task = engine.start_task(plan.id)
    finished = _wait_for_task(mgr, plan.id, task.id)

    assert finished.status == TaskStatus.FAILED
    assert "ACP engine not configured" in (finished.log or "")
    engine.shutdown()
    bus.stop()


def test_subagent_task_runs_engine_and_completes(tmp_path: Path) -> None:
    engine, mgr, bus, fake = _acp_fixture_engine(tmp_path)
    fake.replies = ["subagent result"]
    plan = mgr.create_plan(
        "subagent",
        tasks=[Task(name="ask", type=TaskType.SUBAGENT, prompt="do the thing")],
    )

    task = engine.start_task(plan.id)
    finished = _wait_for_task(mgr, plan.id, task.id)

    assert finished.status == TaskStatus.DONE
    assert (finished.result or "").strip() == "subagent result"
    engine.shutdown()
    bus.stop()


def test_worker_pool_resizes_at_runtime() -> None:
    pool = WorkerPool(max_workers=2)
    assert pool.max_workers == 2

    pool.resize(8)
    assert pool.max_workers == 8
    assert pool.is_running()

    # Resizing to the same value should be a no-op.
    pool.resize(8)
    assert pool.max_workers == 8

    pool.shutdown()


def test_task_engine_enforces_enabled_types(tmp_path: Path) -> None:
    engine, mgr, bus = _fixture_engine(tmp_path)
    plan = mgr.create_plan("typed", tasks=[Task(name="echo", command="echo hi")])

    # Disable shell tasks at runtime.
    engine.task_config.enabled_types = ["noop", "acp"]

    task = engine.start_task(plan.id)
    finished = _wait_for_task(mgr, plan.id, task.id)

    assert finished.status == TaskStatus.FAILED
    assert "disabled" in (finished.log or "")
    engine.shutdown()
    bus.stop()


def test_task_engine_shell_timeout_is_live(tmp_path: Path) -> None:
    engine, mgr, bus = _fixture_engine(tmp_path, shell_timeout=60.0)
    plan = mgr.create_plan(
        "slow",
        tasks=[Task(name="sleep", command="sleep 0.5")],
    )

    # Lower the timeout before starting the task.
    engine.task_config.shell_timeout = 0.1

    task = engine.start_task(plan.id)
    finished = _wait_for_task(mgr, plan.id, task.id, timeout=2.0)

    assert finished.status == TaskStatus.FAILED
    assert "Timed out" in (finished.log or "")
    engine.shutdown()
    bus.stop()


def test_task_engine_worker_count_is_live(tmp_path: Path) -> None:
    engine, mgr, bus = _fixture_engine(tmp_path, workers=2)
    assert engine.task_config.workers == 2

    engine.task_config.workers = 6
    plan = mgr.create_plan("noop", tasks=[Task(name="noop", type=TaskType.NOOP)])
    engine.start_task(plan.id)

    assert engine._pool.max_workers == 6
    engine.shutdown()
    bus.stop()


def test_task_config_validates_workers() -> None:
    with pytest.raises(ValueError):
        TaskConfig(workers=0)


def test_task_config_validates_shell_timeout() -> None:
    with pytest.raises(ValueError):
        TaskConfig(shell_timeout=-1.0)


def test_worker_pool_resize_rejects_invalid() -> None:
    pool = WorkerPool(max_workers=2)
    with pytest.raises(ValueError):
        pool.resize(0)
    with pytest.raises(ValueError):
        pool.resize(-1)
    pool.shutdown()


def test_worker_pool_rejects_invalid_init() -> None:
    with pytest.raises(ValueError):
        WorkerPool(max_workers=0)


def test_task_config_validates_acp_fields() -> None:
    with pytest.raises(ValueError):
        TaskConfig(acp_timeout=0.0)
    with pytest.raises(ValueError):
        TaskConfig(acp_timeout=-1.0)
    with pytest.raises(ValueError):
        TaskConfig(acp_model="")
    with pytest.raises(ValueError):
        TaskConfig(acp_model="   ")
