"""Tests for PlanManager and Plan models."""

from __future__ import annotations

from pathlib import Path

from devin_fleet_harness.plan.manager import PlanManager
from devin_fleet_harness.plan.models import PlanStatus, Task, TaskStatus, TaskType


def test_create_plan_with_ready_task(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    plan = mgr.create_plan("test", tasks=[Task(name="a", command="echo a")])

    assert plan.name == "test"
    assert plan.status == PlanStatus.ACTIVE
    assert len(plan.tasks) == 1
    assert plan.tasks[0].status == TaskStatus.READY


def test_plan_persistence(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    plan = mgr.create_plan("persisted", tasks=[Task(name="t1")])
    mgr.complete_task(plan.id, plan.tasks[0].id, result="ok")

    mgr2 = PlanManager(tmp_path)
    loaded = mgr2.get_plan(plan.id)
    assert loaded is not None
    assert loaded.name == "persisted"
    assert loaded.tasks[0].status == TaskStatus.DONE
    assert loaded.tasks[0].result == "ok"


def test_task_dependencies_become_ready(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    t1 = Task(name="first", command="echo first")
    t2 = Task(name="second", command="echo second", depends_on=[t1.id])
    plan = mgr.create_plan("chain", tasks=[t1, t2])

    assert plan.tasks[0].status == TaskStatus.READY
    assert plan.tasks[1].status == TaskStatus.PENDING

    mgr.start_task(plan.id, t1.id, now=0.0)
    mgr.complete_task(plan.id, t1.id, result="done")

    plan = mgr.get_plan(plan.id)
    assert plan is not None
    assert plan.tasks[1].status == TaskStatus.READY


def test_failed_dependency_blocks_task(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    t1 = Task(name="first", command="false")
    t2 = Task(name="second", depends_on=[t1.id])
    plan = mgr.create_plan("blocked", tasks=[t1, t2])

    mgr.start_task(plan.id, t1.id, now=0.0)
    mgr.fail_task(plan.id, t1.id, log="failed")

    plan = mgr.get_plan(plan.id)
    assert plan is not None
    assert plan.tasks[1].status == TaskStatus.BLOCKED


def test_plan_scoped_to_chat_id(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    p1 = mgr.create_plan("chat-1", chat_id="c1")
    _ = mgr.create_plan("chat-2", chat_id="c2")

    assert len(mgr.list_plans(chat_id="c1")) == 1
    assert mgr.list_plans(chat_id="c1")[0].id == p1.id
    assert len(mgr.list_plans(chat_id="c2")) == 1
    assert len(mgr.list_plans()) == 2


def test_add_task_to_existing_plan(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    plan = mgr.create_plan("addable")
    task = mgr.add_task(plan.id, Task(name="extra"))

    assert task is not None
    loaded = mgr.get_plan(plan.id)
    assert loaded is not None
    assert len(loaded.tasks) == 1
    assert loaded.tasks[0].status == TaskStatus.READY


def test_missing_dependency_marks_blocked(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    plan = mgr.create_plan("missing", tasks=[Task(name="orphan", depends_on=["no-such-task"])])
    assert plan.tasks[0].status == TaskStatus.BLOCKED


def test_task_type_round_trip(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    plan = mgr.create_plan("roundtrip", tasks=[Task(name="noop", type=TaskType.NOOP)])
    data = plan.tasks[0].to_dict()
    restored = Task.from_dict(data)
    assert restored.type == TaskType.NOOP
    assert restored.status == TaskStatus.READY


def test_plan_status_completed(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    plan = mgr.create_plan("finish", tasks=[Task(name="only")])
    mgr.start_task(plan.id, plan.tasks[0].id, now=0.0)
    mgr.complete_task(plan.id, plan.tasks[0].id)

    loaded = mgr.get_plan(plan.id)
    assert loaded is not None
    assert loaded.status == PlanStatus.COMPLETED
