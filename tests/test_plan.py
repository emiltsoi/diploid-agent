"""Tests for PlanManager and Plan models."""

from __future__ import annotations

from pathlib import Path

import pytest

from diploid_agent.plan.manager import PlanManager
from diploid_agent.plan.models import PlanStatus, Task, TaskStatus, TaskType


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


@pytest.mark.parametrize("flag", ["partial", "timed_out", "cancelled"])
def test_complete_task_incomplete_flags(tmp_path: Path, flag: str) -> None:
    mgr = PlanManager(tmp_path)
    plan = mgr.create_plan("flags", tasks=[Task(name="t")])
    mgr.start_task(plan.id, plan.tasks[0].id, now=0.0)
    mgr.complete_task(plan.id, plan.tasks[0].id, result="r", **{flag: True})

    task = mgr.get_plan(plan.id).tasks[0]
    assert task.status == TaskStatus.INCOMPLETE
    assert getattr(task, flag) is True
    assert task.result == "r"


def test_incomplete_task_unblocks_dependents(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    t1 = Task(name="first", command="echo first")
    t2 = Task(name="second", command="echo second", depends_on=[t1.id])
    plan = mgr.create_plan("partial-chain", tasks=[t1, t2])

    mgr.start_task(plan.id, t1.id, now=0.0)
    mgr.complete_task(plan.id, t1.id, result="partial", partial=True)

    plan = mgr.get_plan(plan.id)
    assert plan is not None
    assert plan.tasks[0].status == TaskStatus.INCOMPLETE
    assert plan.tasks[1].status == TaskStatus.READY


def test_incomplete_task_completes_plan(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    plan = mgr.create_plan("partial-finish", tasks=[Task(name="only")])
    mgr.start_task(plan.id, plan.tasks[0].id, now=0.0)
    mgr.complete_task(plan.id, plan.tasks[0].id, result="timed", timed_out=True)

    loaded = mgr.get_plan(plan.id)
    assert loaded is not None
    assert loaded.status == PlanStatus.COMPLETED
    assert loaded.tasks[0].status == TaskStatus.INCOMPLETE
    assert loaded.tasks[0].timed_out is True


def test_incomplete_status_persists(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    plan = mgr.create_plan("persisted", tasks=[Task(name="t1")])
    mgr.complete_task(plan.id, plan.tasks[0].id, result="partial", cancelled=True)

    mgr2 = PlanManager(tmp_path)
    loaded = mgr2.get_plan(plan.id)
    assert loaded is not None
    assert loaded.tasks[0].status == TaskStatus.INCOMPLETE
    assert loaded.tasks[0].cancelled is True


def test_on_change_fires_once_per_mutator(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    fired: list[str] = []
    mgr.on_change = lambda plan: fired.append(plan.id)

    plan = mgr.create_plan("board", chat_id="123", tasks=[Task(name="a")])
    task_b = mgr.add_task(plan.id, Task(name="b"))
    assert task_b is not None
    mgr.start_task(plan.id, plan.tasks[0].id)
    mgr.complete_task(plan.id, plan.tasks[0].id, result="ok")
    mgr.fail_task(plan.id, task_b.id, log="nope")

    assert fired == [plan.id] * 5


def test_on_change_not_fired_on_reads_or_failed_mutations(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    fired: list[str] = []
    mgr.on_change = lambda plan: fired.append(plan.id)
    plan = mgr.create_plan("quiet", tasks=[Task(name="a")])
    fired.clear()

    mgr.get_plan(plan.id)
    mgr.list_plans()
    mgr.get_task(plan.id, plan.tasks[0].id)
    mgr.get_ready_tasks(plan.id)
    assert mgr.add_task("missing-plan", Task(name="b")) is None
    assert mgr.add_task(plan.id, Task(id=plan.tasks[0].id, name="dup")) is None
    assert mgr.start_task(plan.id, "missing-task") is None
    assert mgr.complete_task(plan.id, "missing-task") is None
    assert mgr.fail_task(plan.id, "missing-task") is None

    assert fired == []


def test_on_change_fires_after_transaction_released(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)

    def handler(plan) -> None:
        # Re-enters the manager for selection; would deadlock if the
        # callback fired while _transaction still held the lock.
        assert mgr.get_plan(plan.id) is not None
        assert mgr.list_plans(plan.chat_id)

    mgr.on_change = handler
    mgr.create_plan("reentry", chat_id="7", tasks=[Task(name="a")])


def test_on_change_callback_exception_does_not_fail_mutation(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)

    def boom(_plan) -> None:
        raise RuntimeError("board broke")

    mgr.on_change = boom
    plan = mgr.create_plan("resilient", tasks=[Task(name="a")])

    loaded = mgr.get_plan(plan.id)
    assert loaded is not None
    assert loaded.name == "resilient"
