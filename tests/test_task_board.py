"""Tests for RuntimeTaskBoard — selection, render, and debounce."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from diploid_agent.notifier import NoopNotifier, TelegramNotifier
from diploid_agent.plan.manager import PlanManager
from diploid_agent.plan.models import Plan, PlanStatus, Task, TaskStatus
from diploid_agent.runtime.task_board import (
    RuntimeTaskBoard,
    build_task_board,
    render_plan,
)


class _RecordingNotifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []
        self.event = threading.Event()
        self.fail = False

    def update_task_board(self, chat_id: str, text: str) -> bool:
        if self.fail:
            raise RuntimeError("telegram down")
        self.calls.append((chat_id, text, time.monotonic()))
        self.event.set()
        return True


def _make_board(
    tmp_path: Path,
    *,
    coalesce: float = 0.05,
    min_interval: float = 0.0,
) -> tuple[PlanManager, _RecordingNotifier, RuntimeTaskBoard]:
    mgr = PlanManager(tmp_path)
    notifier = _RecordingNotifier()
    board = RuntimeTaskBoard(mgr, notifier, min_interval=min_interval, coalesce=coalesce)
    mgr.on_change = board.handle
    board.start()
    return mgr, notifier, board


def _wait_calls(notifier: _RecordingNotifier, count: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while len(notifier.calls) < count and time.monotonic() < deadline:
        time.sleep(0.01)


def test_render_glyphs_for_each_status() -> None:
    plan = Plan(
        name="p",
        tasks=[
            Task(name="a", status=TaskStatus.DONE),
            Task(name="b", status=TaskStatus.RUNNING),
            Task(name="c", status=TaskStatus.PENDING),
            Task(name="d", status=TaskStatus.INCOMPLETE),
            Task(name="e", status=TaskStatus.FAILED),
            Task(name="f", status=TaskStatus.BLOCKED),
        ],
    )
    text = render_plan(plan)
    assert "☑ a" in text
    assert "◐ b ← running" in text
    assert "☐ c" in text
    assert "◑ d" in text
    assert "✗ e ← failed" in text
    assert "☐ f" in text
    assert text.endswith("— 1/6 done · failed")


def test_render_cancelled_glyph_and_suffixes() -> None:
    plan = Plan(
        name="p",
        tasks=[
            Task(name="a", status=TaskStatus.INCOMPLETE, cancelled=True),
            Task(name="b", status=TaskStatus.INCOMPLETE, timed_out=True),
            Task(name="c", status=TaskStatus.INCOMPLETE, partial=True),
        ],
    )
    text = render_plan(plan)
    assert "⊘ a ← cancelled" in text
    assert "◑ b ← timed out" in text
    assert "◑ c ← partial" in text


def test_render_folds_done_tail_when_long() -> None:
    tasks = [Task(name=f"done-{i}", status=TaskStatus.DONE) for i in range(10)]
    tasks += [Task(name=f"work-{i}", status=TaskStatus.PENDING) for i in range(5)]
    text = render_plan(Plan(name="p", tasks=tasks))
    assert "☑ done-0" not in text
    assert "— 10 done" in text
    assert "☐ work-4" in text
    assert text.endswith("— 10/15 done")


def test_render_truncates_long_names() -> None:
    plan = Plan(name="p", tasks=[Task(name="x" * 100)])
    line = render_plan(plan).splitlines()[1]
    assert line == f"☐ {'x' * 60}"


def test_board_sends_on_plan_creation(tmp_path: Path) -> None:
    mgr, notifier, board = _make_board(tmp_path)
    try:
        mgr.create_plan("work", chat_id="123", tasks=[Task(name="a")])
        _wait_calls(notifier, 1)
        assert notifier.calls[0][0] == "123"
        assert "☐ a" in notifier.calls[0][1]
        assert notifier.calls[0][1].endswith("— 0/1 done")
    finally:
        board.stop()


def test_board_coalesces_burst_into_one_send(tmp_path: Path) -> None:
    mgr, notifier, board = _make_board(tmp_path)
    try:
        plan = mgr.create_plan("work", chat_id="123", tasks=[Task(name="a"), Task(name="b")])
        mgr.start_task(plan.id, plan.tasks[0].id)
        mgr.complete_task(plan.id, plan.tasks[0].id, result="ok")
        mgr.start_task(plan.id, plan.tasks[1].id)
        _wait_calls(notifier, 1)
        time.sleep(0.2)
        assert len(notifier.calls) == 1
        text = notifier.calls[0][1]
        assert "☑ a" in text
        assert "◐ b" in text
    finally:
        board.stop()


def test_board_enforces_min_interval_floor(tmp_path: Path) -> None:
    mgr, notifier, board = _make_board(tmp_path, coalesce=0.02, min_interval=0.3)
    try:
        plan = mgr.create_plan("work", chat_id="123", tasks=[Task(name="a")])
        _wait_calls(notifier, 1)
        mgr.add_task(plan.id, Task(name="b"))
        _wait_calls(notifier, 2)
        gap = notifier.calls[1][2] - notifier.calls[0][2]
        assert gap >= 0.25
    finally:
        board.stop()


def test_board_skips_chatless_plans(tmp_path: Path) -> None:
    mgr, notifier, board = _make_board(tmp_path)
    try:
        mgr.create_plan("cron-plan", chat_id=None, tasks=[Task(name="a")])
        time.sleep(0.2)
        assert notifier.calls == []
    finally:
        board.stop()


def test_board_prefers_newest_active_over_terminal(tmp_path: Path) -> None:
    mgr, notifier, board = _make_board(tmp_path)
    try:
        old = mgr.create_plan("old", chat_id="123", tasks=[Task(name="a")])
        mgr.complete_task(old.id, old.tasks[0].id, result="ok")
        _wait_calls(notifier, 1)
        notifier.calls.clear()
        notifier.event.clear()

        mgr.create_plan("new", chat_id="123", tasks=[Task(name="z")])
        _wait_calls(notifier, 1)
        assert "new" in notifier.calls[0][1]
        assert "☐ z" in notifier.calls[0][1]
    finally:
        board.stop()


def test_board_freezes_on_terminal_then_reopens(tmp_path: Path) -> None:
    mgr, notifier, board = _make_board(tmp_path)
    try:
        plan = mgr.create_plan("work", chat_id="123", tasks=[Task(name="a")])
        mgr.complete_task(plan.id, plan.tasks[0].id, result="ok")
        _wait_calls(notifier, 1)
        text = notifier.calls[0][1]
        assert "☑ a" in text
        assert text.endswith("— 1/1 done")
    finally:
        board.stop()


def test_board_survives_notifier_exception(tmp_path: Path) -> None:
    mgr, notifier, board = _make_board(tmp_path)
    try:
        notifier.fail = True
        mgr.create_plan("work", chat_id="123", tasks=[Task(name="a")])
        time.sleep(0.2)
        assert notifier.calls == []
        notifier.fail = False
        plan = mgr.list_plans("123")[0]
        mgr.add_task(plan.id, Task(name="b"))
        _wait_calls(notifier, 1)
        assert "☐ b" in notifier.calls[0][1]
    finally:
        board.stop()


def test_board_handle_after_stop_is_quiet(tmp_path: Path) -> None:
    mgr, notifier, board = _make_board(tmp_path)
    board.stop()
    mgr.create_plan("work", chat_id="123", tasks=[Task(name="a")])
    time.sleep(0.15)
    assert notifier.calls == []


def test_notifier_abc_default_is_unsupported() -> None:
    assert NoopNotifier().update_task_board("1", "x") is False


def test_build_task_board_gating(tmp_path: Path) -> None:
    tg = SimpleNamespace(task_board=True, token="tok", min_edit_message_interval=2.0)
    runtime = SimpleNamespace(
        config=SimpleNamespace(harness=SimpleNamespace(telegram=tg)),
        sessions_root=tmp_path,
        plan_manager=PlanManager(tmp_path / "plans"),
        metrics=None,
    )
    board = build_task_board(runtime)
    assert board is not None
    assert isinstance(board._notifier, TelegramNotifier)
    assert board._notifier.state_dir == tmp_path / ".poller-placeholders"
    board.stop()

    tg.task_board = False
    assert build_task_board(runtime) is None
    tg.task_board = True
    tg.token = ""
    assert build_task_board(runtime) is None


def test_board_selects_terminal_plan_when_all_done(tmp_path: Path) -> None:
    mgr = PlanManager(tmp_path)
    plan = mgr.create_plan("done", chat_id="123", tasks=[Task(name="a")])
    mgr.complete_task(plan.id, plan.tasks[0].id, result="ok")
    board = RuntimeTaskBoard(mgr, _RecordingNotifier(), min_interval=0.0, coalesce=0.0)
    selected = board._select("123")
    assert selected is not None
    assert selected.status == PlanStatus.COMPLETED
