"""Debounced per-chat task board — one edited message per plan list."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from diploid_agent.notifier import Notifier, TelegramNotifier
from diploid_agent.plan.models import Plan, PlanStatus, Task, TaskStatus

if TYPE_CHECKING:
    from diploid_agent.plan.manager import PlanManager

logger = logging.getLogger(__name__)

_TERMINAL_PLAN_STATUSES = frozenset({PlanStatus.COMPLETED, PlanStatus.FAILED})

_TASK_GLYPHS = {
    TaskStatus.PENDING: "☐",
    TaskStatus.READY: "☐",
    TaskStatus.BLOCKED: "☐",
    TaskStatus.RUNNING: "◐",
    TaskStatus.DONE: "☑",
    TaskStatus.INCOMPLETE: "◑",
    TaskStatus.FAILED: "✗",
}
_CANCELLED_GLYPH = "⊘"

_MAX_VISIBLE_TASKS = 12
_MAX_NAME_CHARS = 60


def _task_glyph(task: Task) -> str:
    if task.cancelled:
        return _CANCELLED_GLYPH
    return _TASK_GLYPHS.get(task.status, "☐")


def _task_suffix(task: Task) -> str:
    if task.status == TaskStatus.RUNNING:
        return " ← running"
    if task.cancelled:
        return " ← cancelled"
    if task.timed_out:
        return " ← timed out"
    if task.partial:
        return " ← partial"
    if task.status == TaskStatus.FAILED:
        return " ← failed"
    return ""


def render_plan(plan: Plan) -> str:
    """Render a plan as a plain-text checklist for the board message."""
    lines: list[str] = []
    if plan.name:
        lines.append(plan.name)
    tasks = plan.tasks
    done_count = sum(1 for t in tasks if t.status == TaskStatus.DONE)
    if len(tasks) > _MAX_VISIBLE_TASKS:
        visible = [t for t in tasks if t.status != TaskStatus.DONE]
        overflow = 0
        if len(visible) > _MAX_VISIBLE_TASKS:
            overflow = len(visible) - _MAX_VISIBLE_TASKS
            visible = visible[:_MAX_VISIBLE_TASKS]
        for task in visible:
            name = task.name[:_MAX_NAME_CHARS]
            lines.append(f"{_task_glyph(task)} {name}{_task_suffix(task)}")
        if overflow:
            lines.append(f"… +{overflow} more")
        if done_count:
            lines.append(f"— {done_count} done")
    else:
        for task in tasks:
            name = task.name[:_MAX_NAME_CHARS]
            lines.append(f"{_task_glyph(task)} {name}{_task_suffix(task)}")
    failed = plan.status == PlanStatus.FAILED or any(t.status == TaskStatus.FAILED for t in tasks)
    footer = f"— {done_count}/{len(tasks)} done"
    if failed:
        footer += " · failed"
    lines.append(footer)
    return "\n".join(lines)


class RuntimeTaskBoard:
    """Coalesce plan mutations into debounced per-chat board updates.

    ``handle`` is the ``PlanManager.on_change`` subscriber: it only marks a
    chat dirty and returns, so mutator threads (task engine, cron,
    lifecycle) never sleep inside an edit. The worker thread owns the coalesce
    window and the per-chat send floor — the notifier path has no throttle of
    its own, so ``min_interval`` is the only rate control.
    """

    def __init__(
        self,
        plan_manager: PlanManager,
        notifier: Notifier,
        *,
        min_interval: float = 2.0,
        coalesce: float = 1.0,
    ) -> None:
        self._plans = plan_manager
        self._notifier = notifier
        self._min_interval = min_interval
        self._coalesce = coalesce
        self._cv = threading.Condition()
        self._dirty: set[str] = set()
        self._last_sent: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        thread = threading.Thread(target=self._run, daemon=True, name="task-board")
        self._thread = thread
        thread.start()

    def handle(self, plan: Plan) -> None:
        """``PlanManager.on_change`` entry point — never blocks, never raises."""
        chat_id = plan.chat_id
        if chat_id is None or self._stop.is_set():
            return
        with self._cv:
            self._dirty.add(chat_id)
            self._cv.notify()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                self._cv.wait_for(lambda: bool(self._dirty) or self._stop.is_set())
                if self._stop.is_set():
                    return
            # Absorb the burst first, then drain — mutations landing inside
            # the coalesce window join this send instead of firing a second.
            if self._stop.wait(self._coalesce):
                return
            with self._cv:
                chats = self._dirty
                self._dirty = set()
            for chat_id in chats:
                if self._stop.is_set():
                    return
                self._send(chat_id)

    def _select(self, chat_id: str) -> Plan | None:
        """Newest non-terminal plan wins; else the newest terminal (frozen)."""
        plans = self._plans.list_plans(chat_id)
        for plan in plans:
            if plan.status not in _TERMINAL_PLAN_STATUSES:
                return plan
        return plans[0] if plans else None

    def _send(self, chat_id: str) -> None:
        elapsed = time.monotonic() - self._last_sent.get(chat_id, 0.0)
        wait = self._min_interval - elapsed
        if wait > 0 and self._stop.wait(wait):
            return
        plan = self._select(chat_id)
        if plan is None:
            return
        try:
            self._notifier.update_task_board(chat_id, render_plan(plan))
        except Exception:
            logger.exception("Task board update failed for chat %s", chat_id)
        finally:
            self._last_sent[chat_id] = time.monotonic()

    def stop(self) -> None:
        """Stop the debounce worker; pending updates are dropped."""
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def build_task_board(runtime: Any) -> RuntimeTaskBoard | None:
    """Construct a board wired to a direct notifier, or None when off.

    Returns None when the feature flag is off or no Telegram token exists.
    The notifier is a dedicated ``TelegramNotifier`` — a board needs the Bot
    API's edit support, so webhook/noop notifiers are skipped entirely; and
    it bypasses the outbox (the ``_create_direct_notifier`` pattern) so the
    board survives ``outbox_delivery`` being toggled. It carries the
    placeholder state dir so board ids sit beside ``{chat_id}.ask.json``.
    """
    config = runtime.config
    token = config.harness.telegram.token
    if not config.harness.telegram.task_board or not token:
        return None
    notifier = TelegramNotifier(
        token,
        metrics=runtime.metrics,
        state_dir=runtime.sessions_root / ".poller-placeholders",
    )
    return RuntimeTaskBoard(
        plan_manager=runtime.plan_manager,
        notifier=notifier,
        min_interval=config.harness.telegram.min_edit_message_interval,
    )
