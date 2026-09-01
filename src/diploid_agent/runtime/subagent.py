"""Background subagent start, completion, and status helpers."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from diploid_agent.dispatch import Dispatch, DispatchStatus, DispatchStore
from diploid_agent.models import ChatResult, WakeEvent
from diploid_agent.plan.models import Task, TaskStatus, TaskType

logger = logging.getLogger(__name__)


def _locked(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run a RuntimeSubagent method under the runtime RLock."""

    @functools.wraps(method)
    def wrapper(self: RuntimeSubagent, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class RuntimeSubagent:
    """Background subagent start, completion, and status."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @property
    def config(self) -> Any:
        return self._runtime.config

    @property
    def _lock(self) -> Any:
        return self._runtime._lock

    @property
    def dispatch_store(self) -> DispatchStore:
        return self._runtime.dispatch_store

    @property
    def engine(self) -> Any:
        return self._runtime.engine

    @property
    def _outbox(self) -> Any:
        return self._runtime._outbox

    @property
    def _prompts(self) -> Any:
        return self._runtime._prompts

    @property
    def acp_client(self) -> Any:
        return getattr(self._runtime, "acp_client", None)

    @property
    def mcp(self) -> Any:
        return self._runtime.mcp

    @property
    def skills(self) -> Any:
        return self._runtime.skills

    @property
    def _mcp_skills(self) -> Any:
        return self._runtime._mcp_skills

    @property
    def _chat_store(self) -> Any:
        return self._runtime._chat_store

    @property
    def _runtime_plugins(self) -> Any:
        return self._runtime._runtime_plugins

    @_locked
    def subagent_start(
        self,
        chat_id: str,
        prompt: str,
        *,
        context: str | None = None,
        model: str | None = None,
        cwd: Path | None = None,
        acp_timeout: float | None = None,
    ) -> ChatResult:
        """Start a background ACP subagent and return a dispatch id.

        The subagent runs in a fresh AcpEngine, so it survives the parent turn
        being stopped or killed. When it completes, the harness starts a new
        turn for the chat via the existing dispatch/continue flow.
        """
        record = self._runtime._active_record(chat_id)
        if record is None:
            return ChatResult(reply="No active session for this chat.")

        use_model = model or self._runtime._model(record)
        started_at = time.time()
        dispatch = self.dispatch_store.add(
            chat_id,
            record.session_id,
            context=context or prompt[:200],
            dispatch_type="subagent",
            started_at=started_at,
        )
        task = Task(
            name="subagent",
            type=TaskType.SUBAGENT,
            prompt=prompt,
            chat_id=chat_id,
            acp_model=use_model,
            acp_timeout=acp_timeout,
            mcp_servers=self.mcp.enabled_servers(
                chat_id,
                self._mcp_skills._active_mcp_server_names(chat_id),
            ),
            dispatch_id=dispatch.id,
            cwd=cwd or self._runtime._chat_dir(chat_id),
        )
        plan = self._runtime.plan_manager.create_plan(
            f"subagent-{dispatch.id[:8]}",
            description="Background subagent task",
            chat_id=chat_id,
            tasks=[task],
        )

        wake_id = f"wake-{dispatch.id}"
        self._runtime.wake_queue.enqueue(
            WakeEvent(
                id=wake_id,
                chat_id=chat_id,
                reason="dispatch",
                priority=1,
                scheduled_at=time.time(),
                created_at=time.time(),
                silent=False,
                payload={"dispatch_id": dispatch.id, "notify": True, "retry_after": 5.0},
                ready=False,
            )
        )

        self._runtime.task_engine.start_task(plan.id, task.id)
        return ChatResult(
            reply="Subagent started. I'll report back when it finishes.",
            dispatch_id=dispatch.id,
        )

    def _persist_subagent_result(self, dispatch: Dispatch, result: str) -> Path | None:
        """Write the full subagent result to the chat session dir and update the dispatch.

        Returns the path to the written file, or ``None`` on write failure.
        """
        chat_id = dispatch.chat_id
        chat_dir = self._runtime._chat_dir(chat_id)
        result_dir = chat_dir / "subagent-results"
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"subagent-{dispatch.id}.md"
        try:
            result_path.write_text(result, encoding="utf-8")
        except Exception:
            logger.exception("Failed to write subagent result to %s", result_path)
            return None

        summary = self._extract_summary(result, max_chars=240)
        self.dispatch_store.set_result(
            dispatch.id,
            result,
            summary=summary,
            finished_at=time.time(),
            full_result_path=str(result_path),
        )
        return result_path

    def _complete_subagent_task(self, task: Task) -> None:
        """Store the subagent result, persist the full output, and make the wake ready.

        The dispatch is kept PENDING for normal completions so `continue_turn`
        can consume it. For timeout/cancelled subagents the status is updated to
        TIMEOUT/CANCELLED, but `continue_turn` and `wake` still allow those
        states to proceed. The full result is written to the chat session dir.
        """
        if task.dispatch_id is None or task.chat_id is None:
            return
        dispatch = self.dispatch_store.get(task.dispatch_id)
        if dispatch is None:
            return

        result = task.result or task.log or "(no result)"
        if task.status == TaskStatus.FAILED and not result:
            result = "(subagent failed)"

        status, stop_reason, is_timeout, is_cancelled, is_partial = self._subagent_terminal_state(
            task
        )
        self._persist_subagent_result(dispatch, result)

        # Re-read the dispatch after _persist_subagent_result so summary/full_result_path are fresh.
        dispatch = self.dispatch_store.get(task.dispatch_id) or dispatch
        summary = dispatch.summary or self._extract_summary(result, max_chars=240)

        self.dispatch_store.set_result(
            task.dispatch_id,
            result,
            summary=summary,
            finished_at=time.time(),
            status=status,
            stop_reason=stop_reason,
            cancelled=is_cancelled,
            partial=is_partial,
            timed_out=is_timeout,
            full_result_path=dispatch.full_result_path,
        )

        if is_timeout or is_cancelled:
            dispatch = self.dispatch_store.get(task.dispatch_id) or dispatch
            self._notify_subagent_timeout(
                task, task.chat_id, task.dispatch_id, dispatch, summary, is_timeout
            )

        wake_id = f"wake-{task.dispatch_id}"
        try:
            wake = self._runtime.wake_queue.get(wake_id)
            if wake is not None and wake.payload is not None:
                wake.payload["result"] = result
            self._runtime.wake_queue.ready(wake_id, now=time.time())
        except Exception:
            logger.exception("Failed to mark subagent wake %s ready", wake_id)

    @staticmethod
    def _extract_summary(text: str, max_chars: int = 240) -> str:
        """Return a short, one-line summary of a subagent result.

        Prefers the first markdown heading if present, otherwise the first
        block of text up to ``max_chars``.
        """
        if not text:
            return ""
        text = text.strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        first = lines[0]
        if first.startswith("#"):
            return first.lstrip("#").strip()[:max_chars]
        return text[:max_chars]

    @staticmethod
    def _human_duration(seconds: float) -> str:
        """Return a compact, human-readable duration."""
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"{seconds}s"
        minutes, secs = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m {secs}s"

    def _subagent_terminal_state(
        self, task: Task
    ) -> tuple[DispatchStatus | None, str | None, bool, bool, bool]:
        """Map a finished subagent task to its dispatch status and reason flags.

        Returns ``(dispatch_status, stop_reason, is_timeout, is_cancelled, is_partial)``.
        ``dispatch_status`` is ``None`` for normal completions so the dispatch
        stays PENDING until ``continue_turn`` completes it. ``FAILED`` is set for
        genuine failures so the continuation prompt can report them.
        """
        log = (task.log or "").lower()
        if (
            task.timed_out
            or task.stop_reason == "timeout"
            or (task.status == TaskStatus.FAILED and "timeout" in log)
        ):
            return (
                DispatchStatus.TIMEOUT,
                "timeout",
                True,
                False,
                task.partial or True,
            )
        if (
            task.cancelled
            or task.stop_reason == "cancelled"
            or (task.status == TaskStatus.FAILED and "cancel" in log)
        ):
            return (
                DispatchStatus.CANCELLED,
                "cancelled",
                False,
                True,
                task.partial or True,
            )
        if task.status == TaskStatus.FAILED:
            return (
                DispatchStatus.FAILED,
                "failed",
                False,
                False,
                task.partial or False,
            )
        return (
            None,
            None,
            False,
            False,
            task.partial or False,
        )

    def _notify_subagent_timeout(
        self,
        task: Task,
        chat_id: str,
        dispatch_id: str,
        dispatch: Dispatch,
        summary: str,
        is_timeout: bool,
    ) -> None:
        """Send a proactive user-visible notification for a timeout/cancelled subagent."""
        start = task.started_at if task.started_at is not None else dispatch.started_at
        finished = task.completed_at if task.completed_at is not None else dispatch.finished_at
        if start is None or finished is None:
            duration = "unknown"
        else:
            duration = self._human_duration(finished - start)
        reason = "timed out" if is_timeout else "was cancelled"
        text = f"Subagent {dispatch_id} {reason} after {duration}. Partial summary: {summary}"
        chat_result = ChatResult(
            reply=text,
            dispatch_id=dispatch_id,
            session_id=dispatch.session_id if dispatch else None,
        )
        if self._outbox._outbox_delivery_enabled:
            self._outbox._enqueue_outbox(chat_id, chat_result)
            return
        self._outbox._safe_notifier_send(chat_id, text)

    def subagent_status(self, chat_id: str) -> dict[str, Any]:
        """Return the status of all background subagents for a chat."""
        plans = self._runtime.plan_manager.list_plans(chat_id=chat_id)
        subagents: list[dict[str, Any]] = []
        for plan in plans:
            for task in plan.tasks:
                if task.type != TaskType.SUBAGENT:
                    continue
                dispatch = self.dispatch_store.get(task.dispatch_id) if task.dispatch_id else None
                status = self._runtime._subagent_status_name(task, dispatch)
                summary = self._runtime._subagent_summary(task, dispatch)
                started_at = task.started_at or (dispatch.started_at if dispatch else None)
                finished_at = task.completed_at or (dispatch.finished_at if dispatch else None)
                subagents.append(
                    {
                        "plan_id": plan.id,
                        "task_id": task.id,
                        "dispatch_id": task.dispatch_id,
                        "name": plan.name,
                        "status": status,
                        "continued": dispatch.status == DispatchStatus.COMPLETED if dispatch else False,
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "summary": summary,
                        "prompt_snippet": (task.prompt or "")[:200],
                    }
                )
        return {"chat_id": chat_id, "subagents": subagents}
