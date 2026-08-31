"""Dispatch records for harness-driven continuation."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any


class DispatchStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class Dispatch:
    id: str
    chat_id: str
    session_id: str
    status: DispatchStatus
    result: str | None = None
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    type: str = "dispatch"
    started_at: float | None = None
    finished_at: float | None = None
    summary: str | None = None
    stop_reason: str | None = None
    cancelled: bool = False
    partial: bool = False
    timed_out: bool = False
    full_result_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dispatch:
        return cls(
            id=data["id"],
            chat_id=data["chat_id"],
            session_id=data["session_id"],
            status=DispatchStatus(data["status"]),
            result=data.get("result"),
            context=data.get("context"),
            metadata=data.get("metadata", {}),
            type=data.get("type", "dispatch"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            summary=data.get("summary"),
            stop_reason=data.get("stop_reason"),
            cancelled=data.get("cancelled", False),
            partial=data.get("partial", False),
            timed_out=data.get("timed_out", False),
            full_result_path=data.get("full_result_path"),
        )


class DispatchStore:
    def __init__(self, path: Path | None = None) -> None:
        self._dispatches: dict[str, Dispatch] = {}
        self._lock = Lock()
        self._path = path
        if path is not None:
            self.load()

    def load(self) -> None:
        """Rehydrate dispatches from the JSONL backing file.

        Missing files and corrupt/malformed lines are skipped silently so the
        store can recover from partial writes or hand-edited files.
        """
        if self._path is None or not self._path.exists():
            return
        with self._lock:
            for line in self._path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    dispatch = Dispatch.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
                self._dispatches[dispatch.id] = dispatch

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(dispatch.to_dict(), default=str) + "\n"
            for dispatch in self._dispatches.values()
        ]
        tmp_path = self._path.with_suffix(self._path.suffix + ".new")
        tmp_path.write_text("".join(lines))
        tmp_path.replace(self._path)

    def add(
        self,
        chat_id: str,
        session_id: str,
        context: str | None = None,
        *,
        dispatch_type: str = "dispatch",
        started_at: float | None = None,
    ) -> Dispatch:
        dispatch = Dispatch(
            id=f"dispatch-{uuid.uuid4().hex[:12]}",
            chat_id=chat_id,
            session_id=session_id,
            status=DispatchStatus.PENDING,
            context=context,
            type=dispatch_type,
            started_at=started_at,
        )
        with self._lock:
            self._dispatches[dispatch.id] = dispatch
            self._save()
        return dispatch

    def get(self, dispatch_id: str) -> Dispatch | None:
        with self._lock:
            return self._dispatches.get(dispatch_id)

    def set_result(
        self,
        dispatch_id: str,
        result: str,
        *,
        summary: str | None = None,
        finished_at: float | None = None,
        status: DispatchStatus | None = None,
        stop_reason: str | None = None,
        cancelled: bool = False,
        partial: bool = False,
        timed_out: bool = False,
        full_result_path: str | None = None,
    ) -> Dispatch | None:
        with self._lock:
            dispatch = self._dispatches.get(dispatch_id)
            if dispatch is None:
                return None
            if dispatch.status in (
                DispatchStatus.PENDING,
                DispatchStatus.TIMEOUT,
                DispatchStatus.CANCELLED,
            ):
                dispatch.result = result
                if summary is not None:
                    dispatch.summary = summary
                if finished_at is not None:
                    dispatch.finished_at = finished_at
            if status is not None:
                dispatch.status = status
            if stop_reason is not None:
                dispatch.stop_reason = stop_reason
            dispatch.cancelled = cancelled
            dispatch.partial = partial
            dispatch.timed_out = timed_out
            if full_result_path is not None:
                dispatch.full_result_path = full_result_path
            self._save()
            return dispatch

    def complete(self, dispatch_id: str, result: str) -> Dispatch | None:
        with self._lock:
            dispatch = self._dispatches.get(dispatch_id)
            if dispatch is None:
                return None
            dispatch.status = DispatchStatus.COMPLETED
            dispatch.result = result
            self._save()
            return dispatch

    def fail(self, dispatch_id: str, result: str) -> Dispatch | None:
        with self._lock:
            dispatch = self._dispatches.get(dispatch_id)
            if dispatch is None:
                return None
            dispatch.status = DispatchStatus.FAILED
            dispatch.result = result
            self._save()
            return dispatch

    def list_by_chat(
        self,
        chat_id: str,
        status: DispatchStatus | None = None,
    ) -> list[Dispatch]:
        with self._lock:
            return [
                d
                for d in self._dispatches.values()
                if d.chat_id == chat_id and (status is None or d.status == status)
            ]
