"""Per-chat approval token store for plugin mutations."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class ApprovalStore:
    def __init__(self, path: Path, timeout_seconds: float = 300.0) -> None:
        self._path = path
        self._timeout = timeout_seconds

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _now(self) -> float:
        return time.time()

    def propose(self, operation: str, plugin: dict[str, Any]) -> str:
        data = self._load()
        approvals = data.setdefault("approvals", {})
        token = str(uuid.uuid4())[:8]
        approvals[token] = {
            "operation": operation,
            "plugin": plugin,
            "proposed_at": self._now(),
            "approved_at": None,
        }
        self._save(data)
        return token

    def approve(self, token: str) -> bool:
        data = self._load()
        approvals = data.get("approvals", {})
        entry = approvals.get(token)
        if entry is None:
            return False
        if self._now() - entry["proposed_at"] > self._timeout:
            return False
        entry["approved_at"] = self._now()
        self._save(data)
        return True

    def is_approved(self, token: str) -> bool:
        data = self._load()
        entry = data.get("approvals", {}).get(token)
        if not entry:
            return False
        if self._now() - entry["proposed_at"] > self._timeout:
            return False
        return entry.get("approved_at") is not None

    def get(self, token: str) -> dict[str, Any] | None:
        data = self._load()
        return data.get("approvals", {}).get(token)
