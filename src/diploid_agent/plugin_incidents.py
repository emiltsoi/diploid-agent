"""Durable, append-only record of plugin failures and recovery actions."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class PluginIncident:
    id: str
    ts: float
    plugin: str
    chat_id: str
    phase: str
    error: str
    action: str


class PluginIncidentStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        plugin: str,
        chat_id: str = "",
        phase: str,
        error: str,
        action: str = "",
    ) -> PluginIncident:
        incident = PluginIncident(
            id=uuid.uuid4().hex[:12],
            ts=time.time(),
            plugin=plugin,
            chat_id=chat_id,
            phase=phase,
            error=error,
            action=action,
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(incident), ensure_ascii=False) + "\n")
        return incident

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(line) for line in lines[-limit:]]

    def for_plugin(self, name: str, limit: int = 100) -> list[dict[str, Any]]:
        return [i for i in self.recent(limit) if i["plugin"] == name]
