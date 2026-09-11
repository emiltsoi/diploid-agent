"""Shared memory dataclasses."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class MemoryItem:
    """A single retain item."""

    content: str
    timestamp: str = ""
    document_id: str = ""
    session_number: int = 0
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()
        if not self.document_id:
            self.document_id = f"mem-{uuid.uuid4().hex[:12]}"

    def to_hindsight(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "context": self.context or "",
            "metadata": {k: str(v) for k, v in (self.metadata or {}).items()},
            "document_id": self.document_id,
            "tags": self.tags,
            "update_mode": "replace",
        }


@dataclass
class RecallResult:
    """The result of recalling chat memory for a new session."""

    text: str
    truncated: bool
    memory_path: Path | None
    limit: int
    loaded: int
    total: int
