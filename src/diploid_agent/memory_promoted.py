"""Curated memory pockets: the per-chat promoted file and persona memory.

``PromotedMemory`` owns the user-curated ``chat_PROMOTED.md`` pocket (always
loaded compact so the compactor cannot drop it) and appends to the persona's
``MEMORY.md`` — including Hindsight indexing when that backend is active.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from diploid_agent.memory_backends import (
    HindsightMemoryBackend,
    MemoryBackend,
    _trim_to_section,
)
from diploid_agent.memory_models import MemoryItem


class PromotedMemory:
    """User-curated promoted pocket plus persona memory reads/writes."""

    def __init__(
        self,
        config: Any,  # MemoryConfig
        persona: Any,  # PersonaConfig
        sessions_root: Path,
        chat_id: str,
        backend_fn: Callable[[], MemoryBackend],
    ) -> None:
        self._config = config
        self._persona = persona
        self._sessions_root = sessions_root
        self._chat_id = chat_id
        # Late-bound: callers may swap the backend after construction.
        self._backend_fn = backend_fn

    @staticmethod
    def normalize_line(line: str) -> str:
        """Normalize a promoted line for duplicate comparison."""
        text = line.strip()
        while text.startswith("-"):
            text = text[1:].lstrip()
        return " ".join(text.split())

    @property
    def promoted_path(self) -> Path:
        """Path to the user-curated promoted memory file for this chat."""
        safe = self._chat_id.replace("/", "_")
        return self._sessions_root / safe / "chat_PROMOTED.md"

    @property
    def persona_path(self) -> Path:
        """Path to the persona's memory file."""
        return self._persona.profile_root / self._persona.memory_filename

    def persona_memory(self, max_chars: int | None = None) -> dict[str, Any]:
        """Load and optionally cap the persona's MEMORY.md for the prompt."""
        path = self.persona_path
        text = ""
        total = 0
        loaded = 0
        limit = max_chars or 0
        truncated = False

        if path.exists():
            raw = path.read_text()
            total = len(raw)
            if max_chars and total > max_chars:
                text = _trim_to_section(raw, max_chars)
                loaded = len(text)
                truncated = True
            else:
                text = raw
                loaded = total

        return {
            "text": text,
            "path": path if total > 0 else None,
            "truncated": truncated,
            "limit": limit,
            "loaded": loaded,
            "total": total,
        }

    def promoted_memory(self, max_chars: int | None = None) -> dict[str, Any]:
        """Load the promoted memory pocket, always capped tightly."""
        cap = max_chars or 1000
        path = self.promoted_path
        text = ""
        total = 0
        loaded = 0
        truncated = False

        if path.exists():
            raw = path.read_text()
            total = len(raw)
            if total > cap:
                text = _trim_to_section(raw, cap)
                loaded = len(text)
                truncated = True
            else:
                text = raw
                loaded = total

        return {
            "text": text,
            "path": path if total > 0 else None,
            "truncated": truncated,
            "limit": cap,
            "loaded": loaded,
            "total": total,
        }

    def tidy(self) -> None:
        """Cap the promoted pocket and drop duplicate entries."""
        max_lines = getattr(self._config, "max_promoted_lines", 0)
        path = self.promoted_path
        if not path.exists():
            return

        raw = path.read_text(encoding="utf-8")
        lines = [line for line in raw.splitlines() if line.strip()]
        seen: set[str] = set()
        deduped: list[str] = []
        for line in lines:
            key = self.normalize_line(line)
            if key and key not in seen:
                seen.add(key)
                deduped.append(line)
        if max_lines and len(deduped) > max_lines:
            deduped = deduped[-max_lines:]
        if deduped != lines:
            path.write_text("\n".join(deduped) + "\n", encoding="utf-8")

    def append(self, content: str) -> None:
        """Append a user-promoted fact to the curated pocket file."""
        path = self.promoted_path
        path.parent.mkdir(parents=True, exist_ok=True)
        key = self.normalize_line(content)
        if path.exists():
            existing = {
                self.normalize_line(line) for line in path.read_text(encoding="utf-8").splitlines()
            }
            if key in existing:
                return
        with path.open("a", encoding="utf-8") as f:
            f.write(f"- {content.strip()}\n")
        self.tidy()

    def should_auto_promote(self, content: str, tags: list[str]) -> bool:
        """Return True when a retained fact looks durable enough for the promoted pocket."""
        if not getattr(self._config, "auto_promote_enabled", True):
            return False
        if "promoted" in tags or "no-promote" in tags:
            return False

        lower_tags = {t.lower() for t in tags}
        auto_tags = set(getattr(self._config, "auto_promote_tags", []))
        if lower_tags & {t.lower() for t in auto_tags}:
            return True

        lower = content.lower()
        triggers = getattr(self._config, "auto_promote_triggers", [])
        for trigger in triggers:
            if trigger.lower() in lower:
                return True

        return False

    def promote_to_persona(self, fact: str) -> None:
        """Append a fact to the persona's MEMORY.md and, for Hindsight, index it.

        Memory files are not mechanically pruned. The agent has agency to edit
        them using its own file tools when the system notice says they exceed
        the prompt budget.
        """
        path = self.persona_path
        path.parent.mkdir(parents=True, exist_ok=True)
        new_block = f"- {fact.strip()}\n"
        with open(path, "a") as f:
            f.write(new_block)

        backend = self._backend_fn()
        if isinstance(backend, HindsightMemoryBackend):
            item = MemoryItem(
                content=fact.strip(),
                timestamp=datetime.now(UTC).isoformat(),
                document_id=f"promote-{self._chat_id}-{uuid.uuid4().hex[:12]}",
                metadata={
                    "chat_id": self._chat_id,
                    "persona": self._persona.name,
                    "kind": "promoted",
                },
                tags=[
                    "memory",
                    "persona",
                    "promoted",
                    f"chat:{self._chat_id}",
                    f"persona:{self._persona.name}",
                ],
            )
            backend.retain([item])
