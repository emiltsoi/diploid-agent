"""Turn-retain buffer for the conversational harness.

``TurnRetainBuffer`` batches per-turn user/assistant pairs into retained
documents so the backend's fact extraction sees cross-turn context. The
buffer is persisted to disk so a restart does not lose unflushed turns.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from diploid_agent.memory_models import MemoryItem

logger = logging.getLogger(__name__)


class TurnRetainBuffer:
    """Buffer turn pairs and flush them to the retain backend in bundles.

    A session boundary always flushes first so a bundle never spans two ACP
    sessions. On flush failure the buffer (and its backing file) is kept so
    the next append or a restart retries instead of dropping turns.
    """

    def __init__(
        self,
        path: Path,
        retain: Callable[[list[MemoryItem]], None],
        *,
        bundle_turns: int,
        chat_id: str,
        persona_name: str,
    ) -> None:
        self._path = path
        self._retain = retain
        self._bundle_turns = bundle_turns
        self._chat_id = chat_id
        self._persona_name = persona_name
        self.entries: list[dict[str, Any]] = []

    def load(self) -> None:
        """Reload turn pairs buffered before a restart so they are not lost."""
        path = self._path
        if not path.exists():
            return
        try:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict) and entry.get("content"):
                    self.entries.append(entry)
        except OSError:
            logger.warning("Could not read retain buffer %s", path)

    def write(self) -> None:
        """Persist the pending buffer; remove the file when it is empty."""
        path = self._path
        try:
            if self.entries:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps(e) + "\n" for e in self.entries))
            elif path.exists():
                path.unlink()
        except OSError:
            logger.warning("Could not persist retain buffer %s", path)

    def append(
        self,
        content: str,
        *,
        turn_number: int,
        session_number: int,
        model: str,
    ) -> None:
        """Buffer a turn pair and flush when the bundle size is reached."""
        if self.entries and self.entries[0]["session"] != session_number:
            self.flush()
        self.entries.append(
            {
                "content": content,
                "turn": turn_number,
                "session": session_number,
                "model": model,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        if len(self.entries) >= max(1, self._bundle_turns):
            self.flush()
        else:
            self.write()

    def flush(self) -> None:
        """Retain buffered turn pairs, one document per same-session run.

        Normally the buffer holds a single session's pairs; mixed runs can only
        appear after a flush failure at a session boundary, and are split back
        into per-session documents here.
        """
        while self.entries:
            session = self.entries[0]["session"]
            end = 0
            while end < len(self.entries) and self.entries[end]["session"] == session:
                end += 1
            entries = self.entries[:end]
            turns = [e["turn"] for e in entries]
            bundled = len(entries) > 1
            if bundled:
                document_id = f"turns-{self._chat_id}-{session:06d}-{turns[0]:06d}-{turns[-1]:06d}"
                role = "pair_bundle"
            else:
                document_id = f"turn-{self._chat_id}-{session:06d}-{turns[0]:06d}"
                role = "pair"
            item = MemoryItem(
                content="\n\n---\n\n".join(e["content"] for e in entries),
                timestamp=entries[-1].get("timestamp") or datetime.now(UTC).isoformat(),
                document_id=document_id,
                session_number=session,
                metadata={
                    "role": role,
                    "chat_id": self._chat_id,
                    "persona": self._persona_name,
                    "model": entries[-1].get("model"),
                    "turn": turns[-1],
                    "turns": turns,
                    "session": session,
                },
                tags=[
                    "turn",
                    f"chat:{self._chat_id}",
                    f"session:{session}",
                    f"persona:{self._persona_name}",
                ],
            )
            try:
                self._retain([item])
            except Exception as exc:  # noqa: BLE001
                # Keep the buffer and its backing file so the next record_turn
                # or a restart retries instead of dropping the turns.
                logger.warning(
                    "Retain flush failed; keeping %d buffered turns: %s",
                    len(self.entries),
                    exc,
                )
                self.write()
                return
            del self.entries[:end]
        self.write()
