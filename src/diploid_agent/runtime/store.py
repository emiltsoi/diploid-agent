"""ChatSessionStore: persistence for the chat registry and session archive."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from diploid_agent.models import ChatState, SessionRecord

logger = logging.getLogger(__name__)

# This set is intentionally keyed by file name, not full path. Durable files are
# expected to live at the root of each session directory, and that convention is
# enforced by _copy_session_dir. This is a known limitation/acceptance.
_CHAT_DURABLE_FILES = {
    "chat_transcript.jsonl",
    "chat_MEMORY.md",
    "chat_PROMOTED.md",
    "chat_self_state.md",
    "chat_body_state.json",
    "hindsight-pending-retain.jsonl",
}


class ChatSessionStore:
    """Persistence for the chat registry, session archive, and chat state."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._store: dict[str, ChatState] = {}

    @property
    def _lock(self):
        return self._runtime._lock

    @property
    def config(self):
        return self._runtime.config

    @property
    def store_path(self) -> Path:
        return self._runtime.store_path

    @property
    def sessions_root(self) -> Path:
        return self._runtime.sessions_root

    @property
    def _plugins(self):
        return self._runtime._plugins

    @property
    def context_builder(self):
        return self._runtime.context_builder

    # ---------------------------------------------------------------- load/save

    def _load_store(self) -> None:
        if not self.store_path.exists():
            return
        for line in self.store_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = SessionRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError):
                continue
            state = self._store.setdefault(record.chat_id, ChatState())
            state.sessions[record.session_number] = record
            state.next_session_number = max(state.next_session_number, record.session_number + 1)

    def _append_record(self, record: SessionRecord) -> None:
        with self._lock:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.store_path, "a") as f:
                f.write(json.dumps(record.to_dict(), default=str) + "\n")

    def _compact_store(self) -> None:
        """Rewrite the store without pruning churn; called after pruning."""
        with self._lock:
            lines = []
            for state in self._store.values():
                for record in state.sessions.values():
                    lines.append(json.dumps(record.to_dict(), default=str) + "\n")
            tmp_path = self.store_path.with_suffix(".jsonl.new")
            tmp_path.write_text("".join(lines))
            tmp_path.replace(self.store_path)

    # ---------------------------------------------------------------- session dirs

    def _chat_dir(self, chat_id: str) -> Path:
        safe = chat_id.replace("/", "_")
        return self.sessions_root / safe

    def _archive_dir(self, chat_id: str, session_number: int) -> Path:
        return self._chat_dir(chat_id) / ".archive" / str(session_number)

    def _durable_file_names(self) -> set[str]:
        names = set(_CHAT_DURABLE_FILES)
        names.update(self._plugins.durable_files())
        return names

    def _copy_session_dir(self, source: Path, target: Path) -> None:
        if source == target:
            return
        durable = self._durable_file_names()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            for item in target.iterdir():
                if item.name in durable or item.name == ".archive":
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        else:
            target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.name in durable or item.name == ".archive":
                continue
            dest = target / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    def _archive_active_session(self, chat_id: str, record: SessionRecord) -> None:
        """Copy the active directory into the archive for `record`."""
        active_dir = self._chat_dir(chat_id)
        archive = self._archive_dir(chat_id, record.session_number)
        if active_dir.exists() and any(active_dir.iterdir()):
            self._copy_session_dir(active_dir, archive)

    def _clear_active_session(self, chat_id: str) -> None:
        active_dir = self._chat_dir(chat_id)
        if not active_dir.exists():
            active_dir.mkdir(parents=True, exist_ok=True)
            return
        durable = self._durable_file_names()
        for item in active_dir.iterdir():
            if item.name in durable or item.name in (".archive", ".snapshots"):
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    # ---------------------------------------------------------------- active state

    def _chat_state(self, chat_id: str) -> ChatState:
        with self._lock:
            return self._store.setdefault(chat_id, ChatState())

    def _active_record(self, chat_id: str) -> SessionRecord | None:
        with self._lock:
            state = self._store.get(chat_id)
            if state is None or not state.sessions:
                return None
            return max(state.sessions.values(), key=lambda r: r.updated_at)

    def _next_session_number(self, chat_id: str) -> int:
        state = self._chat_state(chat_id)
        number = state.next_session_number
        state.next_session_number = number + 1
        return number

    def _generate_label(self, chat_id: str, user_message: str) -> str:
        """Auto-generate a short label from the first user message."""
        return self.context_builder.generate_label(chat_id, user_message)

    # ---------------------------------------------------------------- pruning

    def _prune_chat(self, chat_id: str) -> None:
        """Delete archived sessions older than the prune window."""
        if not self.config.harness.session_prune_enabled:
            return
        state = self._chat_state(chat_id)
        active = self._active_record(chat_id)
        cutoff = time.time() - (self.config.harness.session_prune_days * 86400)
        to_remove: list[int] = []
        for number, record in state.sessions.items():
            if active and number == active.session_number:
                continue
            if record.updated_at >= cutoff:
                continue
            to_remove.append(number)
        for number in to_remove:
            archive = self._archive_dir(chat_id, number)
            if archive.exists():
                shutil.rmtree(archive)
            del state.sessions[number]

    def _prune_and_compact(self, chat_id: str) -> None:
        if self.config.harness.session_prune_enabled:
            self._prune_chat(chat_id)
            self._compact_store()

    def _prune_all(self) -> None:
        if not self.config.harness.session_prune_enabled:
            return
        for chat_id in list(self._store.keys()):
            self._prune_chat(chat_id)
        self._compact_store()
