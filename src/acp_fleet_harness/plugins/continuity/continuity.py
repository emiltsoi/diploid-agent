"""Continuity plugin: wake state, instance identity, pending dispatches."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from acp_fleet_harness.config import PluginConfig
from acp_fleet_harness.models import PartialTurn
from acp_fleet_harness.plugins.base import SleepContext, StatePlugin, TurnInfo, WakeContext
from acp_fleet_harness.plugins.contexts import TurnErrorContext
from acp_fleet_harness.runtime.plugin_runtime import PluginRuntime


class ContinuityPlugin(StatePlugin):
    """Tracks time asleep, last turn, and pending work across wake events."""

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Path,
        runtime: PluginRuntime | None = None,
    ) -> None:
        super().__init__(config, chat_id, sessions_root, runtime=runtime)
        self._state: dict[str, Any] = self._load_state()
        self._pending_partial: PartialTurn | None = None
        self._last_partial_write: float = 0.0
        self._throttle_seconds: float = 0.2

    def state_path(self) -> Path | None:
        if not self.config.state_file:
            return None
        chat_dir = self._chat_dir()
        return chat_dir / self.config.state_file

    def _chat_dir(self) -> Path:
        return self.sessions_root / self.chat_id.replace("/", "_")

    def _load_state(self) -> dict[str, Any]:
        path = self.state_path()
        if path is None or not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_state(self) -> None:
        path = self.state_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._state, indent=2, default=str))

    def _format_duration(self, seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)} seconds"
        if seconds < 3600:
            return f"{int(seconds / 60)} minutes"
        return f"{seconds / 3600:.1f} hours"

    def _format_time(self, when: float) -> str:
        return datetime.fromtimestamp(when, tz=UTC).isoformat()

    def on_waking(self, context: WakeContext) -> None:
        self._state["last_woken_at"] = context.now
        self._state["this_instance_id"] = context.instance_id
        if context.previous_turn_at:
            self._state["time_asleep_seconds"] = context.now - context.previous_turn_at
        if context.record:
            self._state["last_session_number"] = context.record.session_number
            self._state["last_turn_number"] = context.record.turn_number
            self._state["last_stop_reason"] = context.record.last_stop_reason
            self._state["last_turn_at"] = context.record.updated_at

        pending = context.pending_dispatches or []
        self._state["pending_dispatches"] = pending
        self._state["had_pending_dispatches"] = bool(pending)

        previous_instance = self._state.get("this_instance_id")
        self._state["instance_changed"] = previous_instance != context.instance_id

        self._save_state()

    def _active_turn_path(self) -> Path:
        return self._chat_dir() / "chat_active_turn.json"

    def _write_active_turn(self) -> None:
        if self._pending_partial is None:
            return
        partial = self._pending_partial
        path = self._active_turn_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "session_number": partial.session_number,
                    "turn_number": partial.turn_number,
                    "user_message": partial.user_message,
                    "message_text": partial.message_text,
                    "thought_text": partial.thought_text,
                    "updated_at": partial.updated_at,
                },
                indent=2,
                default=str,
            )
        )

    def _remove_active_turn(self) -> None:
        path = self._active_turn_path()
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    def _flush_active_turn(self) -> None:
        if self._pending_partial is None:
            return
        self._write_active_turn()
        self._last_partial_write = time.time()

    def on_partial(self, partial: PartialTurn) -> None:
        self._pending_partial = partial
        now = partial.updated_at or time.time()
        if now - self._last_partial_write >= self._throttle_seconds:
            self._write_active_turn()
            self._last_partial_write = now

    def on_turn_end(self, turn: TurnInfo) -> None:
        self._state["last_turn_at"] = turn.updated_at
        self._state["last_user_message"] = turn.user_message
        self._state["last_assistant_reply"] = turn.reply
        self._state["last_session_number"] = turn.session_number
        self._state["last_turn_number"] = turn.turn_number
        self._state["last_stop_reason"] = turn.last_stop_reason
        self._save_state()
        self._flush_active_turn()
        self._remove_active_turn()
        self._pending_partial = None

    def on_turn_error(self, context: TurnErrorContext) -> None:
        self._flush_active_turn()

    def prompt_block(self, max_chars: int | None = None) -> str | None:
        last_turn_at = self._state.get("last_turn_at")
        if not last_turn_at:
            return None

        lines = ["## Wake state"]

        instance_id = self._state.get("this_instance_id", "unknown")
        instance_started = self._state.get("instance_started_at")
        if instance_started:
            lines.append(
                f"- Instance: {instance_id} (started {self._format_time(instance_started)})"
            )
        else:
            lines.append(f"- Instance: {instance_id}")

        last_session = self._state.get("last_session_number")
        last_turn = self._state.get("last_turn_number")
        last_reason = self._state.get("last_stop_reason")
        if last_session is not None and last_turn is not None:
            lines.append(
                f"- Last turn: session {last_session}, turn {last_turn}, "
                f"stop reason {last_reason or 'unknown'}, at {self._format_time(last_turn_at)}"
            )

        time_asleep = self._state.get("time_asleep_seconds")
        if time_asleep is not None:
            lines.append(f"- You were silent for {self._format_duration(time_asleep)}.")

        pending = self._state.get("pending_dispatches") or []
        if pending:
            lines.append(f"- Pending background work: {len(pending)} dispatch(es)")
            for d in pending[:5]:
                ctx = d.get("context") or "(no context)"
                lines.append(f"  - {d.get('id')}: {ctx[:60]}")
        else:
            lines.append("- Pending background work: none")

        last_user = self._state.get("last_user_message", "")
        last_reply = self._state.get("last_assistant_reply", "")
        if last_user or last_reply:
            lines.append("- You may have been in the middle of:")
            if last_user:
                lines.append(f"  User: {last_user[:120]}")
            if last_reply:
                lines.append(f"  Assistant: {last_reply[:120]}")

        active_path = self._active_turn_path()
        if active_path.exists():
            try:
                active = json.loads(active_path.read_text())
                lines.append("- You may have been about to say:")
                lines.append(f"  {active.get('message_text', '')[:500]}")
                if active.get("thought_text"):
                    lines.append("- Your last thought was:")
                    lines.append(f"  {active.get('thought_text')[:500]}")
            except (json.JSONDecodeError, OSError):
                pass

        block = "\n".join(lines)
        if max_chars is not None and len(block) > max_chars:
            block = block[:max_chars]
        return block

    def on_sleeping(self, context: SleepContext) -> None:
        self._state["last_process_ended_at"] = context.now
        self._state["last_process_ended_reason"] = context.reason
        self._save_state()
