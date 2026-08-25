"""Auto-continue plugin: resume a chat turn automatically after a stop reason."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from acp_fleet_harness.config import PluginConfig
from acp_fleet_harness.models import WakeEvent
from acp_fleet_harness.plugins.base import StatePlugin, TurnInfo
from acp_fleet_harness.plugins.contexts import TurnStartContext
from acp_fleet_harness.runtime.plugin_runtime import PluginRuntime

logger = logging.getLogger(__name__)


def _join_notices(*parts: str | None) -> str | None:
    """Concatenate non-empty notice strings with a blank line between them."""
    joined = "\n\n".join(p for p in parts if p)
    return joined or None


class AutoContinuePlugin(StatePlugin):
    """Resume a chat turn automatically after a configured stop reason."""

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Path,
        runtime: PluginRuntime | None = None,
    ) -> None:
        super().__init__(config, chat_id, sessions_root, runtime=runtime)
        self._state = self._load_state()
        cfg = self.config.config or {}
        self._delay_seconds = float(cfg.get("delay_seconds", 2.0))
        self._max_attempts = int(cfg.get("max_attempts", 3))
        self._delay_cap_seconds = float(cfg.get("delay_cap_seconds", 60.0))
        self._stop_reasons = set(cfg.get("stop_reasons", ["timeout", "cancelled"]))
        self._per_reason = dict(cfg.get("per_reason", {}))
        self._reason = cfg.get("reason", "auto_continue")

    def _settings(self, stop_reason: str | None) -> tuple[int, float, float]:
        """Return (max_attempts, delay_seconds, delay_cap) for a stop reason."""
        reason_cfg = self._per_reason.get(stop_reason, {})
        max_attempts = int(reason_cfg.get("max_attempts", self._max_attempts))
        delay = float(reason_cfg.get("delay_seconds", self._delay_seconds))
        cap = float(reason_cfg.get("delay_cap_seconds", self._delay_cap_seconds))
        return max_attempts, delay, cap

    def _backoff_delay(self, delay: float, attempt: int, cap: float) -> float:
        """Exponential backoff capped at delay_cap_seconds."""
        return min(delay * (2 ** (attempt - 1)), cap)

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

    def _is_continuation(self, text: str) -> bool:
        if not self._runtime:
            return False
        return self._runtime.is_continuation_message(text)

    def _cancel_pending(self) -> None:
        if not self._runtime:
            return
        now = time.time()
        for event in self._runtime.wake_queue.pending(chat_id=self.chat_id):
            if event.reason != self._reason:
                continue
            if event.leased_until is not None and event.leased_until > now:
                continue
            self._runtime.wake_queue.complete(event.id)

    def before_turn(self, context: TurnStartContext) -> None:
        record = context.record
        if record is None or record.last_stop_reason not in self._stop_reasons:
            return

        if self._is_continuation(context.user_message):
            # Manual continue takes over; cancel any pending auto-continue and
            # drop any deferred partial notice.
            self._cancel_pending()
            self._state.pop("deferred_notice", None)
            self._state.pop("send_deferred", None)
            self._save_state()
            # Attempt count stays as-is because this manual turn is the next attempt.
        else:
            # User changed topic; cancel the auto-continue chain and drop the
            # deferred partial notice. If the user wants the partial result, they
            # can still say Continue; otherwise we should not leak a soft-timeout
            # / stopped notice into a normal reply.
            self._cancel_pending()
            self._state["attempt"] = 0
            self._state.pop("deferred_notice", None)
            self._state.pop("send_deferred", None)
            self._save_state()

    def after_turn(self, turn: TurnInfo) -> None:
        reason = turn.last_stop_reason
        if reason not in self._stop_reasons:
            # The turn finished or was stopped by the user. Send any deferred
            # partial notice that has been pending, then reset the chain.
            if self._state.get("send_deferred") and self._state.get("deferred_notice"):
                turn.notice = _join_notices(self._state.pop("deferred_notice"), turn.notice)
                self._state.pop("send_deferred", None)
                self._save_state()
            elif self._state.get("attempt"):
                self._state["attempt"] = 0
                self._save_state()
            return

        max_attempts, base_delay, delay_cap = self._settings(reason)
        attempt = self._state.get("attempt", 0) + 1
        if attempt > max_attempts:
            logger.info(
                "Auto-continue for %s reached max %d attempts",
                self.chat_id,
                max_attempts,
            )
            # Give up: send the current partial notice along with any older one.
            if turn.partial_notice or self._state.get("deferred_notice"):
                turn.notice = _join_notices(
                    turn.partial_notice,
                    self._state.pop("deferred_notice", None),
                    turn.notice,
                )
                turn.partial_notice = None
                self._state.pop("send_deferred", None)
            self._state["attempt"] = 0
            self._save_state()
            return

        # Suppress the generic partial notice for now; show a working indicator
        # to the user so the auto-continue chain is not a silent black box.
        if turn.partial_notice:
            self._state["deferred_notice"] = turn.partial_notice
            turn.partial_notice = None

        delay = self._backoff_delay(base_delay, attempt, delay_cap)
        turn.notice = _join_notices(
            turn.notice,
            f"Working... (attempt {attempt}/{max_attempts}); continuing automatically in {delay:.0f}s.",
        )

        self._schedule(attempt, max_attempts, delay)

    def _schedule(self, attempt: int, max_attempts: int, delay: float) -> None:
        if not self._runtime:
            return
        event = WakeEvent(
            id="",
            chat_id=self.chat_id,
            reason=self._reason,
            priority=1,
            scheduled_at=time.time() + delay,
            payload={"user_message": "Continue"},
            silent=False,
            created_at=time.time(),
            ready=True,
        )
        self._runtime.wake_queue.enqueue(event)
        self._state["attempt"] = attempt
        self._save_state()
        logger.info(
            "Auto-continue scheduled for %s in %.1fs (attempt %d/%d)",
            self.chat_id,
            delay,
            attempt,
            max_attempts,
        )
