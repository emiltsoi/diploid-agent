"""Context-pressure decisions for follow-up prompts.

Extracted from ``context/builder.py`` — owns soul-mode selection
(normal/small/full/fresh) and the force-new-session decision, driven by the
proactive next-prompt estimate and the last turn's input ratio.  Token math
lives in ``context/token_estimator.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from diploid_agent.config import Config
from diploid_agent.context.token_estimator import TokenEstimator
from diploid_agent.models import SessionRecord

logger = logging.getLogger(__name__)


class ContextPressure:
    """Decide soul-mode and fresh-session pressure for follow-up prompts."""

    def __init__(
        self,
        config: Config,
        token_estimator: TokenEstimator,
        last_full_soul_turn: dict[str, int] | None = None,
    ) -> None:
        self.config = config
        self._token_estimator = token_estimator
        # Shared with ContextBuilder, which resets the entry on cache reset and
        # records the turn whenever a full soul is injected.
        self._last_full_soul_turn: dict[str, int] = (
            last_full_soul_turn if last_full_soul_turn is not None else {}
        )

    def _context_window_for(self, model: str | None) -> int | None:
        return self._token_estimator.context_window_for(model)

    def _context_pressure(self, record: SessionRecord | None) -> dict[str, Any]:
        return self._token_estimator.context_pressure(record)

    def _estimate_next_prompt_tokens(
        self,
        chat_id: str,
        record: SessionRecord,
        formatted_message: str,
    ) -> dict[str, int]:
        return self._token_estimator.estimate_next_prompt_tokens(record, formatted_message)

    def _soul_mode(
        self,
        chat_id: str,
        record: SessionRecord | None,
        rehydrated: bool,
        formatted_message: str = "",
    ) -> tuple[str, bool]:
        """Decide whether to inject a small soul, full soul, or nothing new.

        Returns a tuple of (soul_mode, force_new_session) where soul_mode is
        one of "normal", "small", "full", or "fresh".  force_new_session is
        True when the context window is so full that we should start a fresh
        ACP subprocess.
        """
        if record is None:
            return "normal", False
        if rehydrated:
            return "full", False

        pressure = self._context_pressure(record)
        context_window = pressure["context_window"]
        input_ratio = pressure["input_ratio"]

        thresholds = self.config.harness
        estimated_ratio = 0.0

        # Proactive sizing: estimate the next prompt and trigger a compact fresh
        # session before the prompt overflows the ACP child context window.
        if context_window:
            estimate = self._estimate_next_prompt_tokens(chat_id, record, formatted_message)
            estimated_ratio = estimate["total"] / context_window
            logger.debug(
                "Proactive prompt estimate for %s: %s (ratio %.3f)",
                chat_id,
                estimate,
                estimated_ratio,
            )
            if estimated_ratio > thresholds.proactive_new_session_threshold:
                return "fresh", True

        # Context pressure must reflect the *current* session's occupancy.
        # `input_ratio` (last turn's input tokens / window) is that signal: the
        # ACP child reports the full prompt it consumed, including accumulated
        # history.  `cumulative_ratio` is lifetime chat usage and never resets,
        # so it must not drive fresh-session decisions.
        if input_ratio > thresholds.reinject_soul_full_threshold:
            return "fresh", True

        last_full = self._last_full_soul_turn.get(chat_id, 0)
        turn_number = record.turn_number or 0
        turns_since = turn_number - last_full

        if context_window and (
            estimated_ratio > thresholds.reinject_soul_threshold
            or input_ratio > thresholds.reinject_soul_input_threshold
        ):
            return "small", False

        if not context_window and turns_since > thresholds.reinject_soul_turns:
            return "small", False

        return "normal", False
