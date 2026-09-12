"""Token estimation and context-window pressure math for ContextBuilder.

Extracted from ``context/builder.py`` — owns the characters-per-token table,
live calibration from turn metrics, context-window resolution, and the
proactive next-prompt estimate used for context-pressure decisions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from diploid_agent.config import Config
from diploid_agent.models import SessionRecord
from diploid_agent.persona_composer import identity_anchor


class TokenEstimator:
    """Estimate prompt token footprints and context-window pressure."""

    # Hand-maintained characters-per-token table for known models.  If the model
    # is not listed, the conservative 4:1 fallback is used.
    _CHARS_PER_TOKEN: ClassVar[dict[str, float]] = {
        "swe-1-7": 3.5,
        "claude-sonnet-4-20250514": 4.0,
        "claude-sonnet-4": 4.0,
        "claude-opus-4": 4.0,
        "gpt-4o": 4.0,
        "gpt-4o-mini": 4.0,
    }

    def __init__(
        self,
        config: Config,
        context_window_fn: Callable[[str], int | None] | None = None,
    ) -> None:
        self.config = config
        self.context_window_fn = context_window_fn

    def chars_per_token(self, model: str | None, record: SessionRecord | None = None) -> float:
        """Return a character-to-token ratio for `model`.

        Prefer live calibration from the last turn's prompt length and token
        count, then fall back to the hand-maintained table, then 4:1.
        """
        if not model:
            return 4.0

        if self.config.harness.proactive_calibration_enabled and record is not None:
            min_chars = self.config.harness.proactive_calibration_min_prompt_chars
            # The first turn of a session is the cleanest sample: the prompt we
            # sent dominates input_tokens before session history accumulates.
            # Later turns are only trusted when the ratio still lands in range —
            # on established sessions `input_tokens` includes the accumulated
            # session history, so prompt_chars / input_tokens collapses toward
            # zero and the next-prompt estimate explodes.
            for metrics in (record.first_turn_metrics, record.last_turn_metrics):
                if not metrics:
                    continue
                prompt_chars = metrics.get("prompt_chars") or 0
                input_tokens = metrics.get("input_tokens") or 0
                if prompt_chars < min_chars or input_tokens <= 0:
                    continue
                ratio = prompt_chars / input_tokens
                if 1.0 <= ratio <= 10.0:
                    return ratio

        model_lower = model.lower()
        for name, ratio in self._CHARS_PER_TOKEN.items():
            if name in model_lower:
                return ratio
        return 4.0

    def context_window_for(self, model: str | None) -> int | None:
        """Resolve the context window for a model, if known."""
        if self.context_window_fn is not None and model:
            return self.context_window_fn(model)
        return self.config.engine.context_window

    def context_pressure(self, record: SessionRecord | None) -> dict[str, Any]:
        """Return context-window pressure metrics for the current record.

        Uses the cumulative token count as the primary pressure signal and the
        last turn's input tokens as the secondary signal.  When the context
        window size is unknown, both percentages are zero.
        """
        context_window = self.context_window_for(record.model if record else None)
        cumulative = record.cumulative_metrics if record else {}
        last_turn = record.last_turn_metrics if record else {}

        result: dict[str, Any] = {
            "context_window": context_window,
            "cumulative_ratio": 0.0,
            "input_ratio": 0.0,
        }
        if not context_window:
            return result

        total = cumulative.get("total_tokens", 0) or 0
        input_tokens = last_turn.get("input_tokens", 0) or 0
        result["cumulative_ratio"] = round(total / context_window, 4)
        result["input_ratio"] = round(input_tokens / context_window, 4)
        return result

    def estimate_next_prompt_tokens(
        self,
        record: SessionRecord,
        formatted_message: str,
    ) -> dict[str, int]:
        """Estimate token footprint of the next prompt for proactive sizing.

        Uses the last turn's actual token usage as the primary signal and a
        hand-maintained characters-per-token table for the model.  Returns a
        dict with the components so callers can log them.
        """
        last_turn = record.last_turn_metrics or {}
        last_input = last_turn.get("input_tokens", 0) or 0
        last_output = last_turn.get("output_tokens", 0) or 0
        last_total = last_input + last_output

        chars_per_token = self.chars_per_token(record.model, record)

        memory_cfg = self.config.harness.memory
        short_term_estimate = int((memory_cfg.max_short_term_chars or 0) / chars_per_token)

        # Cheap fresh-soul budget: identity anchor + cheap soul slots.
        anchor_len = len(identity_anchor(self.config.persona))
        cheap_soul_estimate = (
            int(anchor_len / chars_per_token) + self.config.harness.proactive_soul_token_budget
        )

        user_estimate = int(len(formatted_message) / chars_per_token)

        buffer_factor = self.config.harness.proactive_input_buffer_factor
        buffered_turn = int(last_total * buffer_factor)

        return {
            "chars_per_token": chars_per_token,
            "last_total": last_total,
            "buffered_turn": buffered_turn,
            "soul": cheap_soul_estimate,
            "user": user_estimate,
            "short_term": short_term_estimate,
            "total": buffered_turn + cheap_soul_estimate + user_estimate + short_term_estimate,
        }

    def estimate_prompt_tokens(self, prompt: str, record: SessionRecord | None) -> int:
        """Return a rough token estimate for a prompt string."""
        return int(len(prompt) / self.chars_per_token(record.model if record else None, record))
