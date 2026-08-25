"""Lane-based model routing and conversation budget guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from acp_fleet_harness.config import Config


@dataclass
class ModelRoute:
    """Result of resolving a model for a turn."""

    model: str
    notice: str | None = None
    budget_exceeded: bool = False


class ModelRouter:
    """Choose a model for a user message based on lane rules and budget."""

    def __init__(self, config: Config) -> None:
        self._default_model = config.engine.model
        self._routing = config.harness.routing

    @property
    def enabled(self) -> bool:
        return self._routing.enabled

    def resolve(
        self,
        user_message: str,
        cumulative_tokens: int = 0,
    ) -> ModelRoute:
        """Return the model, any warning notice, and whether the budget is exceeded.

        If the conversation budget is exceeded and ``hard_cap`` is enabled, a
        ``BudgetExceeded`` exception is raised. If ``hard_cap`` is disabled, a
        notice is returned but a model is still chosen.
        """
        if not self.enabled:
            return ModelRoute(model=self._default_model)

        budget = self._routing.budget
        if budget.enabled and cumulative_tokens >= budget.max_total_tokens:
            notice = (
                f"This conversation has reached its token budget "
                f"({cumulative_tokens:,} / {budget.max_total_tokens:,}). "
                f"Start a new session with /new to continue."
            )
            return ModelRoute(
                model=self._routing.fallback_model or self._default_model,
                notice=notice,
                budget_exceeded=True,
            )

        lane = self._detect_lane(user_message)
        model = self._routing.lanes.get(lane, self._routing.fallback_model or self._default_model)

        notice: str | None = None
        if budget.enabled and cumulative_tokens > 0:
            ratio = cumulative_tokens / budget.max_total_tokens
            if ratio >= budget.warning_threshold and ratio < 1.0:
                remaining = budget.max_total_tokens - cumulative_tokens
                notice = (
                    f"Approaching conversation token budget: "
                    f"{cumulative_tokens:,} / {budget.max_total_tokens:,} "
                    f"({remaining:,} remaining)."
                )

        return ModelRoute(model=model, notice=notice)

    def _detect_lane(self, user_message: str) -> str | None:
        """Return the first matching lane for the message, or None."""
        text = user_message.lower()
        for lane, keywords in self._routing.lane_keywords.items():
            if any(kw.lower() in text for kw in keywords):
                return lane
        return None

    def remaining_budget(self, cumulative_tokens: int) -> int:
        """Return the number of tokens remaining before the budget is exhausted."""
        if not self._routing.budget.enabled:
            return -1
        return max(0, self._routing.budget.max_total_tokens - cumulative_tokens)

    def budget_status(self, cumulative_tokens: int) -> dict[str, Any]:
        """Return a serializable budget summary."""
        budget = self._routing.budget
        if not budget.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "max_total_tokens": budget.max_total_tokens,
            "used_tokens": cumulative_tokens,
            "remaining_tokens": self.remaining_budget(cumulative_tokens),
            "exceeded": cumulative_tokens >= budget.max_total_tokens,
        }
