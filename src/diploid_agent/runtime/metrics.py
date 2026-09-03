"""Runtime metrics collection, health, and prometheus formatting helpers."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from diploid_agent.models import SessionRecord

logger = logging.getLogger(__name__)


class RuntimeMetrics:
    """Owns per-chat and global metrics, health probes, and prometheus formatting."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._per_chat_metrics: dict[str, dict[str, Any]] = {}
        self._global_metrics: dict[str, Any] = {
            "turns": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "latency_seconds": 0.0,
        }
        self._recent_turns: deque[dict[str, Any]] = deque(
            maxlen=runtime.config.harness.metrics.max_recent_turns
        )

    @property
    def config(self) -> Any:
        return self._runtime.config

    @property
    def _lock(self) -> Any:
        return self._runtime._lock

    @property
    def _store(self) -> dict[str, Any]:
        return self._runtime._store

    @property
    def metrics(self) -> Any:
        return self._runtime.metrics

    @property
    def context_builder(self) -> Any:
        return self._runtime.context_builder

    @property
    def engine(self) -> Any:
        return self._runtime.engine

    @property
    def notifier(self) -> Any:
        return self._runtime.notifier

    @property
    def _plugins(self) -> Any:
        return self._runtime._plugins

    @property
    def instance_started_at(self) -> float:
        return self._runtime.instance_started_at

    def _rehydrate_metrics(self) -> None:
        """Seed per-chat and global metrics from the on-disk session store."""
        with self._lock:
            for chat_state in self._store.values():
                record = max(chat_state.sessions.values(), key=lambda r: r.updated_at, default=None)
                if not record or not record.cumulative_metrics:
                    continue
                cumulative = record.cumulative_metrics
                self._per_chat_metrics[record.chat_id] = {
                    "turns": cumulative.get("turns", 0),
                    "input_tokens": cumulative.get("input_tokens", 0),
                    "output_tokens": cumulative.get("output_tokens", 0),
                    "total_tokens": cumulative.get("total_tokens", 0),
                    "cached_tokens": cumulative.get("cached_tokens", 0),
                    "latency_seconds": cumulative.get("latency_seconds", 0.0),
                    "cumulative": cumulative,
                    "last_turn": record.last_turn_metrics,
                }
                self._global_metrics["turns"] += cumulative.get("turns", 0)
                self._global_metrics["input_tokens"] += cumulative.get("input_tokens", 0)
                self._global_metrics["output_tokens"] += cumulative.get("output_tokens", 0)
                self._global_metrics["total_tokens"] += cumulative.get("total_tokens", 0)
                self._global_metrics["cached_tokens"] += cumulative.get("cached_tokens", 0)
                self._global_metrics["latency_seconds"] += cumulative.get("latency_seconds", 0.0)

    def _record_turn_metrics(
        self,
        chat_id: str,
        turn_number: int,
        model: str,
        usage: dict[str, Any] | None,
        latency_seconds: float,
        prompt_chars: int = 0,
    ) -> dict[str, Any]:
        """Record per-turn metrics and update running totals."""
        usage = usage or {}
        turn_metrics = {
            "chat_id": chat_id,
            "turn_number": turn_number,
            "model": model,
            "input_tokens": usage.get("inputTokens") or usage.get("input_tokens", 0),
            "output_tokens": usage.get("outputTokens") or usage.get("output_tokens", 0),
            "total_tokens": usage.get("totalTokens") or usage.get("total_tokens", 0),
            "cached_tokens": usage.get("cachedReadTokens") or usage.get("cached_tokens", 0),
            "prompt_chars": prompt_chars,
            "latency_seconds": round(latency_seconds, 3),
        }

        with self._lock:
            per_chat = self._per_chat_metrics.setdefault(
                chat_id,
                {
                    "turns": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cached_tokens": 0,
                    "latency_seconds": 0.0,
                    "cumulative": {},
                },
            )
            per_chat["turns"] += 1
            per_chat["input_tokens"] += turn_metrics["input_tokens"]
            per_chat["output_tokens"] += turn_metrics["output_tokens"]
            per_chat["total_tokens"] += turn_metrics["total_tokens"]
            per_chat["cached_tokens"] += turn_metrics["cached_tokens"]
            per_chat["latency_seconds"] += turn_metrics["latency_seconds"]
            per_chat["cumulative"] = {
                "turns": per_chat["turns"],
                "input_tokens": per_chat["input_tokens"],
                "output_tokens": per_chat["output_tokens"],
                "total_tokens": per_chat["total_tokens"],
                "cached_tokens": per_chat["cached_tokens"],
                "latency_seconds": round(per_chat["latency_seconds"], 3),
            }

            self._global_metrics["turns"] += 1
            self._global_metrics["input_tokens"] += turn_metrics["input_tokens"]
            self._global_metrics["output_tokens"] += turn_metrics["output_tokens"]
            self._global_metrics["total_tokens"] += turn_metrics["total_tokens"]
            self._global_metrics["cached_tokens"] += turn_metrics["cached_tokens"]
            self._global_metrics["latency_seconds"] += turn_metrics["latency_seconds"]

            self._recent_turns.append(turn_metrics)

        self.metrics.inc("turns_total")
        self.metrics.inc("tokens_total", value=turn_metrics["input_tokens"], kind="input")
        self.metrics.inc("tokens_total", value=turn_metrics["output_tokens"], kind="output")
        self.metrics.inc("tokens_total", value=turn_metrics["total_tokens"], kind="total")
        self.metrics.inc("tokens_total", value=turn_metrics["cached_tokens"], kind="cached")
        self.metrics.inc("turn_latency_seconds_total", value=turn_metrics["latency_seconds"])

        return dict(turn_metrics)

    def _metrics_context_for_prompt(self, chat_id: str, compact: bool = False) -> str | None:
        """Return a metrics notice for injection into the LLM prompt."""
        return self.context_builder.metrics_context_for_prompt(chat_id, compact=compact)

    def get_metrics(self, chat_id: str | None = None) -> dict[str, Any]:
        """Return cumulative metrics for a chat or globally."""
        with self._lock:
            if chat_id is None:
                return {
                    "global": dict(self._global_metrics),
                    "recent_turns": list(self._recent_turns),
                }
            per_chat = self._per_chat_metrics.get(chat_id, {})
            return {
                "chat_id": chat_id,
                "cumulative": per_chat.get("cumulative", {}),
                "last_turn": per_chat.get("last_turn"),
            }

    def get_prometheus_metrics(self) -> str:
        """Return metrics in Prometheus exposition format."""
        return self.metrics.render()

    def health(self) -> dict[str, Any]:
        """Return the current health of the runtime and its dependencies."""
        components: dict[str, Any] = {}

        acp_healthy = False
        try:
            acp_healthy = bool(self.engine.health())
        except Exception as exc:  # noqa: BLE001
            logger.debug("ACP health check failed: %s", exc)
        components["acp"] = {
            "status": "ok" if acp_healthy else "error",
            "healthy": acp_healthy,
        }

        hindsight_healthy = True
        if self.config.harness.memory.backend == "hindsight":
            hindsight_healthy = self._hindsight_health()
        components["hindsight"] = {
            "status": "ok" if hindsight_healthy else "error",
            "healthy": hindsight_healthy,
        }

        telegram_healthy = False
        try:
            telegram_healthy = bool(self.notifier.health())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Telegram health check failed: %s", exc)
        components["telegram"] = {
            "status": "ok" if telegram_healthy else "error",
            "healthy": telegram_healthy,
        }

        plugin_health = self._plugins.plugin_health("0")
        plugins_healthy = all(p["healthy"] for p in plugin_health)
        components["plugins"] = {
            "status": "ok" if plugins_healthy else "error",
            "healthy": plugins_healthy,
            "details": plugin_health,
        }

        overall = "ok" if all(c["healthy"] for c in components.values()) else "degraded"
        return {
            "status": overall,
            "uptime_seconds": round(time.time() - self.instance_started_at, 3),
            "components": components,
        }

    def _hindsight_health(self) -> bool:
        """Probe the Hindsight backend health endpoint."""
        import urllib.parse

        base = self.config.harness.memory.hindsight.base_url
        if not base:
            return True
        url = urllib.parse.urljoin(base, "/health")
        try:
            import httpx

            resp = httpx.get(url, timeout=5.0)
            return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001
            logger.debug("Hindsight health check failed: %s", exc)
            return False

    def _context_usage(self, record: SessionRecord) -> dict[str, Any]:
        """Return context-window and prompt-budget usage for a chat record."""
        context_window: int | None = None
        try:
            context_window_fn = getattr(self.engine, "model_context_window", None)
            if context_window_fn is not None:
                context_window = context_window_fn(record.model)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to resolve context window for %s: %s", record.model, exc)

        last_turn = record.last_turn_metrics or {}
        cumulative = record.cumulative_metrics or {}

        def _enrich(turn: dict[str, Any]) -> dict[str, Any]:
            enriched = dict(turn)
            if context_window:
                input_tokens = turn.get("input_tokens", 0) or 0
                total_tokens = turn.get("total_tokens", 0) or 0
                enriched["input_percent"] = round(input_tokens / context_window * 100, 2)
                enriched["total_percent"] = round(total_tokens / context_window * 100, 2)
                enriched["available_tokens"] = max(0, context_window - input_tokens)
            return enriched

        return {
            "model": record.model,
            "context_window": context_window,
            "last_turn": _enrich(last_turn),
            "cumulative": cumulative,
            "memory_budgets": {
                "max_chat_memory_chars": self.config.harness.memory.max_chat_memory_chars,
                "max_persona_memory_chars": self.config.harness.memory.max_persona_memory_chars,
                "max_short_term_chars": self.config.harness.memory.max_short_term_chars,
                "max_reply_quote_chars": self.config.harness.memory.max_reply_quote_chars,
                "hindsight_max_recall_tokens": self.config.harness.memory.hindsight.max_recall_tokens,
            },
            "memory_exceeded": {
                "chat_memory_exceeded": record.chat_memory_exceeded,
                "persona_memory_exceeded": record.persona_memory_exceeded,
            },
        }
