"""Persistent-memory plugin: promote facts and recall across sessions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from devin_fleet_harness.config import PluginConfig
from devin_fleet_harness.models import ChatResult
from devin_fleet_harness.plugins.base import SleepContext, StatePlugin, TurnInfo
from devin_fleet_harness.plugins.contexts import TurnStartContext
from devin_fleet_harness.runtime.plugin_runtime import PluginRuntime

DEFAULT_STATE: dict[str, Any] = {
    "recall_results": "",
}

MEMORY_QUERY_RE = re.compile(
    r"\b(remember|what did|how did|where did|when did|did we|do we|have we)\b",
    re.IGNORECASE,
)
MEMORY_BLOCK_RE = re.compile(r"```memory\s*\n(.*?)\n```", re.DOTALL)

STOP_WORDS = {
    "the",
    "and",
    "but",
    "for",
    "you",
    "are",
    "did",
    "what",
    "how",
    "where",
    "when",
    "there",
    "they",
    "them",
    "this",
    "that",
    "with",
    "from",
    "have",
    "has",
    "had",
    "was",
    "were",
    "can",
    "will",
    "would",
    "should",
    "about",
    "some",
    "any",
    "all",
    "each",
    "every",
    "not",
    "now",
    "then",
    "too",
    "also",
    "very",
    "just",
    "only",
    "even",
    "than",
    "more",
    "most",
    "much",
    "many",
    "such",
    "which",
    "who",
    "whom",
    "whose",
    "why",
}


def _query_terms(query: str) -> list[str]:
    return [
        w.lower() for w in re.findall(r"\w+", query) if len(w) > 2 and w.lower() not in STOP_WORDS
    ]


def _contains_terms(query: str, text: str) -> bool:
    terms = _query_terms(query)
    if not terms:
        return False
    text_lower = text.lower()
    return all(t in text_lower for t in terms)


def _load_transcript(path: Path, max_pairs: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open() as f:
            entries = [json.loads(line) for line in f if line.strip()]
    except (json.JSONDecodeError, OSError):
        return []
    return entries[-(max_pairs * 2) :]


class PersistentMemoryPlugin(StatePlugin):
    """Promotes durable facts and recalls them when the user asks."""

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Path,
        runtime: PluginRuntime | None = None,
    ) -> None:
        super().__init__(config, chat_id, sessions_root, runtime=runtime)
        self._state: dict[str, Any] = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        state = dict(DEFAULT_STATE)
        path = self.state_path()
        if path and path.exists():
            try:
                loaded = json.loads(path.read_text())
                if isinstance(loaded, dict):
                    state.update(loaded)
            except (json.JSONDecodeError, OSError):
                pass
        return state

    def _save_state(self) -> None:
        path = self.state_path()
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._state, indent=2, default=str))

    def _chat_dir(self) -> Path:
        return self.sessions_root / self.chat_id.replace("/", "_")

    def _is_memory_seeking(self, user_message: str) -> bool:
        if not self.config.config.get("auto_recall", True):
            return False
        return bool(MEMORY_QUERY_RE.search(user_message))

    def _short_term_has_answer(self, user_message: str) -> bool:
        if not self.config.config.get("check_short_term", True):
            return False
        if self._runtime is None:
            return False
        max_pairs = self._runtime.config.harness.memory.short_term_turns
        transcript_path = self._chat_dir() / "chat_transcript.jsonl"
        entries = _load_transcript(transcript_path, max_pairs)
        text = "\n".join(str(e.get("content", "")) for e in entries)
        return _contains_terms(user_message, text)

    def _recall(self, user_message: str) -> str:
        if self._short_term_has_answer(user_message):
            return ""
        if self._runtime is None:
            return ""

        max_tokens = self.config.config.get("recall_max_tokens", 1500)
        tags = self.config.config.get("recall_tags") or None
        result = self._runtime.recall(
            self.chat_id,
            user_message,
            tags=tags,
            max_tokens=max_tokens,
        )
        if not isinstance(result, ChatResult):
            return ""

        reply = result.reply or ""
        if not reply or "No relevant memory found" in reply:
            return ""

        return reply.removeprefix("Memory from previous turns:\n\n")

    def _trim(self, text: str, cap: int | None = None) -> str:
        cap = cap or self.max_prompt_chars
        if cap and len(text) > cap:
            candidate = text[:cap]
            last_blank = candidate.rfind("\n\n")
            if last_blank > 0:
                return text[:last_blank]
            last_space = candidate.rfind(" ")
            if last_space > 0:
                return text[:last_space]
            return candidate
        return text

    def before_turn(self, context: TurnStartContext) -> TurnStartContext | None:
        self._state["recall_results"] = ""

        if not self._is_memory_seeking(context.user_message):
            return None

        results = self._recall(context.user_message)
        if results:
            self._state["recall_results"] = results
            self._save_state()

        return None

    def _promote_fact(self, fact: str) -> None:
        if self._runtime is None:
            return
        self._runtime.promote(self.chat_id, fact.strip())

    def after_turn(self, turn: TurnInfo) -> None:
        if not self.config.config.get("auto_promote", True):
            return

        changed = False
        for match in MEMORY_BLOCK_RE.finditer(turn.reply):
            fact = match.group(1).strip()
            if fact:
                self._promote_fact(fact)
                changed = True
        if changed:
            self._save_state()

    def _sleep_summary(self, context: SleepContext) -> None:
        if not self.config.config.get("auto_summarize_on_sleep", False):
            return
        if self._runtime is None:
            return

        transcript_path = self._chat_dir() / "chat_transcript.jsonl"
        entries = _load_transcript(transcript_path, 100)
        if not entries:
            return

        text = "\n\n".join(
            f"{e.get('role', '').capitalize()}: {e.get('content', '')}" for e in entries
        )
        prompt = (
            "Extract the most important facts, preferences, and decisions from this "
            "conversation as a concise bullet list. Do not invent information. "
            "Do not include pleasantries.\n\n" + text
        )

        from devin_fleet_harness.engine.base import TurnRequest

        model = context.record.model if context.record else None

        def _summarize() -> Any:
            return self._runtime.engine.prompt(
                TurnRequest(prompt=prompt, cwd=self._chat_dir() / ".summarize", model=model)
            )

        result = self._runtime.call_engine_unlocked(_summarize)
        if result and result.reply:
            for line in result.reply.splitlines():
                line = line.strip().lstrip("- ").strip()
                if line:
                    self._promote_fact(line)

    def on_sleeping(self, context: SleepContext) -> None:
        self._sleep_summary(context)

    def prompt_block(self, max_chars: int | None = None) -> str | None:
        results = self._state.get("recall_results", "")
        if not results:
            return None
        block = f"## Persistent memory\n\n{results}"
        cap = max_chars if max_chars is not None else self.max_prompt_chars
        return self._trim(block, cap)

    def event(
        self,
        *,
        event: str | None = None,
        raw_args: str | None = None,
        **params: Any,
    ) -> str:
        if event == "promote":
            fact = (params.get("fact") or raw_args or "").strip()
            if not fact:
                return "Usage: /state persistent_memory promote <fact>"
            self._promote_fact(fact)
            return f"Promoted: {fact}"

        if event == "state":
            return self.prompt_block() or "No persistent memory recalled for this turn."

        if event == "clear":
            self._state = dict(DEFAULT_STATE)
            self._save_state()
            return "Persistent memory state cleared."

        return "Usage: /state persistent_memory promote <fact> | state | clear"
