"""First/follow-up prompt building, reply trimming, and continuation logic."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from diploid_agent.engine import TurnRequest, TurnResult
from diploid_agent.engine.router import ModelRoute
from diploid_agent.memory import RecallResult
from diploid_agent.models import SessionRecord
from diploid_agent.persona_composer import PersonaPrompt
from diploid_agent.plugins.contexts import MemoryTransitionContext

logger = logging.getLogger(__name__)


class RuntimePrompts:
    """First/follow-up prompt building, continuation, reply quoting, and model resolution."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @property
    def config(self) -> Any:
        return self._runtime.config

    @property
    def _lock(self) -> Any:
        return self._runtime._lock

    @property
    def _chat_store(self) -> Any:
        return self._runtime._chat_store

    @property
    def _mcp_skills(self) -> Any:
        return self._runtime._mcp_skills

    @property
    def skills(self) -> Any:
        return self._runtime.skills

    @property
    def context_builder(self) -> Any:
        return self._runtime.context_builder

    @property
    def engine(self) -> Any:
        return self._runtime.engine

    @property
    def _plugins(self) -> Any:
        return self._runtime._plugins

    @property
    def _active_record(self) -> Any:
        return self._runtime._active_record

    @property
    def _memory_manager(self) -> Any:
        return self._runtime._memory_manager

    @property
    def _per_chat_metrics(self) -> dict[str, Any]:
        return self._runtime._per_chat_metrics

    @property
    def _router(self) -> Any:
        return self._runtime._router

    def _build_system_notice(
        self,
        persona: PersonaPrompt,
        recall: RecallResult,
        chat_status: dict[str, Any],
    ) -> str | None:
        """Build a notice when memory is truncated for a first turn prompt."""
        return self.context_builder.build_system_notice(persona, recall, chat_status)

    def _trim_reply_quote_to(self, quote: str, limit: int) -> str:
        """Trim a reply-to quote to a given budget, with a truncation marker."""
        return self.context_builder.trim_reply_quote_to(quote, limit)

    def _trim_reply_quote(self, quote: str) -> str:
        """Trim a reply-to quote to the configured budget, with a truncation marker."""
        return self.context_builder.trim_reply_quote(quote)

    def _telegram_message_registry_path(self, chat_id: str) -> Path:
        return self._chat_store._chat_dir(chat_id) / "telegram_messages.jsonl"

    def _load_telegram_message_registry(self, chat_id: str) -> dict[int, dict[str, Any]]:
        path = self._telegram_message_registry_path(chat_id)
        if not path.exists():
            return {}
        entries: dict[int, dict[str, Any]] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            message_id = entry.get("message_id")
            if message_id is not None:
                entries[message_id] = entry
        return entries

    def _format_user_message(
        self,
        user_message: str,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        chat_id: str | None = None,
    ) -> str:
        """Wrap the user message with a labeled reply-to reference if present."""
        return self.context_builder.format_user_message(
            user_message,
            reply_to=reply_to,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
            chat_id=chat_id,
        )

    def _build_first_prompt(
        self,
        chat_id: str,
        user_message: str,
        *,
        model: str | None = None,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        continuation_anchor: str | None = None,
    ) -> tuple[str, str | None, dict[str, bool]]:
        """Build a first-turn prompt and any memory truncation notice."""
        record = self._active_record(chat_id)
        pctx = self.context_builder.build_first(
            chat_id,
            user_message,
            record,
            model=model,
            reply_to=reply_to,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
            continuation_anchor=continuation_anchor,
        )
        return pctx.prompt, pctx.notice, pctx.memory_flags

    def _follow_up_prompt(
        self,
        user_message: str,
        *,
        chat_id: str,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        continuation_anchor: str | None = None,
    ) -> str:
        """Build a follow-up prompt for an existing session."""
        record = self._active_record(chat_id)
        pctx = self.context_builder.build_follow_up(
            chat_id,
            user_message,
            record,
            reply_to=reply_to,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
            continuation_anchor=continuation_anchor,
        )
        return pctx.prompt

    def _model(self, record: SessionRecord | None) -> str:
        return record.model if record else self.config.engine.model

    def resolve_model(
        self,
        chat_id: str,
        user_message: str,
        record: SessionRecord | None = None,
    ) -> ModelRoute:
        """Resolve the model for a user message, checking budget and lane rules."""
        cumulative_tokens = 0
        if record and record.cumulative_metrics:
            cumulative_tokens = record.cumulative_metrics.get("total_tokens", 0)
        else:
            with self._lock:
                per_chat = self._per_chat_metrics.get(chat_id, {})
                cumulative_tokens = per_chat.get("total_tokens", 0)
        return self._router.resolve(user_message, cumulative_tokens)

    def routing_context(
        self,
        chat_id: str,
        record: SessionRecord | None = None,
    ) -> dict[str, Any]:
        """Return the current routing/budget context for a chat."""
        cumulative_tokens = 0
        if record and record.cumulative_metrics:
            cumulative_tokens = record.cumulative_metrics.get("total_tokens", 0)
        else:
            with self._lock:
                per_chat = self._per_chat_metrics.get(chat_id, {})
                cumulative_tokens = per_chat.get("total_tokens", 0)
        return self._router.budget_status(cumulative_tokens)

    @staticmethod
    def _partial_notice(result: TurnResult, continue_word: str = "Continue") -> str:
        """Return a user-facing notice for an interrupted/partial ACP turn."""
        if result.timed_out or result.stop_reason == "timeout":
            return (
                f"The agent reached the time limit before completing its reply. A partial result is shown. "
                f"Reply `{continue_word}` to keep going, or tell me what to change."
            )
        if result.cancelled or result.stop_reason == "cancelled":
            return (
                f"The agent was stopped before completing its reply. A partial result is shown. "
                f"Reply `{continue_word}` to keep going, or tell me what to change."
            )
        if result.partial:
            return (
                f"The agent reached the time limit before completing its reply. A partial result is shown. "
                f"Reply `{continue_word}` to keep going, or tell me what to change."
            )
        return ""

    def is_continuation_message(self, user_message: str) -> bool:
        """Return True if the user message is a continuation trigger."""
        return self.context_builder.is_continuation_message(user_message)

    def _continuation_anchor(self, record: SessionRecord | None, user_message: str) -> str | None:
        """Return a prompt anchor when resuming an interrupted turn."""
        return self.context_builder.continuation_anchor(record, user_message)

    def _turn_number(self, record: SessionRecord | None) -> int:
        return record.turn_number if record else 0

    def _create_record(
        self,
        chat_id: str,
        session_number: int,
        session_id: str,
        model: str,
        reply: str,
        memory_flags: dict[str, bool],
        *,
        parent: int | None = None,
        label: str | None = None,
    ) -> SessionRecord:
        cwd = self._chat_store._chat_dir(chat_id)
        now = time.time()
        return SessionRecord(
            chat_id=chat_id,
            session_number=session_number,
            session_id=session_id,
            model=model,
            persona=self.config.persona.name,
            cwd=str(cwd),
            created_at=now,
            updated_at=now,
            turn_number=0,
            label=label,
            parent=parent,
            persona_memory_exceeded=memory_flags.get("persona_memory_exceeded", False),
            chat_memory_exceeded=memory_flags.get("chat_memory_exceeded", False),
        )

    def _start_new_session(
        self,
        chat_id: str,
        prompt: str,
        model: str,
        *,
        mcp_servers: list[dict[str, Any]] | None = None,
        skill_names: set[str] | None = None,
        on_chunk: Any | None = None,
        on_update: Any | None = None,
    ) -> tuple[TurnResult, str]:
        """Create a new ACP session and return the prompt result + session id."""
        cwd = self._chat_store._chat_dir(chat_id)
        cwd.mkdir(parents=True, exist_ok=True)
        self.skills.sync_to_chat(
            chat_id, cwd, skill_names or self._mcp_skills._active_skill_names(chat_id)
        )
        request = TurnRequest(
            prompt=prompt,
            cwd=cwd,
            model=model,
            mcp_servers=mcp_servers or self._mcp_skills._active_mcp_servers(chat_id),
            soft_timeout=self.config.engine.soft_timeout,
        )
        result = self.engine.prompt(request, on_chunk=on_chunk, on_update=on_update)
        if result.session_id is None:
            from diploid_agent.acp_client import AcpError

            raise AcpError(
                "acp.create_session",
                {"message": "Engine returned a result without a session id"},
            )
        return result, result.session_id

    def _check_chat_memory_transition(self, chat_id: str, record: SessionRecord) -> str | None:
        """Return a system notice if the chat memory just exceeded its cap."""
        cap = self.config.harness.memory.max_chat_memory_chars
        if not cap:
            return None

        mgr = self._memory_manager(chat_id)
        path = mgr.chat_memory_path
        if not path or not path.exists():
            return None

        total = len(path.read_text())
        previous = record.chat_memory_exceeded
        exceeded = total > cap
        record.chat_memory_exceeded = exceeded

        if previous == exceeded:
            return None

        default_notice: str | None = None
        if not previous and exceeded:
            default_notice = (
                f"Chat memory ({path}) has grown beyond the context budget "
                f"({total} > {cap} characters). Older content is still saved "
                f"but is not loaded. You can ask me to read and prune it."
            )

        ctx = self._plugins.on_chat_memory_transition(
            chat_id,
            MemoryTransitionContext(
                chat_id=chat_id,
                record=record,
                kind="chat",
                path=path,
                total=total,
                cap=cap,
                notice=default_notice,
                suppress_default=False,
            ),
        )
        if ctx.suppress_default:
            if ctx.notice and ctx.notice != default_notice:
                return ctx.notice
            return None
        return ctx.notice or default_notice

    def _check_persona_memory_transition(self, record: SessionRecord) -> str | None:
        """Return a system notice if the persona memory just exceeded its cap."""
        cap = self.config.harness.memory.max_persona_memory_chars
        if not cap:
            return None

        path = self.config.persona.profile_root / self.config.persona.memory_filename
        if not path.exists():
            return None

        total = len(path.read_text())
        previous = record.persona_memory_exceeded
        exceeded = total > cap
        record.persona_memory_exceeded = exceeded

        if previous == exceeded:
            return None

        default_notice: str | None = None
        if not previous and exceeded:
            default_notice = (
                f"Persona memory ({path}) has grown beyond the context budget "
                f"({total} > {cap} characters). Older content is still saved "
                f"but is not loaded. You can ask me to read and prune it."
            )

        chat_id = record.chat_id
        ctx = self._plugins.on_persona_memory_transition(
            chat_id,
            MemoryTransitionContext(
                chat_id=chat_id,
                record=record,
                kind="persona",
                path=path,
                total=total,
                cap=cap,
                notice=default_notice,
                suppress_default=False,
            ),
        )
        if ctx.suppress_default:
            if ctx.notice and ctx.notice != default_notice:
                return ctx.notice
            return None
        return ctx.notice or default_notice

    def get_model(self, chat_id: str) -> str:
        """Return the model currently used for a chat, or the default."""
        with self._lock:
            return self._model(self._active_record(chat_id))
