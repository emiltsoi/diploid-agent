"""Reply-to quote formatting for user messages.

Extracted from ``context/builder.py`` — wraps a user message with a labeled
reply-to reference, resolving assistant quotes from the per-chat Telegram
message registry when a message id is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from diploid_agent.config import Config
from diploid_agent.persona_composer import _trim_to_section
from diploid_agent.plugins.contexts import UserMessageContext
from diploid_agent.runtime.store import load_message_registry


class ReplyQuoteFormatter:
    """Format reply-to quotes and user messages for prompt assembly."""

    def __init__(
        self,
        config: Config,
        plugin_manager: Any | None = None,
        chat_store: Any | None = None,
    ) -> None:
        self.config = config
        self.plugin_manager = plugin_manager
        self._chat_store = chat_store

    def trim_reply_quote_to(self, quote: str, limit: int) -> str:
        """Trim a reply-to quote to a given budget, with a truncation marker."""
        if not quote or limit <= 0:
            return ""
        if len(quote) <= limit:
            return quote
        trimmed = _trim_to_section(quote, limit - 30)
        return f"{trimmed}\n\n[... {len(quote) - len(trimmed)} characters truncated ...]"

    def trim_reply_quote(self, quote: str) -> str:
        """Trim a reply-to quote to the configured budget, with a truncation marker."""
        limit = self.config.harness.memory.max_reply_quote_chars
        return self.trim_reply_quote_to(quote, limit)

    def _telegram_message_registry_path(self, chat_id: str) -> Path:
        safe = chat_id.replace("/", "_")
        return (
            Path(self.config.harness.sessions_root).expanduser() / safe / "telegram_messages.jsonl"
        )

    def _load_telegram_message_registry(self, chat_id: str) -> dict[int, dict[str, Any]]:
        store = self._chat_store
        path = (
            store.telegram_message_registry_path(chat_id)
            if store is not None
            else self._telegram_message_registry_path(chat_id)
        )
        return load_message_registry(path)

    def format_user_message(
        self,
        user_message: str,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        chat_id: str | None = None,
    ) -> str:
        """Wrap the user message with a labeled reply-to reference if present.

        When `chat_id` is provided, the `before_format_user_message` hook is
        invoked and plugins can modify the raw or formatted message.
        """
        if chat_id is None or self.plugin_manager is None:
            return self._format_message_impl(
                user_message,
                reply_to=reply_to,
                reply_to_is_bot=reply_to_is_bot,
                reply_to_message_id=reply_to_message_id,
                chat_id=chat_id,
            )

        context = UserMessageContext(
            chat_id=chat_id,
            raw_message=user_message,
            formatted_message=None,
            reply_to=reply_to,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
        )

        def _formatter(ctx: UserMessageContext) -> str:
            return self._format_message_impl(
                ctx.raw_message,
                reply_to=ctx.reply_to,
                reply_to_is_bot=ctx.reply_to_is_bot,
                reply_to_message_id=ctx.reply_to_message_id,
                chat_id=ctx.chat_id,
            )

        context = self.plugin_manager.before_format_user_message(chat_id, context, _formatter)
        return context.formatted_message or context.raw_message

    def _format_message_impl(
        self,
        user_message: str,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        chat_id: str | None = None,
    ) -> str:
        """Apply reply-to quoting to the raw user message."""
        if not reply_to and not reply_to_message_id:
            return user_message

        quote = ""
        label = ""

        if reply_to_message_id and chat_id:
            registry = self._load_telegram_message_registry(chat_id)
            entry = registry.get(reply_to_message_id)
            if entry:
                preview = entry.get("preview", "")
                original_length = entry.get("original_length", len(preview))
                session_number = entry.get("session_number")
                turn_number = entry.get("turn_number")
                label = "[In reply to the assistant's earlier message"
                if session_number is not None and turn_number is not None:
                    label += f" (session {session_number}, turn {turn_number})"
                label += ":]"
                if preview:
                    quote = preview
                    if original_length > len(preview):
                        quote += (
                            f"\n\n[... {original_length - len(preview)} characters truncated ...]"
                        )

        if not quote and reply_to:
            if reply_to_is_bot is True:
                limit = self.config.harness.memory.max_bot_reply_quote_chars
                label = "[In reply to the assistant's earlier message:]"
            elif reply_to_is_bot is False:
                limit = self.config.harness.memory.max_reply_quote_chars
                label = "[In reply to your earlier message:]"
            else:
                limit = self.config.harness.memory.max_reply_quote_chars
                label = "[In reply to an earlier message:]"
            quote = self.trim_reply_quote_to(reply_to.strip(), limit)

        if not quote and not reply_to_message_id:
            return user_message

        if quote:
            return f"{label}\n{quote}\n\n[Your new message:]\n{user_message}"
        return f"{label}\n\n[Your new message:]\n{user_message}"
