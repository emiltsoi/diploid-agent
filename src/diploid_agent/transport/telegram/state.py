"""Telegram state mixin for the long-polling transport.

This mixin holds the per-chat placeholder, pending-question, and message
registry helpers used by ``TelegramPoller``. It is not intended to be used
on its own; it expects the host class to provide attributes such as
``state_dir``, ``_message_registry_lock``, ``reply_preview_chars``, and
``_send_message``, ``_edit_message_text``, ``_delete_message``,
``_answer_callback_query``, ``_clear_inline_keyboard``.
"""
from __future__ import annotations

import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from diploid_agent.transport.interactive import (
    AskBlock,
    build_keyboard_remove,
    is_ask_cancel_callback,
    parse_ask_callback_index,
)
from diploid_agent.transport.telegram.models import ChatInput

logger = logging.getLogger("telegram_poll")


class TelegramStateMixin:

    def _chat_dir(self, chat_id: int) -> Path:
        """Return the per-chat session directory, mirroring the harness layout."""
        safe = str(chat_id).replace("/", "_")
        return self.state_dir.parent / safe

    def _message_registry_path(self, chat_id: int) -> Path:
        """Path where the poller records Telegram message_id -> turn mappings."""
        return self._chat_dir(chat_id) / "telegram_messages.jsonl"

    @staticmethod
    def _make_preview(text: str, max_chars: int) -> tuple[str, int]:
        """Return a short preview and the original length."""
        if max_chars <= 0:
            return "", len(text)
        if len(text) <= max_chars:
            return text, len(text)
        from diploid_agent.memory import _trim_to_section

        preview = _trim_to_section(text, max_chars)
        return preview, len(text)

    def _register_message_ids(
        self,
        chat_id: int,
        message_ids: list[int],
        session_number: int,
        turn_number: int,
        text: str,
        kind: str = "reply",
    ) -> None:
        """Record the Telegram message ids for a completed turn.

        This lets a later reply-to reference resolve to a specific turn instead
        of copying the full message text back into the prompt.
        """
        if not message_ids:
            return
        preview, original_length = self._make_preview(text, self.reply_preview_chars)
        path = self._message_registry_path(chat_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            records = []
            now = time.time()
            for message_id in message_ids:
                record = {
                    "message_id": message_id,
                    "chat_id": chat_id,
                    "session_number": session_number,
                    "turn_number": turn_number,
                    "preview": preview,
                    "original_length": original_length,
                    "kind": kind,
                    "timestamp": now,
                }
                records.append(json.dumps(record))
            with self._message_registry_lock, open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(records) + "\n")
        except Exception:
            logger.exception("Failed to register message ids for chat %s", chat_id)

    def _placeholder_state_path(self, chat_id: int) -> Path:
        """Path where the active placeholder for a chat is tracked."""
        return self.state_dir / f"{chat_id}.json"

    def _save_placeholder_state(
        self, chat_id: int, message_id: int | None, thought_id: int | None
    ) -> None:
        """Record the message ids of the in-flight placeholder(s)."""
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            state = {"chat_id": chat_id, "message_id": message_id, "thought_id": thought_id}
            self._placeholder_state_path(chat_id).write_text(json.dumps(state))
        except Exception:
            logger.exception("Failed to save placeholder state for chat %s", chat_id)

    def _remove_placeholder_state(self, chat_id: int) -> None:
        """Remove the placeholder state once the turn has completed."""
        try:
            self._placeholder_state_path(chat_id).unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to remove placeholder state for chat %s", chat_id)

    def _pending_question_path(self, chat_id: int) -> Path:
        """Path where the active question for a chat is tracked."""
        return self.state_dir / f"{chat_id}.ask.json"

    def _save_pending_question(
        self, chat_id: int, ask_block: AskBlock, message_id: int | None
    ) -> None:
        """Persist a pending question so we can map the next button press back to it."""
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "chat_id": chat_id,
                "question": ask_block.question,
                "options": ask_block.options,
                "cancellable": ask_block.cancellable,
                "cancel_label": ask_block.cancel_label,
                "message_id": message_id,
            }
            self._pending_question_path(chat_id).write_text(json.dumps(payload))
        except Exception:
            logger.exception("Failed to save pending question for chat %s", chat_id)

    def _load_pending_question(self, chat_id: int) -> dict[str, Any] | None:
        """Load the pending question for a chat, or None."""
        path = self._pending_question_path(chat_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            options = data.get("options") or []
            if not options:
                return None
            return {
                "question": data.get("question", ""),
                "options": [str(o) for o in options],
                "cancellable": data.get("cancellable", False),
                "cancel_label": data.get("cancel_label", "Cancel"),
                "message_id": data.get("message_id"),
            }
        except (OSError, json.JSONDecodeError):
            return None

    def _remove_pending_question(self, chat_id: int) -> None:
        """Remove the pending question for a chat."""
        try:
            self._pending_question_path(chat_id).unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to remove pending question for chat %s", chat_id)

    def _maybe_answer_pending_question(
        self, chat_input: ChatInput
    ) -> ChatInput | None:
        """If the user is answering a pending question, rewrite the message.

        If the question is cancellable and the user pressed the cancel button,
        remove the keyboard, drop the question, and return ``None`` so no turn
        is started.
        """
        pending = self._load_pending_question(chat_input.chat_id)

        # Callback queries come from inline keyboards. They have no user-facing
        # message, so a cancel can be completely silent and a valid answer is
        # translated back to the option text before being sent to the harness.
        if chat_input.callback_query_id is not None:
            return self._handle_ask_callback(chat_input, pending)

        if pending is None:
            return chat_input

        if pending.get("cancellable") and chat_input.text == pending.get(
            "cancel_label", "Cancel"
        ):
            self._remove_pending_question(chat_input.chat_id)
            try:
                self._send_message(
                    chat_input.chat_id,
                    "Cancelled.",
                    reply_markup=build_keyboard_remove(),
                )
            except Exception:
                logger.exception(
                    "Failed to send cancel confirmation for chat %s", chat_input.chat_id
                )
            # In a private chat the bot can delete the user's button-press
            # message, so the cancel looks like it was swallowed rather than sent.
            self._delete_message(chat_input.chat_id, chat_input.message_id)
            return None

        if chat_input.text not in pending["options"]:
            self._remove_pending_question(chat_input.chat_id)
            return chat_input

        self._remove_pending_question(chat_input.chat_id)
        answer = (
            f'The user answered the question "{pending["question"]}" '
            f"by selecting: {chat_input.text}"
        )
        return ChatInput(
            chat_id=chat_input.chat_id,
            message_id=chat_input.message_id,
            text=answer,
            reply_to=pending["question"],
            reply_to_is_bot=True,
            reply_to_message_id=pending["message_id"],
        )

    def _handle_ask_callback(
        self,
        chat_input: ChatInput,
        pending: dict[str, Any] | None,
    ) -> ChatInput | None:
        """Handle an inline-keyboard button press for a pending ask block.

        Cancels are silent: the question is edited to ``Cancelled.`` and the
        keyboard is removed. Valid answers are translated back to the option
        text and sent to the harness. Unknown/stale callbacks are ignored.
        """
        data = chat_input.text
        self._answer_callback_query(chat_input.callback_query_id)

        if pending is None:
            # Stale callback with no tracked question. Remove the keyboard so
            # the user cannot press it again.
            self._clear_inline_keyboard(chat_input.chat_id, chat_input.message_id)
            return None

        question_message_id = pending.get("message_id")

        if pending.get("cancellable") and is_ask_cancel_callback(data):
            self._remove_pending_question(chat_input.chat_id)
            if question_message_id:
                self._edit_message_text(
                    chat_input.chat_id, question_message_id, "Cancelled."
                )
                self._clear_inline_keyboard(
                    chat_input.chat_id, question_message_id
                )
            return None

        index = parse_ask_callback_index(data)
        if index is None or index < 0 or index >= len(pending["options"]):
            self._remove_pending_question(chat_input.chat_id)
            if question_message_id:
                self._clear_inline_keyboard(
                    chat_input.chat_id, question_message_id
                )
            return None

        selected = pending["options"][index]
        self._remove_pending_question(chat_input.chat_id)
        if question_message_id:
            self._clear_inline_keyboard(
                chat_input.chat_id, question_message_id
            )
        return ChatInput(
            chat_id=chat_input.chat_id,
            message_id=chat_input.message_id,
            text=f'The user answered the question "{pending["question"]}" '
            f"by selecting: {selected}",
            reply_to=pending["question"],
            reply_to_is_bot=True,
            reply_to_message_id=question_message_id,
            callback_query_id=None,
        )

    def _cleanup_orphaned_placeholders(self) -> None:
        """Delete any placeholder messages left over from a previous process."""
        if not self.state_dir.exists():
            return
        for path in self.state_dir.glob("*.json"):
            try:
                state = json.loads(path.read_text())
                chat_id = state.get("chat_id")
                message_id = state.get("message_id")
                thought_id = state.get("thought_id")
                if message_id is not None and chat_id is not None:
                    # Try to update the orphan to a restart notice; if it fails,
                    # delete it instead.
                    try:
                        self._api(
                            "editMessageText",
                            chat_id=chat_id,
                            message_id=message_id,
                            text="Service restarted. The previous reply was interrupted.",
                        )
                    except Exception:  # noqa: BLE001
                        self._delete_message(chat_id, message_id)
                if thought_id is not None and chat_id is not None:
                    self._delete_message(chat_id, thought_id)
            except Exception:
                logger.exception("Failed to clean up placeholder state %s", path)
            finally:
                with contextlib.suppress(OSError):
                    path.unlink()

        for path in self.state_dir.glob("*.ask.json"):
            with contextlib.suppress(OSError):
                path.unlink()
