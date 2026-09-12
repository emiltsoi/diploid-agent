"""Live placeholder/commit state machine for one streamed Telegram turn."""

from __future__ import annotations

import logging
import time
from concurrent.futures import TimeoutError
from typing import TYPE_CHECKING, Any

from diploid_agent.transport.interactive import (
    extract_ask_block,
)
from diploid_agent.transport.telegram.formatting import (
    _HEARTBEAT_INTERVAL,
    _REPLY_PLACEHOLDER,
    _THINKING_PREFIX,
    _build_heartbeat_text,
    _format_thought,
)

if TYPE_CHECKING:
    from diploid_agent.transport.telegram.poller import TelegramPoller

logger = logging.getLogger("telegram_poll")

# How long a user-stopped worker waits for the in-flight /chat request to
# unwind before reporting the partial reply it already streamed.
_STOP_RESULT_WAIT = 15.0


class StreamDisplay:
    """Tracks the live reply/thought placeholders for one streamed turn.

    When ``intermediate_messages`` is enabled and the streamed reply pauses,
    the current placeholder is committed as a real message and a fresh
    placeholder is started below it. This makes tool-call gaps readable as
    separate Telegram messages instead of one confusing, edited block.
    """

    def __init__(
        self,
        *,
        poller: TelegramPoller,
        chat_id: int,
        reply_to_message_id: int | None,
        config: Any,
        message_id: int | None,
        thought_id: int | None,
    ) -> None:
        self.poller = poller
        self.chat_id = chat_id
        self.reply_to_message_id = reply_to_message_id
        self.config = config
        self.message_id = message_id
        self.thought_id = thought_id
        self.last_text_sent = ""
        self.last_thought = ""
        self.last_thought_sent = ""
        self.text = ""
        self.display_text = ""
        self.tail_text = ""
        self.visible = ""
        self.committed_text = ""
        self.committed_display = ""
        self.committed_message_id: int | None = None
        self.committed_raw_ok = True
        self.last_growth_at = self.turn_start_at = time.monotonic()
        self.last_edit_at = self.turn_start_at

    def _send_placeholder(self, text: str) -> int | None:
        return self.poller._send_message(
            self.chat_id,
            text,
            reply_to_message_id=self.reply_to_message_id,
        )

    def _uncommitted_tail(self, full: str) -> str:
        if not full:
            return ""
        if self.committed_display and full.startswith(self.committed_display):
            return full[len(self.committed_display) :]
        # The model somehow backtracked; restart the commit baseline.
        return full

    def _should_commit(self, tail: str, idle: float) -> bool:
        if not self.config.intermediate_messages:
            return False
        if idle < self.config.intermediate_idle:
            return False
        if len(tail) < self.config.intermediate_min_chars:
            return False
        stripped = tail.rstrip()
        if not stripped:
            return False
        return stripped[-1] in ".!?\n"

    def next_wait(self) -> float:
        """Seconds to ask the harness to hold the next /turn long-poll."""
        now = time.monotonic()
        remaining = _HEARTBEAT_INTERVAL - (now - self.last_edit_at)
        idle = now - self.last_growth_at
        # If the current uncommitted tail is a candidate for an
        # intermediate-message split, wake at the configured idle deadline
        # (not just at the heartbeat). This prevents two separate answer
        # blocks separated by a tool-call gap from being glued into one
        # Telegram message.
        tail = self._uncommitted_tail(self.display_text)
        if self._should_commit(tail, idle):
            # Tail is already idle enough; poll very soon to commit it.
            commit_wait = 0.0
        elif (
            self.config.intermediate_messages
            and len(tail) >= self.config.intermediate_min_chars
            and (tail.rstrip() and tail.rstrip()[-1] in ".!?\n")
        ):
            commit_wait = max(0.0, self.config.intermediate_idle - idle)
        else:
            commit_wait = float("inf")
        # Wake for the earlier of the heartbeat deadline and the commit
        # deadline. A 0.5 s floor prevents a tight busy loop when no
        # placeholder can be edited, while still letting us react quickly.
        return min(25.0, max(0.5, min(remaining, commit_wait)))

    def update(self, status: dict[str, Any]) -> None:
        """Apply one /turn status poll to the live placeholder(s)."""
        now = time.monotonic()
        running = status.get("status") == "running"
        if running:
            self.text = status.get("message_text", "")
            self.display_text, _ = extract_ask_block(self.text or "")
            self.display_text = self.display_text.strip()
            # Only the not-yet-committed tail belongs in the live
            # placeholder; earlier chunks already went out as their own
            # committed messages. _uncommitted_tail falls back to the full
            # text when the rolling window dropped the committed prefix.
            self.tail_text = self._uncommitted_tail(self.display_text)
            # Drop the paragraph seam so the new message does not open with
            # blank lines; committed_display still tracks the raw slice.
            self.visible = self.tail_text[:4096].lstrip("\n")
        else:
            self.text = ""
            self.display_text = ""
            self.tail_text = ""
            self.visible = ""
        edited = False

        # Start a reply placeholder the moment text starts arriving, even
        # when thought streaming is still active.
        if self.message_id is None and self.display_text:
            self.message_id = self._send_placeholder(_REPLY_PLACEHOLDER)
            if self.message_id is not None:
                self.poller._save_placeholder_state(
                    self.chat_id, self.message_id, self.thought_id
                )
            self.last_text_sent = _REPLY_PLACEHOLDER

        if self.message_id is not None and self.display_text:
            # If the visible text changed, the model is still writing.
            if self.visible and self.visible != self.last_text_sent:
                self.poller._edit_message_text(self.chat_id, self.message_id, self.visible)
                self.last_text_sent = self.visible
                self.last_growth_at = now
                edited = True
            elif self.visible:
                # No new text: check whether the visible tail we already
                # showed has been sitting idle long enough to be its own
                # message. We use the displayed text (not the raw text with
                # hidden ask blocks) so a trailing ask block does not cause
                # a duplicate commit of the same visible content.
                if self._should_commit(self.tail_text, now - self.last_growth_at):
                    # Freeze the current placeholder as a sent message and
                    # start a fresh one so the rest of the reply can stream
                    # below it. committed_display accumulates every shown
                    # chunk so later tails stay suffixes of display_text.
                    shown = self.tail_text[:4096]
                    self.committed_display = (
                        self.committed_display + shown
                        if self.committed_display
                        and self.display_text.startswith(self.committed_display)
                        else shown
                    )
                    if self.committed_raw_ok and len(self.tail_text) <= 4096:
                        self.committed_text = self.text
                    else:
                        # The displayed commit hit the 4096 cap (or a prior
                        # commit did), so the raw prefix no longer maps to
                        # what was shown; finalization strips by display.
                        self.committed_text = ""
                        self.committed_raw_ok = False
                    self.committed_message_id = self.message_id
                    self.last_text_sent = _REPLY_PLACEHOLDER
                    self.last_growth_at = now
                    self.message_id = self._send_placeholder(_REPLY_PLACEHOLDER)
                    if self.message_id is not None:
                        self.poller._save_placeholder_state(
                            self.chat_id, self.message_id, self.thought_id
                        )
                    edited = True

        if self.thought_id is not None:
            thought = status.get("thought_text", "")
            if thought:
                visible = _format_thought(thought)
                if visible and visible != self.last_thought_sent:
                    self.poller._edit_message_text(self.chat_id, self.thought_id, visible)
                    self.last_thought_sent = visible
                    edited = True
                self.last_thought = thought
        if edited:
            self.last_edit_at = now
        elif now - self.last_edit_at >= _HEARTBEAT_INTERVAL:
            # Nothing new from the model; nudge the placeholder so the user
            # knows the harness is still alive.
            elapsed = now - self.turn_start_at
            if self.message_id is not None:
                base = self.tail_text[:4096].lstrip("\n") or _REPLY_PLACEHOLDER
                heartbeat = _build_heartbeat_text(base, elapsed)
                if heartbeat != self.last_text_sent:
                    self.poller._edit_message_text(self.chat_id, self.message_id, heartbeat)
                    self.last_text_sent = heartbeat
                    edited = True
            if self.thought_id is not None:
                base = (
                    _format_thought(self.last_thought)
                    if self.last_thought
                    else _THINKING_PREFIX
                )
                heartbeat = _build_heartbeat_text(base, elapsed)
                if heartbeat != self.last_thought_sent:
                    self.poller._edit_message_text(self.chat_id, self.thought_id, heartbeat)
                    self.last_thought_sent = heartbeat
                    edited = True
            # Reset the timer even if we had no placeholder to update, so a
            # failed sendMessage cannot turn this loop into a tight poll.
            self.last_edit_at = now

    def await_result(self, chat_future: Any, *, stopped: bool) -> dict[str, Any]:
        """Collect the turn result, tolerating stop and harness errors."""
        try:
            if stopped and not chat_future.done():
                # The user asked to stop; give the harness a short window to
                # unwind the turn, then fall back to whatever we streamed.
                return chat_future.result(timeout=_STOP_RESULT_WAIT)
            return chat_future.result()
        except TimeoutError:
            return {
                "reply": self.display_text or self.text,
                "notice": "Turn stopped by user; the harness did not return a final reply.",
            }
        except Exception:
            logger.exception("Turn failed")
            return {
                "reply": "Sorry, the harness is having trouble. Try again in a moment.",
                "notice": None,
            }

    def _register_and_notice(self, result: dict[str, Any], sent: list[int], reply: str) -> None:
        session_number = result.get("session_number")
        turn_number = result.get("turn_number")
        if sent and session_number is not None and turn_number is not None:
            self.poller._register_message_ids(
                self.chat_id, sent, session_number, turn_number, reply, kind="reply"
            )
        notice = result.get("notice")
        if notice:
            self.poller._send_text(
                self.chat_id,
                f"System: {notice}",
                reply_to_message_id=self.reply_to_message_id,
            )

    def finalize(self, result: dict[str, Any]) -> None:
        """Lay out the final reply once the turn completes."""
        if result.get("continuation", False):
            if (
                self.committed_message_id is not None
                and self.committed_message_id != self.message_id
            ):
                self.poller._delete_message(self.chat_id, self.committed_message_id)
            return

        thought = self.last_thought if self.thought_id is not None else ""
        reply = result.get("reply", "")

        if thought:
            self._finalize_with_thought(thought, reply, result)
        else:
            self._finalize_placeholder(reply, result)

    def _finalize_with_thought(
        self, thought: str, reply: str, result: dict[str, Any]
    ) -> None:
        # A thought was streamed. Delete the live-edited placeholder(s) and
        # any committed intermediate reply, then send the full thought as
        # multi-part Telegram messages, then the full final reply below it.
        # This keeps the reasoning block above the answer and avoids both
        # interleaving and duplicating committed text.
        if self.thought_id is not None:
            self.poller._delete_message(self.chat_id, self.thought_id)
            self.thought_id = None
        if self.message_id is not None:
            self.poller._delete_message(self.chat_id, self.message_id)
            self.message_id = None
        if self.committed_message_id is not None:
            self.poller._delete_message(self.chat_id, self.committed_message_id)
            self.committed_message_id = None
            self.committed_text = ""
            self.committed_display = ""

        self.poller._send_text(
            self.chat_id,
            f"{_THINKING_PREFIX}\n{thought}",
            reply_to_message_id=self.reply_to_message_id,
        )

        sent: list[int] = []
        if reply and reply.strip():
            sent = self.poller._send_text(
                self.chat_id,
                reply,
                reply_to_message_id=self.reply_to_message_id,
            )
        self._register_and_notice(result, sent, reply)

    def _finalize_placeholder(self, reply: str, result: dict[str, Any]) -> None:
        # No thought stream. Use the original placeholder-based finalisation
        # so intermediate-message commits are preserved and only the suffix
        # of the final reply is sent.
        if self.thought_id is not None:
            self.poller._delete_message(self.chat_id, self.thought_id)
            self.thought_id = None

        # The final placeholder is only created after thinking completes, so it
        # is always below the thought block.
        if self.message_id is None:
            self.message_id = self._send_placeholder("...")
            if self.message_id is not None:
                self.poller._save_placeholder_state(
                    self.chat_id, self.message_id, self.thought_id
                )

        # Replace the placeholder with the final reply. If we already committed
        # an earlier chunk as its own message, send only the uncommitted suffix
        # so the user does not see the same text twice.
        display_reply, _ = extract_ask_block(reply)
        display_reply = display_reply.strip()
        if self.committed_text and reply.startswith(self.committed_text):
            # The raw final reply still contains the already-committed text;
            # strip the raw prefix so the suffix (which may include a trailing
            # ask block for the keyboard) is sent below the committed message.
            reply = reply[len(self.committed_text) :].lstrip("\n")
        elif self.committed_display and display_reply.startswith(self.committed_display):
            # The visible prefix was already committed, but the raw reply was
            # transformed (e.g. the ask block was stripped). Send only the
            # visible suffix so the committed message is not duplicated.
            reply = display_reply[len(self.committed_display) :].lstrip("\n")
        if not reply or not reply.strip():
            # If the turn produced no final text, do not leave the placeholder
            # hanging. Delete it and send the notice (if any) as a fresh message.
            if self.message_id is not None:
                self.poller._delete_message(self.chat_id, self.message_id)
            sent = []
        elif self.message_id is not None:
            sent = self.poller._send_text(
                self.chat_id,
                reply,
                first_message_id=self.message_id,
                reply_to_message_id=self.reply_to_message_id,
            )
        else:
            sent = self.poller._send_text(
                self.chat_id,
                reply,
                reply_to_message_id=self.reply_to_message_id,
            )
        self._register_and_notice(result, sent, reply)
