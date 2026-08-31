"""Interactive question/answer helpers for Telegram."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

ASK_FENCE_RE = re.compile(r"^```ask\s*\n([\s\S]*?)\n```\s*$", re.MULTILINE)

ASK_CALLBACK_PREFIX = "ask_"
ASK_CANCEL_CALLBACK_DATA = f"{ASK_CALLBACK_PREFIX}cancel"


@dataclass(frozen=True)
class AskBlock:
    """A question, its selectable options, and an optional cancel button."""

    question: str
    options: list[str]
    cancellable: bool = False
    cancel_label: str = "Cancel"


def build_reply_keyboard(
    options: list[str], cancel: str | None = None
) -> dict[str, Any]:
    """Return a Telegram ReplyKeyboardMarkup for a list of option strings.

    If ``cancel`` is provided and not already in ``options``, it is added as a
    final row so the user can dismiss the prompt.
    """
    keyboard = [[{"text": opt}] for opt in options]
    if cancel and cancel not in options:
        keyboard.append([{"text": cancel}])
    return {
        "keyboard": keyboard,
        "resize_keyboard": True,
        "one_time_keyboard": True,
    }


def build_keyboard_remove() -> dict[str, Any]:
    """Return a ReplyKeyboardRemove markup."""
    return {"remove_keyboard": True}


def build_inline_keyboard(
    options: list[str], cancel: str | None = None, *, prefix: str = ASK_CALLBACK_PREFIX
) -> dict[str, Any]:
    """Return a Telegram InlineKeyboardMarkup for a list of option strings.

    Each option gets a short, deterministic ``callback_data`` value so the
    option text can be long without exceeding Telegram's 64-byte limit.
    If ``cancel`` is provided, a final row with a dedicated cancel callback is
    added.
    """
    keyboard: list[list[dict[str, Any]]] = [
        [{"text": opt, "callback_data": f"{prefix}{i}"}] for i, opt in enumerate(options)
    ]
    if cancel and cancel not in options:
        keyboard.append([{"text": cancel, "callback_data": f"{prefix}cancel"}])
    return {"inline_keyboard": keyboard}


def build_empty_inline_keyboard() -> dict[str, Any]:
    """Return an empty InlineKeyboardMarkup to remove an inline keyboard."""
    return {"inline_keyboard": []}


def is_ask_cancel_callback(data: str, *, prefix: str = ASK_CALLBACK_PREFIX) -> bool:
    """Return True if ``data`` is the inline cancel callback."""
    return data == f"{prefix}cancel"


def parse_ask_callback_index(
    data: str, *, prefix: str = ASK_CALLBACK_PREFIX
) -> int | None:
    """Parse a callback_data string into its option index, or None."""
    if not data.startswith(prefix):
        return None
    tail = data[len(prefix) :]
    if tail == "cancel":
        return None
    try:
        return int(tail)
    except ValueError:
        return None


def extract_ask_block(text: str) -> tuple[str, AskBlock | None]:
    """Extract a ```ask fenced JSON block from the text.

    Returns the text with the block removed and an AskBlock if one was found.
    """
    match = ASK_FENCE_RE.search(text)
    if not match:
        return text, None

    body = match.group(1).strip()
    if not body:
        return text, None

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return text, None

    options = data.get("options") or []
    if not options or not isinstance(options, list):
        return text, None

    question = data.get("question") or ""
    if not question:
        before = text[: match.start()].rstrip()
        if before:
            question = before.split("\n")[-1].strip() or before.strip()

    cancellable = bool(data.get("cancellable", False))
    cancel_label = data.get("cancel_label")
    if not cancel_label:
        cancel_label = "Cancel" if cancellable else ""

    visible = (text[: match.start()] + text[match.end() :]).strip()
    return visible, AskBlock(
        question=question,
        options=[str(o) for o in options],
        cancellable=cancellable,
        cancel_label=cancel_label,
    )
