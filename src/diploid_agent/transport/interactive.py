"""Interactive question/answer helpers for Telegram."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

ASK_FENCE_RE = re.compile(r"^```ask\s*\n([\s\S]*?)\n```\s*$", re.MULTILINE)

ASK_CALLBACK_PREFIX = "ask_"
ASK_CANCEL_CALLBACK_DATA = f"{ASK_CALLBACK_PREFIX}cancel"

# Normalised forms of the old open-ended "Other (please specify)" escape option.
# These are removed from the option list because every ask dialog now has a
# default cancel button instead.
_OTHER_ESCAPE_FORMS = frozenset(
    {
        "otherpleasespecify",
        "otherspleasespecify",
        "otherandspecify",
        "othersandspecify",
        "otherspecify",
        "othersspecify",
    }
)


def _is_open_ended_escape(option: str) -> bool:
    """Return True if the option is the old 'Other (please specify)' escape hatch."""
    normalized = re.sub(r"[^a-z0-9]", "", option.lower())
    return normalized in _OTHER_ESCAPE_FORMS


@dataclass(frozen=True)
class AskBlock:
    """A question, its selectable options, and a cancel button (shown by default)."""

    question: str
    options: list[str]
    cancellable: bool = True
    cancel_label: str = "Cancel"


def build_reply_keyboard(options: list[str], cancel: str | None = None) -> dict[str, Any]:
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
    added. If one of ``options`` matches the cancel label, that option is
    treated as the cancel and gets the cancel callback instead of an index.
    """
    keyboard: list[list[dict[str, Any]]] = []
    for i, opt in enumerate(options):
        if cancel and opt == cancel:
            callback = f"{prefix}cancel"
        else:
            callback = f"{prefix}{i}"
        keyboard.append([{"text": opt, "callback_data": callback}])
    if cancel and cancel not in options:
        keyboard.append([{"text": cancel, "callback_data": f"{prefix}cancel"}])
    return {"inline_keyboard": keyboard}


def build_empty_inline_keyboard() -> dict[str, Any]:
    """Return an empty InlineKeyboardMarkup to remove an inline keyboard."""
    return {"inline_keyboard": []}


def is_ask_cancel_callback(data: str, *, prefix: str = ASK_CALLBACK_PREFIX) -> bool:
    """Return True if ``data`` is the inline cancel callback."""
    return data == f"{prefix}cancel"


def parse_ask_callback_index(data: str, *, prefix: str = ASK_CALLBACK_PREFIX) -> int | None:
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

    raw_options = data.get("options")
    if not isinstance(raw_options, list) or not raw_options:
        return text, None
    options = [str(o) for o in raw_options]

    question = data.get("question") or ""
    if not question:
        before = text[: match.start()].rstrip()
        if before:
            question = before.split("\n")[-1].strip() or before.strip()

    # Ask blocks are cancellable by default. An explicit `cancellable: false`
    # disables the cancel button for forced-choice prompts.
    cancellable = bool(data.get("cancellable", True))

    # Remove the old open-ended "Other (please specify)" escape option. The
    # default cancel button is the canonical way to dismiss a prompt.
    filtered_options: list[str] = []
    for opt in options:
        if _is_open_ended_escape(opt):
            cancellable = True
        else:
            filtered_options.append(opt)
    options = filtered_options

    if not options:
        return text, None

    cancel_label = data.get("cancel_label")
    if not cancel_label:
        cancel_label = "Cancel" if cancellable else ""

    visible = (text[: match.start()] + text[match.end() :]).strip()
    return visible, AskBlock(
        question=question,
        options=options,
        cancellable=cancellable,
        cancel_label=cancel_label,
    )
