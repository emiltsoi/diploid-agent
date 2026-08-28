"""Interactive question/answer helpers for Telegram."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

ASK_FENCE_RE = re.compile(r"^```ask\s*\n([\s\S]*?)\n```\s*$", re.MULTILINE)


@dataclass(frozen=True)
class AskBlock:
    """A question and its selectable options."""

    question: str
    options: list[str]


def build_reply_keyboard(options: list[str]) -> dict[str, Any]:
    """Return a Telegram ReplyKeyboardMarkup for a list of option strings."""
    return {
        "keyboard": [[{"text": opt}] for opt in options],
        "resize_keyboard": True,
        "one_time_keyboard": True,
    }


def build_keyboard_remove() -> dict[str, Any]:
    """Return a ReplyKeyboardRemove markup."""
    return {"remove_keyboard": True}


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

    visible = (text[: match.start()] + text[match.end() :]).strip()
    return visible, AskBlock(question=question, options=[str(o) for o in options])
