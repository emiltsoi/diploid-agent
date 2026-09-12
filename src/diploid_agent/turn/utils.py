"""Small shared helpers for the turn pipeline."""

from __future__ import annotations


def join_notices(*parts: str | None) -> str | None:
    """Concatenate non-empty notice strings with a blank line between them."""
    joined = "\n\n".join(p for p in parts if p)
    return joined or None
