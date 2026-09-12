"""Shared human-readable text formatters."""

from __future__ import annotations


def human_duration(seconds: float) -> str:
    """Return a compact, human-readable duration like ``1h 2m 3s``."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


def compact_duration(seconds: float) -> str:
    """Return a short duration string, dropping zero components.

    ``90`` -> ``1m 30s``, ``3600`` -> ``1h``, ``3660`` -> ``1h 1m``.
    """
    seconds = max(seconds, 0)
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours == 1:
        return f"1h {minutes}m" if minutes else "1h"
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def elapsed_short(seconds: float) -> str:
    """Return elapsed time as ``Nm Ss`` (minutes not capped at 60)."""
    total = int(seconds)
    mins, secs = divmod(total, 60)
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"
