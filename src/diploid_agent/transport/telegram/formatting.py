"""Telegram message formatting helpers used by the long-polling transport."""

from __future__ import annotations

import time

_THINKING_PREFIX = "Thinking..."
_THINKING_CONTINUED = "... (thinking continues)"
_HEARTBEAT_INTERVAL = 30.0
_REPLY_PLACEHOLDER = "..."

_TELEGRAM_HELP = """Available commands:

/status - current model, session id, working directory, and context-window usage
/metrics - token usage and latency for this chat
/mcp list | /mcp enable <name> | /mcp disable <name> - manage MCP servers
/skill list | /skill enable <name> | /skill disable <name> | /skill create <name> <markdown> - manage skills
/plugin list | /plugin enable <name> | /plugin disable <name> | /plugin reload <name> - manage state plugins
/state <plugin> <event> [args...] - dispatch a state event to a plugin
/memory - show per-chat memory
/models - list ACP model names
/model <name> - switch this chat to a new model
/new - start a fresh Devin session while keeping chat memory
/stop - cancel the current turn and return a partial reply
/restart - kill the ACP subprocess and start a fresh transport
/graceful-restart [service] - schedule a graceful restart of the named service
/subagent <prompt> - start a background subagent and continue when it finishes
/subagents - list background subagents for this chat
/continue - resume the previous turn after a partial reply or timeout
/sessions - list numbered sessions for this chat
/resume <n> - resume session n as the active session
/branch <n> - branch from session n and make it active
/summarize - trigger file-backed summarization
/recall <query> - search memory for relevant context
/promote <fact> - append a fact to persona global memory
/stream_thoughts on|off - toggle the optional real-time thought stream
/config <section> <key>=<value> [key=value...] - update live runtime config (task|waker|timer|notifications|telegram)
/help - show this list"""


def _format_thought(thought: str, limit: int = 4096) -> str:
    """Return a Telegram-sized thought block, rolling the tail once it grows past the limit.

    Short thoughts are shown with the normal prefix. Once the limit is exceeded,
    the placeholder switches to a "continues" marker and shows the latest tail so
    the user can always see the most recent reasoning without generating new
    Telegram messages.
    """
    if not thought:
        return ""
    full = f"{_THINKING_PREFIX}\n{thought}"
    if len(full) <= limit:
        return full
    tail_limit = limit - len(_THINKING_CONTINUED) - 1  # -1 for the newline
    if tail_limit <= 0:
        return thought[-limit:]
    return f"{_THINKING_CONTINUED}\n{thought[-tail_limit:]}"


def _format_elapsed(seconds: float) -> str:
    """Return a short, human-readable elapsed time."""
    minutes, secs = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_subagent_time(ts: float | None) -> str:
    """Return a concise UTC time string for a subagent started/finished time."""
    if ts is None:
        return "—"
    return time.strftime("%H:%M:%S", time.gmtime(ts))


def _build_heartbeat_text(base: str, elapsed: float, limit: int = 4096) -> str:
    """Return ``base`` with a small liveness suffix, truncating if needed."""
    suffix = f"\n\n(still working, {_format_elapsed(elapsed)})"
    if not base:
        return suffix[1:] if len(suffix) > limit else suffix
    total = base + suffix
    if len(total) <= limit:
        return total
    max_base = limit - len(suffix) - 3
    if max_base <= 0:
        return total[:limit]
    return base[:max_base] + "..." + suffix
