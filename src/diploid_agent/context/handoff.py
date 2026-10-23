"""Harness-written handoff stub for context-pressure folds.

When pressure forces a fresh session the handoff convention relies on the
outgoing turn having authored ``## next-self`` in ``chat_self_state.md`` during
its handoff turn. A turn that dies before writing — or a pressure path that
never granted one — leaves the next self nothing, which is how a compaction
becomes a cold start. The stub is the floor, not a competitor: a harness-written
block carrying the durable record fields, written *before* the fold, and it
never overwrites a model-authored ``## next-self`` section.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from diploid_agent._compat import safe_dir_name

logger = logging.getLogger(__name__)

_MARKER_OPEN = "<!-- harness-handoff -->"
_MARKER_CLOSE = "<!-- /harness-handoff -->"
_HEADING = "## next-self"
_MARKER_RE = re.compile(re.escape(_MARKER_OPEN) + r".*?" + re.escape(_MARKER_CLOSE), re.DOTALL)


def handoff_stub_path(sessions_root: Path, chat_id: str) -> Path:
    """The per-chat self-state file. The name matches ``_CHAT_DURABLE_FILES``;
    a persona that renames its self-state slot is out of the stub's reach —
    the stub writes the convention, not every override."""
    return Path(sessions_root).expanduser() / safe_dir_name(chat_id) / "chat_self_state.md"


def _last_turn_snapshot(chat_dir: Path) -> dict:
    """The continuity plugin's streaming snapshot: intent and last side effect
    of the most recent in-flight (or interrupted) turn. ``chat_active_turn.json``
    is promoted to ``chat_interrupted_turn.json`` when a turn dies mid-write —
    the exact case this stub exists for."""
    for name in ("chat_interrupted_turn.json", "chat_active_turn.json"):
        try:
            data = json.loads((chat_dir / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return {}


def _stub_block(
    *, session_number: int, turn_number: int, intent: str, side_effect: str, arriving: str
) -> str:
    when = datetime.fromtimestamp(time.time(), tz=UTC).isoformat(timespec="seconds")
    return (
        f"{_HEADING}\n\n{_MARKER_OPEN}\n"
        f"Harness-written stub at the context-pressure fold "
        f"(session {session_number}, before turn {turn_number}, {when}). "
        "The previous turn may not have survived to write its own handoff — "
        "if a fresher `## next-self` appears elsewhere in this file, prefer it.\n"
        f"- last durable intent: {intent or '(none recorded)'}\n"
        f"- last side effect: {side_effect or '(none recorded)'}\n"
        f"- message arriving at the fold: {arriving or '(none)'}\n"
        "- durable record: chat_MEMORY.md + persona MEMORY.md "
        "+ chat_open_items.jsonl; live thread: chat_transcript.jsonl tail\n"
        f"{_MARKER_CLOSE}\n"
    )


def write_handoff_stub(
    path: Path,
    *,
    session_number: int,
    turn_number: int,
    arriving: str,
) -> bool:
    """Ensure the self-state file carries a handoff for the fold.

    Rules:
    - an existing ``<!-- harness-handoff -->`` block is refreshed in place
      (ours to update — keeps the stub current across repeated folds);
    - a ``## next-self`` without our marker means a previous turn authored
      its own handoff — the pen wins, nothing is written;
    - otherwise the stub is appended as the file's ``## next-self`` section.

    Never raises into the caller: a stub that breaks prompt construction
    would be worse than no stub. Returns True when a write happened.
    """
    arriving = " ".join((arriving or "").split())[:300]
    snapshot = _last_turn_snapshot(path.parent)
    intent = " ".join(str(snapshot.get("current_intent") or "").split())[:300]
    if not intent:
        msg = str(snapshot.get("user_message") or "").strip().splitlines()
        intent = (msg[0] if msg else "")[:200]
    side_effect = " ".join(str(snapshot.get("last_side_effect") or "").split())[:300]
    block = _stub_block(
        session_number=session_number,
        turn_number=turn_number,
        intent=intent,
        side_effect=side_effect,
        arriving=arriving,
    )
    try:
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        if _MARKER_OPEN in text:
            inner = block.split(_MARKER_OPEN, 1)[1].rsplit(_MARKER_CLOSE, 1)[0]
            text = _MARKER_RE.sub(_MARKER_OPEN + inner + _MARKER_CLOSE, text, count=1)
        elif _HEADING in text:
            return False
        else:
            text = text.rstrip("\n") + ("\n\n" if text.strip() else "") + block
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return True
    except OSError:
        logger.warning("Could not write handoff stub at %s", path)
        return False
