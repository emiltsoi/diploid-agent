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
# Handoff detection must anchor at line start — a mid-prose mention
# (`## next-self` inside backticks, convention docs, quoted stubs) is not a
# heading and must never count as an authored handoff. Same convention as
# the continuity plugin's `^##\s*next-self\b` check.
_HEADING_RE = re.compile(r"^##\s*next-self\b", re.IGNORECASE | re.MULTILINE)
_MARKER_RE = re.compile(re.escape(_MARKER_OPEN) + r".*?" + re.escape(_MARKER_CLOSE), re.DOTALL)
# The stub writes its own bare `_HEADING` above the marker, so classifying a
# file must strip the whole stub *section* — heading included — before asking
# whether a model-authored handoff exists alongside it.
_STUB_SECTION_RE = re.compile(
    r"^" + re.escape(_HEADING) + r"\s*\n+\s*" + _MARKER_RE.pattern,
    re.DOTALL | re.MULTILINE,
)


def handoff_outcome(text: str) -> str:
    """Classify a self-state file's handoff state at a fold.

    - ``"authored"`` — a model-written ``## next-self`` exists (with or
      without a stub alongside it);
    - ``"stub_only"`` — the harness floor stub is the only handoff present;
    - ``"none"`` — no handoff of either kind. A stray marker mention in
      prose is not a stub and classifies here too.
    """
    if _STUB_SECTION_RE.search(text):
        stripped = _STUB_SECTION_RE.sub("", text)
        return "authored" if _HEADING_RE.search(stripped) else "stub_only"
    return "authored" if _HEADING_RE.search(text) else "none"


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
    *,
    session_number: int,
    turn_number: int,
    intent: str,
    side_effect: str,
    arriving: str,
    model_authored: bool,
) -> str:
    when = datetime.fromtimestamp(time.time(), tz=UTC).isoformat(timespec="seconds")
    if model_authored:
        handoff_line = (
            "- outgoing handoff: the previous self DID author `## next-self` — "
            "prefer it over this floor."
        )
    else:
        handoff_line = (
            "- outgoing handoff: the previous self did NOT author `## next-self` "
            "— this stub is the only floor."
        )
    return (
        f"{_HEADING}\n\n{_MARKER_OPEN}\n"
        f"Harness-written stub at the context-pressure fold "
        f"(session {session_number}, before turn {turn_number}, {when}).\n"
        f"{handoff_line}\n"
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
    try:
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    except OSError:
        logger.warning("Could not read handoff stub at %s", path)
        return False
    block = _stub_block(
        session_number=session_number,
        turn_number=turn_number,
        intent=intent,
        side_effect=side_effect,
        arriving=arriving,
        model_authored=handoff_outcome(text) == "authored",
    )
    try:
        if _MARKER_OPEN in text:
            inner = block.split(_MARKER_OPEN, 1)[1].rsplit(_MARKER_CLOSE, 1)[0]
            text = _MARKER_RE.sub(_MARKER_OPEN + inner + _MARKER_CLOSE, text, count=1)
        elif _HEADING_RE.search(text):
            return False
        else:
            text = text.rstrip("\n") + ("\n\n" if text.strip() else "") + block
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return True
    except OSError:
        logger.warning("Could not write handoff stub at %s", path)
        return False
