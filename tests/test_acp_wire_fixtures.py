"""Replay real captured ACP session/update payloads through TurnStream.

The fixture (tests/fixtures/acp_updates.jsonl) is produced by
tests/capture_acp_updates.py — a manual script that runs a real
``devin acp`` child and dumps every on_update payload. Replaying it pins
the actual wire shape so extraction fixes are tested against reality,
not guessed fixtures.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from diploid_agent.models import ActiveTurn
from diploid_agent.turn.stream import TurnStream

FIXTURE = Path(__file__).parent / "fixtures" / "acp_updates.jsonl"


class _FakeRuntime:
    def __init__(self) -> None:
        self._active_turns: dict[str, ActiveTurn] = {}
        self._lock = threading.RLock()
        self._plugins = SimpleNamespace(on_partial=lambda *a, **k: None)
        self.active_record = lambda chat_id: None  # type: ignore[method-assign]


def _replay() -> ActiveTurn:
    updates = [json.loads(l) for l in FIXTURE.read_text().splitlines() if l.strip()]
    runtime = _FakeRuntime()
    active = ActiveTurn("chat-1", "s-1", "hello", time.time())
    runtime._active_turns["chat-1"] = active
    stream = TurnStream(runtime, "chat-1")
    for update in updates:
        stream.on_update(update)
    return active


@pytest.mark.skipif(not FIXTURE.exists(), reason="run tests/capture_acp_updates.py")
def test_real_updates_compose_stable_side_effects() -> None:
    active = _replay()

    assert active.side_effects, "fixture should contain tool updates"
    # No effect may ever surface a raw toolCallId hash as its title.
    for fx in active.side_effects:
        assert "#" not in fx["title"], f"leaked call id: {fx['title']}"
        assert fx["title"] != "tool"


@pytest.mark.skipif(not FIXTURE.exists(), reason="run tests/capture_acp_updates.py")
def test_real_updates_never_flap_title_per_call() -> None:
    """Each toolCallId keeps one title across its update chunks."""
    updates = [json.loads(l) for l in FIXTURE.read_text().splitlines() if l.strip()]
    runtime = _FakeRuntime()
    active = ActiveTurn("chat-1", "s-1", "hello", time.time())
    runtime._active_turns["chat-1"] = active
    stream = TurnStream(runtime, "chat-1")

    titles_by_call: dict[str, set[str]] = {}
    for update in updates:
        if update.get("sessionUpdate") not in ("tool_call", "tool_call_update"):
            continue
        stream.on_update(update)
        call_id = update.get("toolCallId")
        if isinstance(call_id, str):
            titles_by_call.setdefault(call_id, set()).add(active.last_side_effect)

    for call_id, titles in titles_by_call.items():
        assert len({t.rsplit(" (", 1)[0] for t in titles}) == 1, (
            f"{call_id} flapped titles: {titles}"
        )
