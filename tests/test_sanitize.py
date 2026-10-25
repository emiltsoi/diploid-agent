"""Lone-surrogate scrubbing — the wake-queue wedge regression suite.

A lone UTF-16 surrogate can live in a Python str but can never encode as
UTF-8: on Windows it arrives via locale-codec surrogateescape decodes (an
em-dash's trailing UTF-8 byte under GBK), rides a wire payload into durable
state, and then crashes every json.dumps().encode("utf-8") that touches it —
a peer's wake queue retried one such message 48 times before manual repair.
"""

from __future__ import annotations

import json
from pathlib import Path

from diploid_agent.sanitize import sanitize_json_value, sanitize_surrogates


def test_sanitize_surrogates_replaces_lone_surrogates() -> None:
    assert sanitize_surrogates("ok — \udc94 tail") == "ok — � tail"
    assert sanitize_surrogates("\ud800\udfff") == "��"


def test_sanitize_surrogates_leaves_valid_text_alone() -> None:
    text = "em-dash — emoji 👾 pair \U0001f47e survives"
    assert sanitize_surrogates(text) == text


def test_sanitize_json_value_recurses() -> None:
    dirty = {
        "text": "a\udc94b",
        "nested": ["x", {"inner": "\udc80"}],
        "key\udc81": 1,
        "n": 3,
    }
    clean = sanitize_json_value(dirty)
    assert clean["text"] == "a�b"
    assert clean["nested"][1]["inner"] == "�"
    assert clean["key�"] == 1
    assert clean["n"] == 3
    json.dumps(clean).encode("utf-8")  # must not raise


def test_wake_enqueue_scrubs_payload(tmp_path: Path) -> None:
    from diploid_agent.models import WakeEvent
    from diploid_agent.runtime.wake_queue import WakeQueue

    q = WakeQueue(tmp_path / "wake_queue.jsonl")
    event = WakeEvent(
        id="",
        chat_id="mesh:vesper",
        reason="mesh",
        priority=5,
        scheduled_at=0.0,
        payload={"text": "a\udc94b"},
    )
    q.enqueue(event)
    raw = (tmp_path / "wake_queue.jsonl").read_text(encoding="utf-8")
    assert "\\ufffd" in raw and "\\udc94" not in raw
    # The poisoned event itself must encode cleanly on every later retry.
    json.dumps(event.payload).encode("utf-8")


def test_transcript_append_survives_surrogate(tmp_path: Path) -> None:
    from diploid_agent.memory import MemoryManager

    mgr = object.__new__(MemoryManager)
    mgr.sessions_root = tmp_path
    mgr.chat_id = "c1"
    mgr._append_transcript("has \udc94 surrogate", "reply ok")
    entries = [
        json.loads(line)
        for line in (tmp_path / "c1" / "chat_transcript.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert entries[0]["content"] == "has � surrogate"
    assert entries[1]["content"] == "reply ok"
