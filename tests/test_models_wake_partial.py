"""Tests for the Milestone 5 dataclasses."""

from dataclasses import fields

from acp_fleet_harness.models import PartialTurn, WakeEvent
from acp_fleet_harness.plugins.contexts import IdleContext


def test_wake_event_defaults() -> None:
    e = WakeEvent(
        id="wake-1",
        chat_id="chat-1",
        reason="dispatch",
        priority=1,
        scheduled_at=0.0,
        created_at=0.0,
    )
    assert e.silent is True
    assert e.ready is False
    assert e.attempts == 0
    assert e.leased_until is None


def test_wake_event_roundtrip() -> None:
    e = WakeEvent(
        id="wake-1",
        chat_id="chat-1",
        reason="dispatch",
        priority=1,
        scheduled_at=0.0,
        created_at=0.0,
        leased_until=123.0,
    )
    restored = WakeEvent.from_dict(e.to_dict())
    assert restored == e


def test_partial_turn_fields() -> None:
    p = PartialTurn(
        chat_id="chat-1",
        session_number=1,
        turn_number=2,
        user_message="hello",
        message_text="partial reply",
        thought_text="partial thought",
        updated_at=1.0,
    )
    assert p.message_text == "partial reply"
    assert p.thought_text == "partial thought"


def test_idle_context_fields() -> None:
    c = IdleContext(chat_id="chat-1", now=1.0, instance_id="i-1")
    assert c.now == 1.0
    assert "record" in {f.name for f in fields(IdleContext)}
