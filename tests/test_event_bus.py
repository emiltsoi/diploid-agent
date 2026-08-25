"""Tests for the EventBus skeleton."""

from __future__ import annotations

import time

from devin_fleet_harness.runtime.event_bus import Event, EventBus


def _wait_for(captured: list[Event], timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if captured:
            return
        time.sleep(0.05)
    raise AssertionError("expected event was not delivered")


def test_post_and_subscribe() -> None:
    bus = EventBus()
    bus.start()

    captured: list[Event] = []
    bus.subscribe(lambda e: captured.append(e))

    bus.post(Event(type="test.hello", payload={"x": 1}))
    _wait_for(captured)

    assert len(captured) == 1
    assert captured[0].type == "test.hello"
    assert captured[0].payload == {"x": 1}
    bus.stop()


def test_multiple_subscribers() -> None:
    bus = EventBus()
    bus.start()

    first: list[Event] = []
    second: list[Event] = []
    bus.subscribe(lambda e: first.append(e))
    bus.subscribe(lambda e: second.append(e))

    bus.post(Event(type="multi"))
    _wait_for(first)
    _wait_for(second)

    assert len(first) == 1
    assert len(second) == 1
    bus.stop()


def test_unsubscribe() -> None:
    bus = EventBus()
    bus.start()

    captured: list[Event] = []

    def handler(event: Event) -> None:
        captured.append(event)

    bus.subscribe(handler)
    bus.post(Event(type="subscribed"))
    _wait_for(captured)
    assert len(captured) == 1

    captured.clear()
    bus.unsubscribe(handler)
    bus.post(Event(type="after"))
    time.sleep(0.2)

    assert len(captured) == 0
    bus.stop()


def test_subscriber_exception_does_not_crash_bus() -> None:
    bus = EventBus()
    bus.start()

    captured: list[Event] = []

    def bad(_event: Event) -> None:
        raise RuntimeError("boom")

    bus.subscribe(bad)
    bus.subscribe(lambda e: captured.append(e))

    bus.post(Event(type="resilient"))
    _wait_for(captured)

    assert len(captured) == 1
    bus.stop()
