"""Tests for the M5 plugin hook surface."""

from pathlib import Path

from devin_fleet_harness.config import PluginConfig
from devin_fleet_harness.dispatch import Dispatch
from devin_fleet_harness.models import PartialTurn
from devin_fleet_harness.plugins.base import StatePlugin
from devin_fleet_harness.plugins.contexts import IdleContext
from devin_fleet_harness.plugins.manager import PluginManager


class HookSpy(StatePlugin):
    def __init__(self) -> None:
        super().__init__(PluginConfig(name="spy", enabled=True), "c1", Path("."))
        self.partial: PartialTurn | None = None
        self.dispatch_id: str | None = None
        self.event_name: str | None = None
        self.idle: IdleContext | None = None

    def on_partial(self, partial: PartialTurn) -> None:
        self.partial = partial

    def on_dispatch(self, chat_id: str, dispatch: Dispatch) -> None:
        self.dispatch_id = dispatch.id

    def on_event(self, event: str, payload: dict) -> None:
        self.event_name = event

    def on_idle(self, context: IdleContext) -> None:
        self.idle = context


def test_on_partial_and_dispatch_hooks():
    spy = HookSpy()
    mgr = PluginManager(
        plugins=[PluginConfig(name="spy", enabled=True, module=None)],
        sessions_root=Path("."),
        instance_id="i-1",
        instance_started_at=0.0,
    )
    # Force the spy instance into the manager's plugin cache.
    mgr._instances["c1"] = {"spy": spy}

    p = PartialTurn(
        chat_id="c1",
        session_number=1,
        turn_number=2,
        user_message="hi",
    )
    mgr.on_partial("c1", p)
    assert spy.partial is p

    d = Dispatch(id="d1", chat_id="c1", session_id="s1", status="pending")
    mgr.on_dispatch("c1", d)
    assert spy.dispatch_id == "d1"

    mgr.on_event("c1", "ping", {"x": 1})
    assert spy.event_name == "ping"

    mgr.on_idle(
        "c1",
        IdleContext(chat_id="c1", now=1.0, instance_id="i-1"),
    )
    assert spy.idle is not None
    assert spy.idle.chat_id == "c1"
