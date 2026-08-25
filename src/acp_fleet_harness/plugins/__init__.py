"""Pluggable per-chat state layer."""

from acp_fleet_harness.plugins.base import (
    SleepContext,
    StatePlugin,
    TurnInfo,
    WakeContext,
)
from acp_fleet_harness.plugins.json_state import JsonStatePlugin
from acp_fleet_harness.plugins.manager import PluginManager

__all__ = [
    "JsonStatePlugin",
    "PluginManager",
    "SleepContext",
    "StatePlugin",
    "TurnInfo",
    "WakeContext",
]
