"""Pluggable per-chat state layer."""

from diploid_agent.plugins.base import (
    SleepContext,
    StatePlugin,
    TurnInfo,
    WakeContext,
)
from diploid_agent.plugins.json_state import JsonStatePlugin
from diploid_agent.plugins.manager import PluginManager

__all__ = [
    "JsonStatePlugin",
    "PluginManager",
    "SleepContext",
    "StatePlugin",
    "TurnInfo",
    "WakeContext",
]
