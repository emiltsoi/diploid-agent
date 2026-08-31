"""Tests for the typed PluginRuntime surface."""

from __future__ import annotations

from pathlib import Path

from diploid_agent.config import PluginConfig
from diploid_agent.models import ChatResult
from diploid_agent.plugins.base import StatePlugin
from diploid_agent.runtime.plugin_runtime import PluginRuntime


class FakeRuntime:
    """A minimal object that satisfies the PluginRuntime protocol."""

    @property
    def config(self):
        return None

    @property
    def instance_id(self):
        return "fake"

    @property
    def instance_started_at(self):
        return 0.0

    @property
    def sessions_root(self):
        return Path("/tmp")

    @property
    def engine(self):
        return None

    @property
    def wake_queue(self):
        return None

    def plan_create(self, *args, **kwargs):
        return None

    def plan_task_start(self, *args, **kwargs):
        return None

    def call_engine_unlocked(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def is_continuation_message(self, text):
        return text.lower() == "continue"

    def recall(self, chat_id, query, tags=None, max_tokens=None):
        return ChatResult(reply=f"Recall: {query}")

    def promote(self, chat_id, fact):
        return ChatResult(reply="Promoted.")

    def suppress_auto_continue(self, chat_id, seconds=300.0):
        return None

    def is_auto_continue_suppressed(self, chat_id):
        return False


class FakePlugin(StatePlugin):
    pass


def test_plugin_runtime_is_typed_surface():
    cfg = PluginConfig(name="fake", module=None)
    plugin = FakePlugin(cfg, "1", Path("/tmp"), runtime=FakeRuntime())
    assert isinstance(plugin._runtime, PluginRuntime)
