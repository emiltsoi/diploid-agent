"""Tests for the turn-flow plugin lifecycle hooks."""

from pathlib import Path
from typing import Any

import pytest

from diploid_agent.acp_client import AcpPromptResult
from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    PersonaConfig,
    PluginConfig,
    Secrets,
)
from diploid_agent.harness import ConversationHarness
from diploid_agent.models import ChatResult
from diploid_agent.plugins.base import StatePlugin
from diploid_agent.plugins.contexts import (
    DispatchCompleteContext,
    DispatchContinueContext,
    DispatchCreateContext,
    EngineCallContext,
    EngineResultContext,
    McpCommandContext,
    MemoryTransitionContext,
    PromoteContext,
    PromptContext,
    RecordTurnContext,
    RetainContext,
    SessionActiveContext,
    SessionArchiveContext,
    SessionClearContext,
    SessionStartContext,
    ShutdownContext,
    SkillCommandContext,
    TurnErrorContext,
    TurnStartContext,
    UserMessageContext,
)


def _make_config(tmp_path: Path, fixture_root: Path) -> Config:
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=fixture_root,
            fleet_root=tmp_path / "fleet",
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            memory={"backend": "file"},  # type: ignore[arg-type]
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


class _BaseTestPlugin(StatePlugin):
    """A no-op test plugin that can be subclassed for specific hooks."""

    def __init__(self) -> None:
        super().__init__(
            PluginConfig(name="test", enabled=True, prompt_slot="wake"),
            "test-chat",
            Path("."),
        )


def _install_plugin(harness: ConversationHarness, plugin: StatePlugin) -> None:
    """Replace the plugin manager's plugin list with a single test plugin."""
    monkeypatchable = harness._plugins

    def _plugins_for(_chat_id: str) -> list[StatePlugin]:
        return [plugin]

    monkeypatchable._plugins_for = _plugins_for  # type: ignore[assignment]


def test_before_turn_short_circuits_turn(monkeypatch, tmp_path: Path) -> None:
    """A gate hook can return ChatResult and skip the entire turn."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    called = False

    class InterceptPlugin(_BaseTestPlugin):
        def before_turn(self, context: TurnStartContext) -> ChatResult | None:
            nonlocal called
            called = True
            return ChatResult(
                reply="Short-circuited",
                session_id="plugin-session",
                session_number=1,
                turn_number=1,
            )

    _install_plugin(harness, InterceptPlugin())

    result = harness.process("chat-1", "hello")
    assert called
    assert result.reply == "Short-circuited"
    assert result.session_id == "plugin-session"


def test_before_turn_can_modify_user_message(monkeypatch, tmp_path: Path) -> None:
    """A gate hook can modify the user message before prompt building."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    captured: list[str] = []

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        captured.append(prompt)
        return AcpPromptResult(reply="ok", session_id="s1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    class ModifyPlugin(_BaseTestPlugin):
        def before_turn(self, context: TurnStartContext) -> TurnStartContext:
            context.user_message = "[prefix] " + context.user_message
            return context

    _install_plugin(harness, ModifyPlugin())

    harness.process("chat-1", "hello")
    assert captured
    assert "[prefix] hello" in captured[0]


def test_before_format_user_message_modifies_prompt(monkeypatch, tmp_path: Path) -> None:
    """A consult hook can rewrite the user message before it is formatted."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    captured: list[str] = []

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        captured.append(prompt)
        return AcpPromptResult(reply="ok", session_id="s1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    class FormatPlugin(_BaseTestPlugin):
        def before_format_user_message(
            self,
            context: UserMessageContext,
        ) -> UserMessageContext:
            context.raw_message = "[fmt] " + context.raw_message
            context.formatted_message = None
            return context

    _install_plugin(harness, FormatPlugin())

    harness.process("chat-1", "hello")
    assert captured
    assert "[fmt] hello" in captured[0]


def test_after_prompt_built_injects_context(monkeypatch, tmp_path: Path) -> None:
    """A consult hook can modify the assembled prompt."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    captured: list[str] = []

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        captured.append(prompt)
        return AcpPromptResult(reply="ok", session_id="s1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    class PromptPlugin(_BaseTestPlugin):
        def after_prompt_built(self, pctx):
            pctx.prompt += "\n\n[INJECTED]"
            return pctx

    _install_plugin(harness, PromptPlugin())

    harness.process("chat-1", "hello")
    assert captured
    assert "[INJECTED]" in captured[0]


def test_before_engine_call_short_circuits(monkeypatch, tmp_path: Path) -> None:
    """The before_engine_call gate can skip the ACP call."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    called = False

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        raise AssertionError("engine should not be called")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    class EngineInterceptPlugin(_BaseTestPlugin):
        def before_engine_call(
            self,
            context: EngineCallContext,
        ) -> ChatResult | None:
            nonlocal called
            called = True
            return ChatResult(
                reply="from-plugin",
                session_id="plugin-session",
                session_number=1,
                turn_number=1,
            )

    _install_plugin(harness, EngineInterceptPlugin())

    result = harness.process("chat-1", "hello")
    assert called
    assert result.reply == "from-plugin"


def test_after_engine_call_modifies_reply(monkeypatch, tmp_path: Path) -> None:
    """A consult hook can rewrite the engine reply."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        return AcpPromptResult(reply="engine-reply", session_id="s1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    class EngineResultPlugin(_BaseTestPlugin):
        def after_engine_call(
            self,
            context: EngineResultContext,
        ) -> EngineResultContext:
            context.reply = "rewritten"
            context.result.reply = "rewritten"
            return context

    _install_plugin(harness, EngineResultPlugin())

    result = harness.process("chat-1", "hello")
    assert result.reply == "rewritten"


def test_before_record_turn_modifies_recorded_turn(monkeypatch, tmp_path: Path) -> None:
    """A consult hook can modify the reply before it is recorded."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        return AcpPromptResult(reply="engine-reply", session_id="s1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    class RecordPlugin(_BaseTestPlugin):
        def before_record_turn(
            self,
            context: RecordTurnContext,
        ) -> RecordTurnContext:
            context.reply = "recorded-reply"
            return context

    _install_plugin(harness, RecordPlugin())

    result = harness.process("chat-1", "hello")
    assert result.reply == "recorded-reply"


def test_first_chatresult_wins_and_stops_chain(tmp_path: Path) -> None:
    """When the first plugin returns ChatResult, later plugins are not called."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    second_called = False

    class FirstPlugin(_BaseTestPlugin):
        def __init__(self) -> None:
            super().__init__()
            self.config.prompt_order = 1

        def before_turn(self, context: TurnStartContext) -> ChatResult:
            return ChatResult(
                reply="first-wins",
                session_id="s1",
                session_number=1,
                turn_number=1,
            )

    class SecondPlugin(_BaseTestPlugin):
        def __init__(self) -> None:
            super().__init__()
            self.config.prompt_order = 2

        def before_turn(self, context: TurnStartContext) -> ChatResult:
            nonlocal second_called
            second_called = True
            return ChatResult(
                reply="second-wins",
                session_id="s2",
                session_number=2,
                turn_number=2,
            )

    # Keep the manager's sorting behavior by giving it two configs and a cache.
    harness._plugins._plugins = [
        FirstPlugin().config,
        SecondPlugin().config,
    ]

    def _plugins_for(chat_id: str) -> list[StatePlugin]:
        return [FirstPlugin(), SecondPlugin()]

    harness._plugins._plugins_for = _plugins_for  # type: ignore[assignment]

    result = harness.process("chat-1", "hello")
    assert result.reply == "first-wins"
    assert not second_called


def test_on_turn_error_notifies_on_exception(monkeypatch, tmp_path: Path) -> None:
    """The on_turn_error notify hook is called when a turn raises."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        raise ValueError("boom")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    captured: TurnErrorContext | None = None

    class ErrorPlugin(_BaseTestPlugin):
        def on_turn_error(self, context: TurnErrorContext) -> None:
            nonlocal captured
            captured = context

    _install_plugin(harness, ErrorPlugin())

    with pytest.raises(ValueError, match="boom"):
        harness.process("chat-1", "hello")

    assert captured is not None
    assert captured.chat_id == "chat-1"
    assert captured.user_message == "hello"
    assert isinstance(captured.exception, ValueError)


def test_session_hooks_fire_on_new_session(monkeypatch, tmp_path: Path) -> None:
    """Session archive/clear/start/active hooks fire during /new."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        return AcpPromptResult(reply="ok", session_id="s2")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    captured: dict[str, Any] = {}

    class SessionPlugin(_BaseTestPlugin):
        def before_session_archive(self, context: SessionArchiveContext) -> SessionArchiveContext:
            captured["archive"] = context
            return context

        def before_session_clear(self, context: SessionClearContext) -> SessionClearContext:
            captured["clear"] = context
            return context

        def before_session_start(self, context: SessionStartContext) -> SessionStartContext:
            captured["start"] = context
            context.user_message = "[new] " + context.user_message
            return context

        def after_session_active(self, context: SessionActiveContext) -> None:
            captured["active"] = context

    _install_plugin(harness, SessionPlugin())

    harness.process("chat-1", "hello")
    harness.new_session("chat-1")

    assert "archive" in captured
    assert "clear" in captured
    assert "start" in captured
    assert "active" in captured
    assert captured["start"].kind == "new"
    assert captured["active"].record is not None


def test_before_session_start_can_short_circuit_new_session(monkeypatch, tmp_path: Path) -> None:
    """The before_session_start gate can prevent a new session."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    call_count = 0

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise AssertionError("engine should not be called for new session")
        return AcpPromptResult(reply="ok", session_id="s1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    class ShortCircuitNewPlugin(_BaseTestPlugin):
        def before_session_start(self, context: SessionStartContext) -> ChatResult | None:
            return ChatResult(reply="no-new-session")

    _install_plugin(harness, ShortCircuitNewPlugin())

    harness.process("chat-1", "hello")
    result = harness.new_session("chat-1")
    assert result.reply == "no-new-session"


def test_dispatch_hooks_fire_on_dispatch_and_continue(monkeypatch, tmp_path: Path) -> None:
    """Dispatch create/continue hooks fire and can modify context."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        return AcpPromptResult(reply="ok", session_id="s1")

    def fake_send_message(session_id: str, prompt: str, **kwargs: Any) -> AcpPromptResult:
        return AcpPromptResult(reply="follow-up")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)

    captured: dict[str, Any] = {}

    class DispatchPlugin(_BaseTestPlugin):
        def before_dispatch(self, context: DispatchCreateContext) -> DispatchCreateContext:
            captured["before_dispatch"] = context
            context.context = "[plugin-context]"
            return context

        def after_dispatch(self, context: DispatchCreateContext) -> None:
            captured["after_dispatch"] = context

        def before_dispatch_continue(
            self, context: DispatchContinueContext
        ) -> DispatchContinueContext:
            captured["before_dispatch_continue"] = context
            context.result = "[plugin-result]"
            return context

        def after_dispatch_continue(self, context: DispatchCompleteContext) -> None:
            captured["after_dispatch_continue"] = context

    _install_plugin(harness, DispatchPlugin())

    harness.process("chat-1", "hello")
    result = harness.dispatch("chat-1", "do work")
    assert "before_dispatch" in captured
    assert "after_dispatch" in captured

    continue_result = harness.continue_turn(result.dispatch_id, "work done")
    assert continue_result.reply
    assert "before_dispatch_continue" in captured
    assert "after_dispatch_continue" in captured
    assert captured["before_dispatch_continue"].result == "[plugin-result]"


def test_chat_memory_transition_hook_can_suppress_notice(monkeypatch, tmp_path: Path) -> None:
    """The on_chat_memory_transition consult hook can suppress the default notice."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    # Set a very low chat memory cap so the transition fires immediately.
    config.harness.memory.max_chat_memory_chars = 1
    harness = ConversationHarness(config)

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        return AcpPromptResult(reply="ok", session_id="s1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    class TransitionPlugin(_BaseTestPlugin):
        def on_chat_memory_transition(
            self, context: MemoryTransitionContext
        ) -> MemoryTransitionContext:
            context.suppress_default = True
            return context

    _install_plugin(harness, TransitionPlugin())

    result = harness.process("chat-1", "hello")
    assert result.notice is None or "budget" not in result.notice


def test_after_first_prompt_built_injects_first_prompt(monkeypatch, tmp_path: Path) -> None:
    """The first-prompt consult hook can modify only the initial prompt."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    captured: list[str] = []

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        captured.append(prompt)
        return AcpPromptResult(reply="ok", session_id="s1")

    def fake_send_message(session_id: str, prompt: str, **kwargs: Any) -> AcpPromptResult:
        captured.append(prompt)
        return AcpPromptResult(reply="follow-up")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)

    class FirstPromptPlugin(_BaseTestPlugin):
        def after_first_prompt_built(self, context: PromptContext) -> PromptContext:
            context.prompt += "\n\n[ONLY-FIRST-PROMPT]"
            return context

    _install_plugin(harness, FirstPromptPlugin())

    harness.process("chat-1", "hello")
    harness.process("chat-1", "again")
    assert "[ONLY-FIRST-PROMPT]" in captured[0]
    assert len(captured) == 2
    assert "[ONLY-FIRST-PROMPT]" not in captured[1]


def test_skill_hooks_fire_and_modify_command(monkeypatch, tmp_path: Path) -> None:
    """Skill enable/disable consult and notify hooks fire."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        return AcpPromptResult(reply="ok", session_id="s1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    captured: dict[str, Any] = {}

    class SkillPlugin(_BaseTestPlugin):
        def before_skill_enabled(self, context: SkillCommandContext) -> SkillCommandContext:
            context.skill_name = f"plugin-{context.skill_name}"
            captured["before_enable"] = context
            return context

        def after_skill_enabled(self, context: SkillCommandContext) -> None:
            captured["after_enable"] = context

        def before_skill_disabled(self, context: SkillCommandContext) -> SkillCommandContext:
            captured["before_disable"] = context
            return context

        def after_skill_disabled(self, context: SkillCommandContext) -> None:
            captured["after_disable"] = context

    _install_plugin(harness, SkillPlugin())

    harness.process("chat-1", "hello")
    harness.skill_enable("chat-1", "dummy")
    assert captured["before_enable"].skill_name == "plugin-dummy"
    assert "after_enable" in captured

    harness.skill_disable("chat-1", "dummy")
    assert captured["before_disable"].skill_name == "dummy"
    assert "after_disable" in captured


def test_mcp_hooks_fire_and_modify_command(monkeypatch, tmp_path: Path) -> None:
    """MCP enable/disable consult and notify hooks fire."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        return AcpPromptResult(reply="ok", session_id="s1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    captured: dict[str, Any] = {}

    class McpPlugin(_BaseTestPlugin):
        def before_mcp_enabled(self, context: McpCommandContext) -> McpCommandContext:
            context.server_name = f"plugin-{context.server_name}"
            captured["before_enable"] = context
            return context

        def after_mcp_enabled(self, context: McpCommandContext) -> None:
            captured["after_enable"] = context

        def before_mcp_disabled(self, context: McpCommandContext) -> McpCommandContext:
            captured["before_disable"] = context
            return context

        def after_mcp_disabled(self, context: McpCommandContext) -> None:
            captured["after_disable"] = context

    _install_plugin(harness, McpPlugin())

    harness.process("chat-1", "hello")
    harness.mcp_enable("chat-1", "dummy")
    assert captured["before_enable"].server_name == "plugin-dummy"
    assert "after_enable" in captured

    harness.mcp_disable("chat-1", "plugin-dummy")
    assert captured["before_disable"].server_name == "plugin-dummy"
    assert "after_disable" in captured


def test_retain_and_promote_hooks_fire(monkeypatch, tmp_path: Path) -> None:
    """Retain and promote consult/notify hooks fire and can modify inputs."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        return AcpPromptResult(reply="ok", session_id="s1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(
        "diploid_agent.memory.MemoryManager.promote_to_persona",
        lambda self, fact: None,
    )

    captured: dict[str, Any] = {}

    class MemoryCommandPlugin(_BaseTestPlugin):
        def before_retain(self, context: RetainContext) -> RetainContext:
            context.content = f"[retained] {context.content}"
            captured["before_retain"] = context
            return context

        def after_retain(self, context: RetainContext) -> None:
            captured["after_retain"] = context

        def before_promote(self, context: PromoteContext) -> PromoteContext:
            context.fact = f"[promoted] {context.fact}"
            captured["before_promote"] = context
            return context

        def after_promote(self, context: PromoteContext) -> None:
            captured["after_promote"] = context

    _install_plugin(harness, MemoryCommandPlugin())

    harness.process("chat-1", "hello")
    harness.retain("chat-1", "hello", tags=["greeting"])
    assert captured["before_retain"].content == "[retained] hello"
    assert "after_retain" in captured

    harness.promote("chat-1", "hello")
    assert captured["before_promote"].fact == "[promoted] hello"
    assert "after_promote" in captured


def test_on_shutdown_notifies_plugin(monkeypatch, tmp_path: Path) -> None:
    """The on_shutdown notify hook fires for each chat."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        return AcpPromptResult(reply="ok", session_id="s1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    captured: list[ShutdownContext] = []

    class ShutdownPlugin(_BaseTestPlugin):
        def on_shutdown(self, context: ShutdownContext) -> None:
            captured.append(context)

    _install_plugin(harness, ShutdownPlugin())

    harness.process("chat-1", "hello")
    harness.shutdown()

    assert any(c.chat_id == "chat-1" for c in captured)
    assert captured[0].reason == "shutdown"


def test_plugin_disable_hides_from_prompt(monkeypatch, tmp_path: Path) -> None:
    """A disabled plugin's prompt block is excluded for this chat."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    config.harness.plugins = [
        PluginConfig(name="visible", enabled=True, prompt_slot="persona_state"),
        PluginConfig(name="hidden", enabled=True, prompt_slot="persona_state"),
    ]
    harness = ConversationHarness(config)

    captured: list[str] = []

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        captured.append(prompt)
        return AcpPromptResult(reply="ok", session_id="s1")

    def fake_send_message(session_id: str, prompt: str, **kwargs: Any) -> AcpPromptResult:
        captured.append(prompt)
        return AcpPromptResult(reply="ok", session_id=session_id)

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)

    class VisiblePromptPlugin(StatePlugin):
        def __init__(self, config: PluginConfig, chat_id: str, sessions_root: Any) -> None:
            super().__init__(config, chat_id, sessions_root)

        def prompt_block(self, max_chars: int | None = None) -> str | None:
            return f"[{self.config.name}]"

    visible_cfg = PluginConfig(name="visible", enabled=True, prompt_slot="persona_state")
    hidden_cfg = PluginConfig(name="hidden", enabled=True, prompt_slot="persona_state")
    harness._plugins._plugins = [visible_cfg, hidden_cfg]

    def _plugins_for(chat_id: str) -> list[StatePlugin]:
        plugins = []
        for cfg in harness._plugins._plugins:
            if harness._plugins._is_enabled_for(chat_id, cfg) and cfg.name in ("visible", "hidden"):
                plugins.append(VisiblePromptPlugin(cfg, chat_id, tmp_path))
        return plugins

    harness._plugins._plugins_for = _plugins_for  # type: ignore[assignment]
    harness._plugins._instances.clear()

    harness.process("chat-1", "hello")
    assert "[visible]" in captured[0]
    assert "[hidden]" in captured[0]

    harness.plugin_set_enabled("chat-1", "hidden", False)
    harness._plugins._instances.clear()
    captured.clear()

    harness.new_session("chat-1")
    harness.process("chat-1", "again")
    assert "[visible]" in captured[0]
    assert "[hidden]" not in captured[0]


def test_plugin_list_shows_status(monkeypatch, tmp_path: Path) -> None:
    """The plugin list reflects enabled/disabled/failed state."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    config.harness.plugins = [
        PluginConfig(name="ok", enabled=True),
        PluginConfig(name="off", enabled=False),
        PluginConfig(name="broken", enabled=True, module="no_such_module"),
    ]
    harness = ConversationHarness(config)

    def fake_create_session(prompt: str, **kwargs: Any) -> AcpPromptResult:
        return AcpPromptResult(reply="ok", session_id="s1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    harness.process("chat-1", "hello")
    status = harness.plugin_list("chat-1")
    by_name = {s["name"]: s for s in status}
    assert by_name["ok"]["enabled"] is True
    assert by_name["ok"]["failed"] is False
    assert by_name["off"]["enabled"] is False
    # Startup validation disables plugins whose module cannot be loaded.
    assert by_name["broken"]["enabled"] is False
    assert by_name["broken"]["failed"] is False

    harness.plugin_set_enabled("chat-1", "ok", False)
    status = harness.plugin_list("chat-1")
    by_name = {s["name"]: s for s in status}
    assert by_name["ok"]["enabled"] is False
