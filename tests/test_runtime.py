"""Tests for the AgentRuntime / TurnController split.

These tests verify that the runtime package exposes the same behavior as the
legacy ConversationHarness by delegating turn logic to TurnController while
keeping the service container and non-turn API on AgentRuntime.
"""

from pathlib import Path

from devin_fleet_harness.config import (
    Config,
    DevinConfig,
    HarnessConfig,
    PersonaConfig,
    PlanConfig,
    Secrets,
)
from devin_fleet_harness.harness import ConversationHarness
from devin_fleet_harness.runtime import AgentRuntime, TurnController
from devin_fleet_harness.transport.base import RuntimeAPI


def _make_config(tmp_path: Path) -> Config:
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    return Config(
        devin=DevinConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=fixture_root,
            fleet_root=tmp_path / "fleet",
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            plan=PlanConfig(root=tmp_path / "plans"),
            memory={"backend": "file"},  # type: ignore[arg-type]
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


def test_agent_runtime_implements_runtime_api(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    assert isinstance(runtime, RuntimeAPI)


def test_agent_runtime_has_engine_and_turn_controller(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    assert runtime.engine is not None
    assert isinstance(runtime.turn_controller, TurnController)


def test_agent_runtime_client_property_aliased_engine(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    assert runtime.client is runtime.engine
    fake = object()
    runtime.client = fake  # type: ignore[assignment]
    assert runtime.engine is fake


def test_agent_runtime_non_turn_public_api(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    for name in (
        "mcp_list",
        "mcp_enable",
        "mcp_disable",
        "skill_list",
        "skill_enable",
        "skill_disable",
        "skill_create",
        "get_metrics",
        "list_models",
        "get_model",
        "status",
        "list_sessions",
        "memory",
        "summarize",
        "recall",
        "retain",
        "promote",
    ):
        assert hasattr(runtime, name)
        assert callable(getattr(runtime, name))


def test_turn_controller_delegates_to_runtime_for_services(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    tc = runtime.turn_controller
    assert tc.runtime is runtime
    assert tc.runtime.config is runtime.config
    assert tc.runtime.engine is runtime.engine
    assert tc.runtime._lock is runtime._lock


def test_runtime_status_plan_active(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    status = runtime.get_status()
    assert status.plan_active is False
    assert status.active_plans == []


def test_conversation_harness_delegates_to_runtime(tmp_path: Path) -> None:
    harness = ConversationHarness(_make_config(tmp_path))
    assert isinstance(harness.runtime, AgentRuntime)
    assert isinstance(harness.turn_controller, TurnController)

    # RuntimeAPI turn methods should be present and delegate to the controller.
    for name in (
        "process",
        "continue_turn",
        "stop",
        "turn_status",
        "dispatch",
        "switch_model",
        "new_session",
        "resume_session",
        "branch_session",
    ):
        assert hasattr(harness, name)
        assert getattr(harness, name).__func__ is getattr(harness.runtime, name).__func__

    # Private helpers that live on the runtime are accessible via __getattr__.
    for name in (
        "is_continuation_message",
        "_build_first_prompt",
        "_memory_manager",
        "_active_record",
        "_chat_dir",
        "_telegram_message_registry_path",
    ):
        assert hasattr(harness, name)
