"""Tests for the AgentRuntime / TurnController split.

These tests verify that the runtime package exposes the same behavior as the
legacy ConversationHarness by delegating turn logic to TurnController while
keeping the service container and non-turn API on AgentRuntime.
"""

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    PersonaConfig,
    PlanConfig,
    Secrets,
)
from diploid_agent.engine.base import AgentEngine, TurnRequest, TurnResult
from diploid_agent.harness import ConversationHarness
from diploid_agent.runtime import AgentRuntime, TurnController
from diploid_agent.transport.base import RuntimeAPI


def _make_config(tmp_path: Path) -> Config:
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
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


class _ChunkingEngine(AgentEngine):
    """Fake engine that emits a chunk from a background thread.

    If the TurnController holds the runtime RLock while the engine prompt
    runs, the on_chunk callback will deadlock. This only completes when the
    lock is correctly released during the engine call.
    """

    def prompt(
        self,
        request: TurnRequest,
        *,
        session_id: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> TurnResult:
        def _emit() -> None:
            time.sleep(0.05)
            if on_chunk is not None:
                on_chunk("chunk")

        thread = threading.Thread(target=_emit)
        thread.start()
        thread.join()
        return TurnResult(reply="ok", session_id="s-1")

    def cancel(self, session_id: str) -> None:
        pass

    def list_models(self) -> list[str]:
        return ["swe-1-7"]

    def session_alive(self, session_id: str) -> bool:
        return True

    def close(self) -> None:
        pass

    def restart(self) -> None:
        pass

    def restart_transport(self) -> None:
        pass

    def is_stale_session_error(self, exc: BaseException) -> bool:
        return False

    def model_context_window(self, model: str) -> int | None:
        return None


def test_process_releases_runtime_lock_for_streaming_chunks(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = _ChunkingEngine()

    start = time.perf_counter()
    result = runtime.process("chat-1", "hello")
    elapsed = time.perf_counter() - start

    assert result.reply == "ok"
    assert elapsed < 1.0
