"""Tests for the HTTP/Telegram ingress."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from acp_fleet_harness.acp_client import AcpPromptResult
from acp_fleet_harness.config import (
    Config,
    DevinConfig,
    HarnessConfig,
    PersonaConfig,
    PluginConfig,
    Secrets,
)
from acp_fleet_harness.engine import TurnResult
from acp_fleet_harness.models import ChatResult
from acp_fleet_harness.telegram_ingress import create_app


class FakeClient:
    def __init__(self, models: list[str] | None = None) -> None:
        self._models = models or ["swe-1-7"]

    def create_session(
        self, prompt: str, *, cwd: Path | None = None, model: str | None = None, **kwargs
    ) -> AcpPromptResult:
        return AcpPromptResult(reply="reply", session_id="session-1")

    def send_message(
        self,
        session_id: str,
        prompt: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        **kwargs,
    ) -> AcpPromptResult:
        return AcpPromptResult(reply="follow-up")

    def list_models(self) -> list[str]:
        return self._models

    def session_alive(self, session_id: str) -> bool:
        return False

    def cancel(self, session_id: str) -> None:
        return None

    def health(self) -> bool:
        return True

    def prompt(self, request, *, session_id=None, on_chunk=None, on_update=None):
        if session_id is None:
            result = self.create_session(
                request.prompt,
                cwd=request.cwd,
                model=request.model,
                soft_timeout=request.soft_timeout,
            )
        else:
            result = self.send_message(
                session_id,
                request.prompt,
                cwd=request.cwd,
                model=request.model,
                soft_timeout=request.soft_timeout,
            )
        if isinstance(result, AcpPromptResult):
            return TurnResult(
                reply=result.reply,
                session_id=result.session_id,
                stop_reason=result.stop_reason,
                usage=result.usage,
                cancelled=result.cancelled,
                partial=result.partial,
                timed_out=result.timed_out,
            )
        return result


def _test_config(tmp_path: Path) -> Config:
    return Config(
        devin=DevinConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=Path(__file__).parent / "fixtures" / "test-pilot",
            fleet_root=Path(__file__).parent / "fixtures" / "fleet",
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "test-sessions",
            session_store_path=tmp_path / "test-sessions.jsonl",
            memory={"backend": "file"},  # type: ignore[arg-type]
            skills={"shared_root": str(tmp_path / "shared")},  # type: ignore[arg-type]
            plugins=[
                PluginConfig(
                    name="curriculum",
                    module="acp_fleet_harness.plugins.curriculum",
                    prompt_slot="persona_state",
                    state_file="chat_curriculum.json",
                    max_prompt_chars=1024,
                ),
            ],  # type: ignore[arg-type]
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


def _test_config_with_auth(tmp_path: Path) -> Config:
    return Config(
        devin=DevinConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=Path(__file__).parent / "fixtures" / "test-pilot",
            fleet_root=Path(__file__).parent / "fixtures" / "fleet",
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "test-sessions",
            session_store_path=tmp_path / "test-sessions.jsonl",
            memory={"backend": "file"},  # type: ignore[arg-type]
            skills={"shared_root": str(tmp_path / "shared")},  # type: ignore[arg-type]
            plugins=[
                PluginConfig(
                    name="curriculum",
                    module="acp_fleet_harness.plugins.curriculum",
                    prompt_slot="persona_state",
                    state_file="chat_curriculum.json",
                    max_prompt_chars=1024,
                ),
            ],  # type: ignore[arg-type]
        ),
        secrets=Secrets(
            WINDSURF_API_KEY="test-key",
            HARNESS_API_KEY="harness-secret",
        ),
    )


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    app = create_app(_test_config(tmp_path))
    fake_client = FakeClient()
    monkeypatch.setattr(app.state.harness, "client", fake_client)
    return TestClient(app)


@pytest.fixture
def auth_client(monkeypatch, tmp_path: Path):
    app = create_app(_test_config_with_auth(tmp_path))
    fake_client = FakeClient()
    monkeypatch.setattr(app.state.harness, "client", fake_client)
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["uptime_seconds"] >= 0
    assert data["components"]["acp"]["healthy"] is True
    assert data["components"]["hindsight"]["healthy"] is True
    assert data["components"]["telegram"]["healthy"] is True


def test_models(client: TestClient) -> None:
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == {"models": ["swe-1-7"]}


def test_chat(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"chat_id": "chat-1", "message": "hello"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reply"]
    assert body["notice"] is None
    assert body.get("metrics") is not None


def test_chat_and_metrics_endpoints(client: TestClient, monkeypatch) -> None:
    """Metrics are returned by /chat and exposed via /metrics."""

    def fake_create_session(prompt, *, cwd=None, model=None, **kwargs):
        return AcpPromptResult(
            reply="Ready.",
            session_id="session-m",
            usage={"inputTokens": 100, "outputTokens": 50, "totalTokens": 150},
        )

    monkeypatch.setattr(client.app.state.harness.client, "create_session", fake_create_session)

    response = client.post("/chat", json={"chat_id": "chat-m", "message": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"]["input_tokens"] == 100
    assert body["metrics"]["output_tokens"] == 50
    assert body["metrics"]["total_tokens"] == 150

    response = client.get("/metrics/chat-m")
    assert response.status_code == 200
    body = response.json()
    assert body["cumulative"]["turns"] == 1
    assert body["cumulative"]["total_tokens"] == 150

    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["global"]["turns"] >= 1


def test_new_session(client: TestClient) -> None:
    client.post("/chat", json={"chat_id": "chat-2", "message": "hello"})
    response = client.post("/new/chat-2")
    assert response.status_code == 200
    assert "New session started" in response.json()["reply"]


def test_switch_model(client: TestClient) -> None:
    client.post("/chat", json={"chat_id": "chat-3", "message": "hello"})
    response = client.post(
        "/switch-model",
        json={"chat_id": "chat-3", "model": "glm-5-2"},
    )
    assert response.status_code == 200
    assert "glm-5-2" in response.json()["reply"]


def test_list_sessions(client: TestClient) -> None:
    client.post("/chat", json={"chat_id": "chat-4", "message": "hello"})
    client.post("/new/chat-4")
    response = client.get("/sessions/chat-4")
    assert response.status_code == 200
    body = response.json()
    assert len(body["sessions"]) == 2


def test_resume_session(client: TestClient) -> None:
    client.post("/chat", json={"chat_id": "chat-5", "message": "hello"})
    client.post("/new/chat-5")
    response = client.post(
        "/resume",
        json={"chat_id": "chat-5", "session_number": 1},
    )
    assert response.status_code == 200
    assert "Resumed session 1" in response.json()["reply"]


def test_branch_session(client: TestClient) -> None:
    client.post("/chat", json={"chat_id": "chat-6", "message": "hello"})
    client.post("/new/chat-6")
    response = client.post(
        "/branch",
        json={"chat_id": "chat-6", "session_number": 1},
    )
    assert response.status_code == 200
    assert "Branched" in response.json()["reply"]


def test_stop_session(client: TestClient) -> None:
    response = client.post("/stop", json={"chat_id": "chat-7"})
    assert response.status_code == 200
    assert "No running turn" in response.json()["reply"]


def test_chat_with_reply_to(client: TestClient, monkeypatch) -> None:
    """POST /chat forwards reply_to and reply_to_is_bot to the harness."""
    call: dict[str, Any] = {}

    def fake_process(
        chat_id: str,
        message: str,
        *,
        model: str | None = None,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        notify: bool = True,
    ) -> ChatResult:
        call.update(
            chat_id=chat_id,
            message=message,
            model=model,
            reply_to=reply_to,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
            notify=notify,
        )
        return ChatResult(reply="ok")

    monkeypatch.setattr(client.app.state.harness, "process", fake_process)

    response = client.post(
        "/chat",
        json={
            "chat_id": "chat-8",
            "message": "explain",
            "reply_to": "the earlier message",
            "reply_to_is_bot": True,
            "reply_to_message_id": 99,
        },
    )
    assert response.status_code == 200
    assert call["chat_id"] == "chat-8"
    assert call["message"] == "explain"
    assert call["reply_to"] == "the earlier message"
    assert call["reply_to_is_bot"] is True
    assert call["reply_to_message_id"] == 99
    assert call["notify"] is False


def test_webhook_with_reply_to(client: TestClient, monkeypatch) -> None:
    """POST /webhook extracts reply_to_message from a Telegram update."""
    call: dict[str, Any] = {}

    def fake_process(
        chat_id: str,
        message: str,
        *,
        model: str | None = None,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
    ) -> ChatResult:
        call.update(
            chat_id=chat_id,
            message=message,
            reply_to=reply_to,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
        )
        return ChatResult(reply="ok")

    monkeypatch.setattr(client.app.state.harness, "process", fake_process)

    payload = {
        "update_id": 1,
        "message": {
            "message_id": 42,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "can you expand?",
            "reply_to_message": {
                "message_id": 10,
                "from": {"id": 2, "is_bot": True},
                "text": "The first answer.",
            },
        },
    }

    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    assert call["chat_id"] == "12345"
    assert call["message"] == "can you expand?"
    assert call["reply_to"] == "The first answer."
    assert call["reply_to_is_bot"] is True
    assert call["reply_to_message_id"] == 10


def test_mcp_get_list(client):
    response = client.get("/mcp/mcp-test")
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "No MCP servers configured."


def test_skill_get_list(client):
    response = client.get("/skill/skill-test")
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "No skills available."


def test_state_event(client: TestClient) -> None:
    response = client.post(
        "/state",
        json={
            "chat_id": "chat-state",
            "plugin": "curriculum",
            "event": "set_target_language",
            "params": {"language": "Klingon"},
        },
    )
    assert response.status_code == 200
    assert "Klingon" in response.json()["reply"]

    # The plugin should have persisted state.
    chat_dir = client.app.state.harness._chat_dir("chat-state")
    state_path = chat_dir / "chat_curriculum.json"
    assert state_path.exists()


def test_dispatch_and_continue_endpoints(client: TestClient, monkeypatch) -> None:
    from acp_fleet_harness.acp_client import AcpPromptResult

    def fake_create_session(prompt, **kwargs):
        return AcpPromptResult(reply="Ready.", session_id="session-1")

    def fake_send_message(session_id, prompt, **kwargs):
        return AcpPromptResult(reply="Continued.")

    monkeypatch.setattr(client.app.state.harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(client.app.state.harness.client, "send_message", fake_send_message)

    # First turn.
    resp = client.post("/chat", json={"chat_id": "chat-1", "message": "Please dispatch"})
    assert resp.status_code == 200

    # Start dispatch.
    resp = client.post("/dispatch", json={"chat_id": "chat-1", "context": "demo"})
    assert resp.status_code == 200
    data = resp.json()
    assert "dispatch_id" in data
    dispatch_id = data["dispatch_id"]

    # Complete dispatch.
    resp = client.post("/continue", json={"dispatch_id": dispatch_id, "result": "Done"})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "Continued."


def test_post_without_token_rejected(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/chat",
        json={"chat_id": "chat-auth", "message": "hello"},
    )
    assert response.status_code == 403
    assert "X-API-Key" in response.json()["detail"]


def test_post_with_correct_token_accepted(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/chat",
        json={"chat_id": "chat-auth", "message": "hello"},
        headers={"X-API-Key": "harness-secret"},
    )
    assert response.status_code == 200
    assert response.json()["reply"]


def test_get_and_webhook_stay_open_with_auth(auth_client: TestClient, monkeypatch) -> None:
    call: dict[str, Any] = {}

    def fake_process(
        chat_id: str,
        message: str,
        *,
        model: str | None = None,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
    ) -> ChatResult:
        call.update(chat_id=chat_id, message=message)
        return ChatResult(reply="ok")

    monkeypatch.setattr(auth_client.app.state.harness, "process", fake_process)

    # Health is always open.
    response = auth_client.get("/health")
    assert response.status_code == 200

    # Telegram's /webhook must stay unauthenticated.
    payload = {
        "update_id": 1,
        "message": {
            "message_id": 1,
            "chat": {"id": 42, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "hello",
        },
    }
    response = auth_client.post("/webhook", json=payload)
    assert response.status_code == 200
    assert call["chat_id"] == "42"
    assert call["message"] == "hello"


def test_endpoints_work_without_auth(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"chat_id": "chat-no-auth", "message": "hello"},
    )
    assert response.status_code == 200
    assert response.json()["reply"]
