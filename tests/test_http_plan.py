"""End-to-end tests for the POST /plan endpoints."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from devin_fleet_harness.config import (
    Config,
    DevinConfig,
    HarnessConfig,
    PersonaConfig,
    PlanConfig,
    Secrets,
    TaskConfig,
)
from devin_fleet_harness.plan.models import TaskStatus
from devin_fleet_harness.runtime.agent_runtime import AgentRuntime
from devin_fleet_harness.transport.http import create_app


def _fixture_root() -> Path:
    return Path(__file__).parent / "fixtures" / "test-pilot"


def _make_config(tmp_path: Path) -> Config:
    return Config(
        devin=DevinConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=_fixture_root(),
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            plan=PlanConfig(root=tmp_path / "plans"),
            task=TaskConfig(workers=2, shell_timeout=5.0),
            memory={"backend": "file"},  # type: ignore[arg-type]
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


class FakeEngine:
    def prompt(self, *a, **k):
        from devin_fleet_harness.engine import TurnResult

        return TurnResult(reply="ok", session_id="s1")

    def list_models(self):
        return ["m1"]

    def restart(self):
        pass

    def is_stale_session_error(self, exc):
        return False

    def close(self):
        pass


def test_post_plan_create(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()
    app = create_app(_make_config(tmp_path), runtime)

    with TestClient(app) as client:
        resp = client.post(
            "/plan/create",
            json={
                "name": "test-plan",
                "description": "a test",
                "chat_id": "chat-1",
                "tasks": [{"name": "echo", "command": "echo hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-plan"
        assert data["chat_id"] == "chat-1"
        assert data["status"] == "active"
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["status"] == "ready"


def test_post_plan_create_acp_task_model(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()
    app = create_app(_make_config(tmp_path), runtime)

    with TestClient(app) as client:
        resp = client.post(
            "/plan/create",
            json={
                "name": "model-plan",
                "chat_id": "chat-1",
                "tasks": [
                    {"name": "default", "type": "acp", "command": "do the default"},
                    {
                        "name": "custom",
                        "type": "acp",
                        "command": "do the custom",
                        "acp_model": "glm-5-2",
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tasks"][0]["acp_model"] is None
        assert data["tasks"][1]["acp_model"] == "glm-5-2"


def test_post_plan_task_start_and_done(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()
    app = create_app(_make_config(tmp_path), runtime)

    with TestClient(app) as client:
        resp = client.post(
            "/plan/create",
            json={
                "name": "run",
                "tasks": [{"name": "echo", "command": "echo hello"}],
            },
        )
        assert resp.status_code == 200
        plan_id = resp.json()["id"]
        task_id = resp.json()["tasks"][0]["id"]

        start = client.post(
            "/plan/task/start",
            json={"plan_id": plan_id, "task_id": task_id},
        )
        assert start.status_code == 200
        assert start.json()["status"] == "running"

        # Wait for the shell task to finish.
        for _ in range(50):
            task = runtime.plan_manager.get_task(plan_id, task_id)
            if task is not None and task.status == TaskStatus.DONE:
                break
            time.sleep(0.05)
        assert task is not None
        assert task.status == TaskStatus.DONE

        done = client.post(
            "/plan/task/done",
            json={"plan_id": plan_id, "task_id": task_id, "result": "manual"},
        )
        assert done.status_code == 200
        assert done.json()["status"] == "done"
        assert done.json()["result"] == "manual"


def test_post_plan_task_start_without_task_id(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()
    app = create_app(_make_config(tmp_path), runtime)

    with TestClient(app) as client:
        resp = client.post(
            "/plan/create",
            json={"name": "auto", "tasks": [{"name": "noop", "type": "noop"}]},
        )
        plan_id = resp.json()["id"]

        start = client.post("/plan/task/start", json={"plan_id": plan_id})
        assert start.status_code == 200
        assert start.json()["status"] == "running"


def test_post_plan_task_done_missing_task(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()
    app = create_app(_make_config(tmp_path), runtime)

    with TestClient(app) as client:
        resp = client.post(
            "/plan/create",
            json={"name": "missing", "tasks": []},
        )
        plan_id = resp.json()["id"]

        resp = client.post(
            "/plan/task/done",
            json={"plan_id": plan_id, "task_id": "nope", "result": "x"},
        )
        assert resp.status_code == 400
