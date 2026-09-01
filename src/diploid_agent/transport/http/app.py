"""HTTP transport for the conversational harness.

Provides a generic `/chat` endpoint and a Telegram-compatible `/webhook`.
The generic endpoint lets any caller (Telegram bot, other clients, curl) send a
message and receive a reply.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException, status

from diploid_agent.config import Config
from diploid_agent.runtime.agent_runtime import AgentRuntime
from diploid_agent.transport.base import OutboundMessage, RuntimeAPI, Transport
from diploid_agent.transport.command_handler import CommandHandler
from diploid_agent.transport.http.routes import (
    register_chat,
    register_config,
    register_health,
    register_mesh,
    register_models,
    register_plans,
    register_plugins,
    register_runtime,
    register_sessions,
    register_skills,
    register_state,
    register_webhook,
)


def create_app(config: Config, runtime: RuntimeAPI | None = None) -> FastAPI:
    runtime = runtime or AgentRuntime(config)
    command_handler = CommandHandler(runtime=runtime)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            runtime.start()
            yield
        finally:
            runtime.shutdown()

    app = FastAPI(title="diploid-agent", lifespan=lifespan)
    app.state.runtime = runtime
    # Backward-compatible alias used by existing tests.
    app.state.harness = runtime

    def _require_api_key(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ) -> None:
        """Require X-API-Key on POST endpoints when HARNESS_API_KEY is configured."""
        token = config.secrets.harness_api_key if config.secrets else None
        if token is None:
            return
        if x_api_key != token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or missing X-API-Key",
            )

    register_health(app, runtime, command_handler, config, _require_api_key)
    register_mesh(app, runtime, command_handler, config, _require_api_key)
    register_chat(app, runtime, command_handler, config, _require_api_key)
    register_sessions(app, runtime, command_handler, config, _require_api_key)
    register_models(app, runtime, command_handler, config, _require_api_key)
    register_skills(app, runtime, command_handler, config, _require_api_key)
    register_plugins(app, runtime, command_handler, config, _require_api_key)
    register_plans(app, runtime, command_handler, config, _require_api_key)
    register_runtime(app, runtime, command_handler, config, _require_api_key)
    register_state(app, runtime, command_handler, config, _require_api_key)
    register_config(app, runtime, command_handler, config, _require_api_key)
    register_webhook(app, runtime, command_handler, config, _require_api_key)

    return app


class HttpTransport(Transport):
    """HTTP transport exposing the harness as a FastAPI application."""

    def __init__(self, config: Config, runtime: RuntimeAPI | None = None) -> None:
        self._config = config
        self._runtime = runtime
        self._app: FastAPI | None = None
        self._server: uvicorn.Server | None = None

    def start(self, runtime: RuntimeAPI | None = None) -> None:
        if runtime is not None:
            self._runtime = runtime
        self._app = create_app(self._config, self._runtime)
        uvicorn.run(
            self._app,
            host=self._config.harness.listen_host,
            port=self._config.harness.listen_port,
        )

    def stop(self) -> None:
        """HTTP transport currently has no running background server to stop."""
        return

    def send(self, message: OutboundMessage) -> None:
        """HTTP transport does not send outbound messages; it returns None."""
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "harness.yaml",
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    transport = HttpTransport(config)
    transport.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
