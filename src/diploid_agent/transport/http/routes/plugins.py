from __future__ import annotations

import json
from collections.abc import Callable

from fastapi import Depends, FastAPI, HTTPException, status

from diploid_agent.config import (
    Config,
    PluginConfig,
)
from diploid_agent.models import ChatResult
from diploid_agent.plugins import PluginManager
from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.transport.command_handler import CommandHandler
from diploid_agent.transport.http.models import *
from diploid_agent.transport.http.utils import _to_response


def register_plugins(
    app: FastAPI,
    runtime: RuntimeAPI,
    command_handler: CommandHandler,
    config: Config,
    _require_api_key: Callable[[str | None], None],
) -> None:
    @app.get("/plugins/{chat_id}", response_model=PluginListResponse)
    def plugin_list(chat_id: str) -> PluginListResponse:
        raw = command_handler.call(
            method="plugin_list",
            chat_id=chat_id,
            http_path="/plugins/{chat_id}",
            http_method="GET",
            catch=False,
        )
        return PluginListResponse(plugins=raw if isinstance(raw, list) else [])

    @app.post(
        "/plugin/enable", response_model=ChatResponse, dependencies=[Depends(_require_api_key)]
    )
    def plugin_enable(req: PluginEnableRequest) -> ChatResponse:
        raw = command_handler.call(
            method="plugin_set_enabled",
            chat_id=req.chat_id,
            name=req.name,
            enabled=req.enabled,
            catch=False,
        )
        return _to_response(raw)

    @app.post(
        "/plugin/reload", response_model=ChatResponse, dependencies=[Depends(_require_api_key)]
    )
    def plugin_reload(req: PluginCommandRequest) -> ChatResponse:
        raw = command_handler.call(
            method="plugin_reload",
            chat_id=req.chat_id,
            name=req.name,
            catch=False,
        )
        return _to_response(raw)

    @app.post(
        "/plugins",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_add(req: PluginAddRequest) -> ChatResponse:
        config = PluginConfig(**req.plugin)
        if req.dry_run:
            try:
                PluginManager.validate_module(config.module)
                return _to_response(ChatResult(reply=f"Dry run OK for {config.name}"))
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
                ) from exc
        try:
            raw = command_handler.call(
                method="plugin_add",
                config=config,
                requires_chat_id=False,
                catch=False,
            )
            return _to_response(raw)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post(
        "/plugin/sandbox",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_sandbox(req: PluginSandboxRequest) -> ChatResponse:
        try:
            raw = command_handler.call(
                method="plugin_sandbox",
                module=req.module,
                plugin=req.plugin or {},
                requires_chat_id=False,
                catch=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return _to_response(ChatResult(reply=json.dumps(raw, ensure_ascii=False)))

    @app.post(
        "/plugins/create",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_create(req: PluginCreateRequest) -> ChatResponse:
        try:
            raw = command_handler.call(
                method="plugin_create",
                name=req.name,
                module=req.module,
                prompt_slot=req.prompt_slot,
                state_file=req.state_file,
                mcp_server=req.mcp_server,
                config=req.config,
                requires_chat_id=False,
                catch=False,
            )
            return _to_response(ChatResult(reply=json.dumps(raw, ensure_ascii=False)))
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.delete(
        "/plugins/{name}",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_remove(name: str) -> ChatResponse:
        try:
            raw = command_handler.call(
                method="plugin_remove",
                name=name,
                requires_chat_id=False,
                catch=False,
            )
            return _to_response(raw)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.patch(
        "/plugins/{name}",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_update(name: str, req: PluginUpdateRequest) -> ChatResponse:
        if req.name != name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Name mismatch")
        try:
            # Validate the merged config before mutating runtime state.
            merged = dict(req.plugin)
            merged["name"] = name
            config = PluginConfig(**merged)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        if req.dry_run:
            try:
                PluginManager.validate_module(config.module)
                return _to_response(ChatResult(reply=f"Dry run OK for {name}"))
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
                ) from exc
        # Validate the module before applying the update.
        try:
            PluginManager.validate_module(config.module)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        # Merge via the existing update_plugins_config path, then reload.
        patch = {"plugins": [{"name": name, **req.plugin}]}
        command_handler.call(
            method="update_config",
            patch=patch,
            requires_chat_id=False,
            catch=False,
        )
        command_handler.call(
            method="plugin_reload",
            name=name,
            chat_id="0",  # chat_id is ignored by reload for modules
            catch=False,
        )
        return _to_response(ChatResult(reply=f"Plugin {name} updated"))

    @app.post(
        "/plugins/{name}/toggle",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_toggle(name: str, req: PluginToggleRequest) -> ChatResponse:
        try:
            raw = command_handler.call(
                method="plugin_toggle",
                name=name,
                enabled=req.enabled,
                chat_id=req.chat_id,
                catch=False,
            )
            return _to_response(raw)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post(
        "/config/rollback",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def config_rollback(req: PluginRollbackRequest) -> ChatResponse:
        try:
            raw = command_handler.call(
                method="plugin_rollback",
                steps=req.steps,
                requires_chat_id=False,
                catch=False,
            )
            return _to_response(raw)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.get(
        "/plugin-incidents",
        response_model=PluginIncidentListResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_incidents() -> PluginIncidentListResponse:
        raw = command_handler.call(
            method="incidents",
            requires_chat_id=False,
            catch=False,
        )
        return PluginIncidentListResponse(incidents=raw if isinstance(raw, list) else [])

    @app.get(
        "/plugin-incidents/{plugin_name}",
        response_model=PluginIncidentListResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_incidents_for(plugin_name: str) -> PluginIncidentListResponse:
        return PluginIncidentListResponse(incidents=runtime.incidents_for_plugin(plugin_name))

    @app.post(
        "/plugin-incidents",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_incident_create(req: PluginIncidentCreateRequest) -> ChatResponse:
        runtime.record_incident(
            plugin=req.plugin,
            phase=req.phase,
            error=req.error,
            action=req.action,
            chat_id=req.chat_id,
        )
        return _to_response(ChatResult(reply="incident recorded"))
