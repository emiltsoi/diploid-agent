"""Telegram command handler mixin for the long-polling transport.

This mixin holds the ``_harness_*`` command handler methods used by
``TelegramPoller``. It is not intended to be used on its own; it expects the
host class to provide attributes such as ``command_handler``, ``client``,
``_send_text``, ``_send_result``, ``_typing_context``, ``runtime``,
``_api_key``, ``_worker_lock``, and ``_active_workers``.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from diploid_agent.config import (
    ConfigPersistenceError,
    NotificationsConfig,
    TaskConfig,
    TelegramConfig,
    TimerConfig,
    WakerConfig,
)
from diploid_agent.models import ChatResult
from diploid_agent.transport.command_handler import _coerce_chat_result
from diploid_agent.transport.telegram.formatting import (
    _TELEGRAM_HELP,
    _format_subagent_time,
)

logger = logging.getLogger("telegram_poll")


class TelegramCommandMixin:
    def _harness_reply(self, chat_id: int, message: str) -> dict[str, Any]:
        try:
            with self._typing_context(chat_id):
                resp = self.client.post(
                    f"{self.harness_url}/chat",
                    json={"chat_id": str(chat_id), "message": message},
                    timeout=self.reply_timeout,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception:
            logger.exception("Harness /chat failed")
            return {
                "reply": "Sorry, the harness is having trouble. Try again in a moment.",
                "notice": None,
            }

    def _harness_metrics(self, chat_id: int) -> str:
        raw = self.command_handler.call(
            method="get_metrics",
            chat_id=chat_id,
            http_path="/metrics/{chat_id}",
            http_method="GET",
        )
        if not isinstance(raw, dict) or "error" in raw:
            return "Sorry, I could not fetch your chat metrics."

        data = raw
        cumulative = data.get("cumulative") or {}
        last_turn = data.get("last_turn")
        if not cumulative:
            return "No metrics for this chat yet."
        lines = [
            f"Turns: {cumulative.get('turns', 0)}",
            (
                f"Tokens: {cumulative.get('total_tokens', 0)} total "
                f"({cumulative.get('input_tokens', 0)} in / {cumulative.get('output_tokens', 0)} out)"
            ),
        ]
        if cumulative.get("cached_tokens"):
            lines.append(f"Cached tokens: {cumulative['cached_tokens']}")
        lines.append(f"Latency: {cumulative.get('latency_seconds', 0.0):.2f}s")
        if last_turn:
            lines.append(
                f"Last turn: #{last_turn.get('turn_number', '?')} "
                f"({last_turn.get('model', '?')}) — "
                f"{last_turn.get('total_tokens', 0)} tokens in "
                f"{last_turn.get('latency_seconds', 0.0):.2f}s"
            )
        return "\n".join(lines)

    def _harness_config(self, chat_id: int, arg: str) -> str:
        """Handle /config <section> <key>=<value> [...]."""
        parts = arg.split(None, 1)
        if len(parts) < 2:
            return (
                "Usage: /config <section> <key>=<value> [key=value...]\n"
                "Sections: task, waker, timer, notifications, telegram"
            )

        section, rest = parts
        section = section.lower()
        section_map: dict[str, tuple[type, str]] = {
            "task": (TaskConfig, "update_task_config"),
            "waker": (WakerConfig, "update_waker_config"),
            "timer": (TimerConfig, "update_timer_config"),
            "notifications": (NotificationsConfig, "update_notifications_config"),
            "telegram": (TelegramConfig, "update_telegram_config"),
        }

        if section not in section_map:
            return (
                f"Unknown config section: {section}. Use task|waker|timer|notifications|telegram."
            )

        model_cls, update_method = section_map[section]

        fields: dict[str, Any] = {}
        for pair in rest.split():
            if "=" not in pair:
                return f"Invalid pair (expected key=value): {pair}"
            key, value = pair.split("=", 1)
            fields[key] = self._parse_config_value(value)

        try:
            cfg = model_cls(**fields)
        except (ValueError, TypeError) as exc:
            return f"Invalid {section} config: {exc}"

        if self.runtime is not None:
            try:
                updater = getattr(self.runtime, update_method)
                updater(cfg)
                getter = getattr(self.runtime, f"get_{section}_config")
                current = getter()
                return json.dumps(current.model_dump(), indent=2, default=str)
            except ConfigPersistenceError as exc:
                return f"Updated in memory, but could not persist: {exc}"
            except (AttributeError, ValueError, TypeError) as exc:
                logger.exception("Runtime config update failed")
                return f"Sorry, I could not update {section} config: {exc}"

        if self.harness_url is None:
            return "No runtime or harness URL configured."

        headers = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key

        try:
            resp = self.client.post(
                f"{self.harness_url}/{section}/config",
                json=fields,
                headers=headers,
                timeout=30.0,
            )
            resp.raise_for_status()
            return json.dumps(resp.json(), indent=2, sort_keys=False)
        except httpx.HTTPStatusError as exc:
            return f"Harness returned {exc.response.status_code}: {exc.response.text}"
        except (httpx.HTTPError, OSError):
            logger.exception("Harness /%s/config failed", section)
            return f"Sorry, I could not update {section} config via the harness."

    def _harness_mcp_list(self, chat_id: int) -> str:
        raw = self.command_handler.call(
            method="mcp_list",
            chat_id=chat_id,
            http_path="/mcp/{chat_id}",
            http_method="GET",
        )
        if isinstance(raw, str):
            return raw
        if not isinstance(raw, dict) or "error" in raw:
            return "Sorry, I could not fetch the MCP server list."
        return _coerce_chat_result(raw).reply

    def _harness_mcp_enable(self, chat_id: int, name: str) -> str:
        raw = self.command_handler.call(
            method="mcp_enable",
            chat_id=chat_id,
            http_path="/mcp",
            name=name,
            http_body={"command": "enable", "name": name},
        )
        if isinstance(raw, str):
            return raw
        if not isinstance(raw, dict) or "error" in raw:
            return f"Sorry, I could not enable {name}."
        return _coerce_chat_result(raw).reply

    def _harness_mcp_disable(self, chat_id: int, name: str) -> str:
        raw = self.command_handler.call(
            method="mcp_disable",
            chat_id=chat_id,
            http_path="/mcp",
            name=name,
            http_body={"command": "disable", "name": name},
        )
        if isinstance(raw, str):
            return raw
        if not isinstance(raw, dict) or "error" in raw:
            return f"Sorry, I could not disable {name}."
        return _coerce_chat_result(raw).reply

    def _harness_skill_list(self, chat_id: int) -> str:
        raw = self.command_handler.call(
            method="skill_list",
            chat_id=chat_id,
            http_path="/skill/{chat_id}",
            http_method="GET",
        )
        if isinstance(raw, str):
            return raw
        if not isinstance(raw, dict) or "error" in raw:
            return "Sorry, I could not fetch the skill list."
        return _coerce_chat_result(raw).reply

    def _harness_skill_enable(self, chat_id: int, name: str) -> str:
        raw = self.command_handler.call(
            method="skill_enable",
            chat_id=chat_id,
            http_path="/skill",
            name=name,
            http_body={"command": "enable", "name": name},
        )
        if isinstance(raw, str):
            return raw
        if not isinstance(raw, dict) or "error" in raw:
            return f"Sorry, I could not enable {name}."
        return _coerce_chat_result(raw).reply

    def _harness_skill_disable(self, chat_id: int, name: str) -> str:
        raw = self.command_handler.call(
            method="skill_disable",
            chat_id=chat_id,
            http_path="/skill",
            name=name,
            http_body={"command": "disable", "name": name},
        )
        if isinstance(raw, str):
            return raw
        if not isinstance(raw, dict) or "error" in raw:
            return f"Sorry, I could not disable {name}."
        return _coerce_chat_result(raw).reply

    def _harness_skill_create(self, chat_id: int, name: str, content: str) -> str:
        raw = self.command_handler.call(
            method="skill_create",
            chat_id=chat_id,
            http_path="/skill",
            name=name,
            content=content,
            http_body={"command": "create", "name": name, "content": content},
        )
        if isinstance(raw, str):
            return raw
        if not isinstance(raw, dict) or "error" in raw:
            return f"Sorry, I could not create {name}."
        return _coerce_chat_result(raw).reply

    def _harness_plugin_list(self, chat_id: int) -> str:
        raw = self.command_handler.call(
            method="plugin_list",
            chat_id=chat_id,
            http_path="/plugins/{chat_id}",
            http_method="GET",
        )
        if isinstance(raw, dict) and "error" in raw:
            return "Sorry, I could not fetch the plugin list."
        if isinstance(raw, list):
            plugins = raw
        else:
            plugins = raw.get("plugins", []) if isinstance(raw, dict) else []
        return json.dumps(plugins, default=str, indent=2)

    def _harness_plugin_enable(self, chat_id: int, name: str, enabled: bool) -> str:
        raw = self.command_handler.call(
            method="plugin_set_enabled",
            chat_id=chat_id,
            http_path="/plugin/enable",
            name=name,
            enabled=enabled,
        )
        if isinstance(raw, dict) and "error" in raw:
            return f"Sorry, I could not {'enable' if enabled else 'disable'} {name}."
        return _coerce_chat_result(raw).reply

    def _harness_plugin_reload(self, chat_id: int, name: str) -> str:
        raw = self.command_handler.call(
            method="plugin_reload",
            chat_id=chat_id,
            http_path="/plugin/reload",
            name=name,
        )
        if isinstance(raw, dict) and "error" in raw:
            return f"Sorry, I could not reload {name}."
        return _coerce_chat_result(raw).reply

    def _harness_status(self, chat_id: int) -> str:
        raw = self.command_handler.call(
            method="status",
            chat_id=chat_id,
            http_path="/status/{chat_id}",
            http_method="GET",
        )
        if not isinstance(raw, dict) or "error" in raw:
            return "Sorry, I could not fetch your chat status."

        data = raw
        if not data.get("active"):
            return "No active session for this chat yet."
        lines = [
            f"Persona: {data.get('persona')}",
            f"Model: {data.get('model')}",
            f"Session: {data.get('session_id')}",
            f"Working directory: {data.get('cwd')}",
        ]
        active_turn = data.get("active_turn") or {}
        if active_turn.get("status") == "running":
            elapsed = active_turn.get("elapsed_seconds", 0)
            user_message = active_turn.get("user_message", "")
            lines.append(
                f"Turn: running for {elapsed}s on \"{user_message}\""
            )
        else:
            lines.append("Turn: idle")

        ctx = data.get("context_usage") or {}
        context_window = ctx.get("context_window")
        if context_window:
            lines.append(f"Context window: {context_window} tokens")
            last_turn = ctx.get("last_turn") or {}
            if last_turn.get("input_percent") is not None:
                stop_reason = last_turn.get("stop_reason")
                stop_note = f" (stopped: {stop_reason})" if stop_reason else ""
                lines.append(
                    f"Last turn: {last_turn['input_percent']}% input, "
                    f"{last_turn.get('total_percent', 0)}% total "
                    f"({last_turn.get('available_tokens', 0)} available){stop_note}"
                )
            cumulative = ctx.get("cumulative") or {}
            if cumulative.get("turns") is not None:
                lines.append(
                    f"Cumulative: {cumulative.get('turns', 0)} turns, "
                    f"{cumulative.get('total_tokens', 0)} tokens"
                )
        return "\n".join(lines)

    def _harness_memory(self, chat_id: int) -> str:
        raw = self.command_handler.call(
            method="memory",
            chat_id=chat_id,
            http_path="/memory/{chat_id}",
            http_method="GET",
        )
        if isinstance(raw, str):
            return raw or ""
        if not isinstance(raw, dict) or "error" in raw:
            return "Sorry, I could not fetch your chat memory."

        return raw.get("memory", "") or ""

    def _harness_summarize(self, chat_id: int) -> dict[str, Any]:
        raw = self.command_handler.call(
            method="summarize",
            chat_id=chat_id,
            http_path="/summarize/{chat_id}",
        )
        if isinstance(raw, dict) and "error" in raw:
            return {
                "reply": "Sorry, I could not summarize the conversation.",
                "notice": None,
            }
        return _coerce_chat_result(raw).to_dict()

    def _harness_recall(self, chat_id: int, query: str, tags: list[str] | None = None) -> str:
        raw = self.command_handler.call(
            method="recall",
            chat_id=chat_id,
            http_path="/recall",
            query=query,
            tags=tags or [],
        )
        if isinstance(raw, dict) and "error" in raw:
            return "Sorry, I could not recall anything."
        return _coerce_chat_result(raw).reply

    def _harness_promote(self, chat_id: int, fact: str) -> dict[str, Any]:
        raw = self.command_handler.call(
            method="promote",
            chat_id=chat_id,
            http_path="/promote",
            fact=fact,
            http_body={"message": fact},
        )
        if isinstance(raw, dict) and "error" in raw:
            return {
                "reply": "Sorry, I could not promote that to persona memory.",
                "notice": None,
            }
        return _coerce_chat_result(raw).to_dict()

    def _harness_models(self) -> str:
        raw = self.command_handler.call(
            method="list_models",
            http_path="/models",
            http_method="GET",
            requires_chat_id=False,
        )
        if not isinstance(raw, (dict, list)) or (isinstance(raw, dict) and "error" in raw):
            return "Sorry, I could not fetch the model list."

        if isinstance(raw, dict):
            models = raw.get("models", [])
        else:
            models = raw
        # Telegram message limit is 4096 chars; truncate list if needed.
        text = ", ".join(models)
        if len(text) > 4000:
            text = text[:3997] + "..."
        return f"Available models:\n{text}"

    def _harness_switch_model(self, chat_id: int, model: str) -> dict[str, Any]:
        raw = self.command_handler.call(
            method="switch_model",
            chat_id=chat_id,
            http_path="/switch-model",
            model=model,
        )
        if isinstance(raw, dict) and "error" in raw:
            return {
                "reply": f"Sorry, I could not switch to model `{model}`.",
                "notice": None,
            }
        return _coerce_chat_result(raw).to_dict()

    def _harness_new(self, chat_id: int) -> dict[str, Any]:
        raw = self.command_handler.call(
            method="new_session",
            chat_id=chat_id,
            http_path="/new/{chat_id}",
        )
        if isinstance(raw, dict) and "error" in raw:
            return {
                "reply": "Sorry, I could not start a new session.",
                "notice": None,
            }
        return _coerce_chat_result(raw).to_dict()

    def _harness_stop(self, chat_id: int) -> dict[str, Any]:
        raw = self.command_handler.call(
            method="stop",
            chat_id=chat_id,
            http_path="/stop",
        )
        if isinstance(raw, dict) and "error" in raw:
            return {
                "reply": "Sorry, I could not stop the current turn.",
                "notice": None,
            }
        return _coerce_chat_result(raw).to_dict()

    def _harness_restart(self, chat_id: int) -> dict[str, Any]:
        raw = self.command_handler.call(
            method="restart",
            chat_id=chat_id,
            http_path="/restart",
        )
        if isinstance(raw, dict) and "error" in raw:
            return {
                "reply": "Sorry, I could not restart the ACP transport.",
                "notice": None,
            }
        return _coerce_chat_result(raw).to_dict()

    def _harness_graceful_restart(
        self,
        chat_id: int,
        service: str,
    ) -> dict[str, Any]:
        raw = self.command_handler.call(
            method="graceful_service_restart",
            chat_id=chat_id,
            http_path="/graceful-restart",
            service=service,
            reason="telegram command",
        )
        if isinstance(raw, dict) and "error" in raw:
            return {
                "reply": "Sorry, I could not schedule a graceful restart.",
                "notice": None,
            }
        return _coerce_chat_result(raw).to_dict()

    def _harness_state_event(
        self,
        chat_id: int,
        plugin: str,
        event: str,
        raw_args: str | None,
    ) -> str:
        http_body = {"plugin": plugin, "event": event}
        if raw_args:
            http_body["params"] = {"raw_args": raw_args}
        raw = self.command_handler.call(
            method="plugin_event",
            chat_id=chat_id,
            http_path="/state",
            plugin=plugin,
            event=event,
            raw_args=raw_args,
            http_body=http_body,
        )
        if isinstance(raw, dict) and "error" in raw:
            return "Sorry, I could not dispatch that state event."
        return _coerce_chat_result(raw).reply

    def _harness_sessions(self, chat_id: int) -> str:
        raw = self.command_handler.call(
            method="list_sessions",
            chat_id=chat_id,
            http_path="/sessions/{chat_id}",
            http_method="GET",
        )
        if not isinstance(raw, dict) or "error" in raw:
            return "Sorry, I could not list sessions."

        data = raw
        lines: list[str] = []
        for session in data.get("sessions", []):
            prefix = "* " if session.get("is_active") else "  "
            lines.append(
                f"{prefix}{session['number']}. {session.get('label') or 'session'} "
                f"({session['model']})"
            )
        return "Sessions:\n" + "\n".join(lines)

    def _harness_resume(self, chat_id: int, session_number: int) -> dict[str, Any]:
        raw = self.command_handler.call(
            method="resume_session",
            chat_id=chat_id,
            http_path="/resume",
            session_number=session_number,
        )
        if isinstance(raw, dict) and "error" in raw:
            return {
                "reply": "Sorry, I could not resume that session.",
                "notice": None,
            }
        return _coerce_chat_result(raw).to_dict()

    def _harness_branch(self, chat_id: int, session_number: int) -> dict[str, Any]:
        raw = self.command_handler.call(
            method="branch_session",
            chat_id=chat_id,
            http_path="/branch",
            session_number=session_number,
        )
        if isinstance(raw, dict) and "error" in raw:
            return {
                "reply": "Sorry, I could not branch that session.",
                "notice": None,
            }
        return _coerce_chat_result(raw).to_dict()

    def _harness_subagent(self, chat_id: int, prompt: str) -> ChatResult:
        return self.command_handler.handle("/subagent", chat_id=chat_id, arg=prompt)

    def _harness_subagent_status(self, chat_id: int) -> str:
        raw = self.command_handler.call(
            method="subagent_status",
            chat_id=chat_id,
            http_path="/subagents/{chat_id}",
            http_method="GET",
        )
        if not isinstance(raw, dict) or "error" in raw:
            return "Sorry, I could not fetch subagent status."

        subagents = raw.get("subagents", [])
        if not subagents:
            return "No background subagents for this chat."

        lines: list[str] = []
        for sa in subagents:
            status = sa.get("status", "unknown")
            dispatch_id = sa.get("dispatch_id") or sa.get("task_id") or "?"
            summary = sa.get("summary")
            started_at = sa.get("started_at")
            finished_at = sa.get("finished_at")

            parts = [f"{status}: {dispatch_id}"]
            if summary:
                parts.append(f"— {summary}")

            started = _format_subagent_time(started_at)
            finished = _format_subagent_time(finished_at)
            if started_at is not None and finished_at is not None:
                parts.append(f"(started {started}, finished {finished})")
            elif started_at is not None:
                parts.append(f"(started {started})")
            else:
                parts.append("(not started)")

            lines.append(" ".join(parts))
        return "\n".join(lines)

    def _harness_help(self, chat_id: int) -> str:
        return _TELEGRAM_HELP

