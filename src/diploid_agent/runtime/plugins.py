"""RuntimePlugins: plugin lifecycle, sandbox, and incident helpers."""

from __future__ import annotations

import functools
import json as _json
import shutil
import subprocess
import sys
from collections.abc import Callable
from typing import Any

from diploid_agent.config import PluginConfig
from diploid_agent.models import ChatResult


def _locked(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run a RuntimePlugins method under the runtime RLock."""

    @functools.wraps(method)
    def wrapper(self: RuntimePlugins, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class RuntimePlugins:
    """Plugin lifecycle, sandbox, and incident helpers for AgentRuntime."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @property
    def config(self) -> Any:
        return self._runtime.config

    @property
    def _lock(self) -> Any:
        return self._runtime._lock

    @property
    def _plugins(self) -> Any:
        return self._runtime._plugins

    @property
    def _incidents(self) -> Any:
        return self._runtime._incidents

    @property
    def _mcp_skills(self) -> Any:
        return self._runtime._mcp_skills

    @property
    def _chat_store(self) -> Any:
        return self._runtime._chat_store

    @property
    def _config_manager(self) -> Any:
        return self._runtime._config_manager

    @property
    def _runtime_metrics(self) -> Any:
        return self._runtime._runtime_metrics

    @property
    def _plugin_mcp_server_names(self) -> set[str]:
        return self._runtime._plugin_mcp_server_names

    @_plugin_mcp_server_names.setter
    def _plugin_mcp_server_names(self, value: set[str]) -> None:
        self._runtime._plugin_mcp_server_names = value

    @property
    def context_builder(self) -> Any:
        return self._runtime.context_builder

    def _register_plugin_mcp_servers(self) -> None:
        """Append plugin MCP server configs to the harness config before McpManager sees it."""
        active_plugin_servers = self._plugins.mcp_server_configs()
        active_names = {s.name for s in active_plugin_servers}
        stale = self._plugin_mcp_server_names - active_names

        kept: list[Any] = []
        replaced: set[str] = set()
        for server in self.config.harness.mcp.servers:
            if server.name in stale:
                # This server was provided by a plugin that is now disabled or removed.
                continue
            if server.name in active_names:
                # Replace in place so updates to args/env are picked up.
                for ps in active_plugin_servers:
                    if ps.name == server.name:
                        kept.append(ps)
                        replaced.add(ps.name)
                        break
                else:
                    kept.append(server)
            else:
                # Static or otherwise non-plugin server; preserve it.
                kept.append(server)

        # Append any brand-new plugin servers.
        for ps in active_plugin_servers:
            if ps.name not in replaced:
                kept.append(ps)

        self.config.harness.mcp.servers = kept
        self._plugin_mcp_server_names = active_names

    def plugin_event(
        self,
        chat_id: str,
        plugin: str,
        *,
        event: str | None = None,
        raw_args: str | None = None,
        **params: Any,
    ) -> ChatResult:
        """Dispatch an event to a state plugin and wrap the reply in a ChatResult."""
        reply = self._plugins.event(chat_id, plugin, event=event, raw_args=raw_args, **params)
        return ChatResult(reply=reply)

    @_locked
    def plugin_list(self, chat_id: str) -> list[dict[str, Any]]:
        return self._plugins.list_plugin_status(chat_id)

    @_locked
    def plugin_set_enabled(self, chat_id: str, name: str, enabled: bool) -> ChatResult:
        return ChatResult(reply=self._plugins.set_plugin_enabled(chat_id, name, enabled))

    @_locked
    def plugin_reload(self, chat_id: str, name: str) -> ChatResult:
        return ChatResult(reply=self._plugins.reload_plugin(chat_id, name))

    @_locked
    def plugin_add(self, config: PluginConfig) -> ChatResult:
        result = self._plugins.add_plugin(config)
        self.config.harness.plugins = self._plugins._plugins
        self._register_plugin_mcp_servers()
        self.context_builder.plugin_manager = self._plugins
        self._runtime._save_runtime_overrides()
        return ChatResult(reply=result)

    @_locked
    def plugin_remove(self, name: str) -> ChatResult:
        result = self._plugins.remove_plugin(name)
        self.config.harness.plugins = self._plugins._plugins
        self._register_plugin_mcp_servers()
        self.context_builder.plugin_manager = self._plugins
        self._runtime._save_runtime_overrides()
        return ChatResult(reply=result)

    @_locked
    def plugin_toggle(self, name: str, enabled: bool, chat_id: str | None = None) -> ChatResult:
        if chat_id is not None:
            result = self._plugins.set_plugin_enabled(chat_id, name, enabled)
        else:
            result = self._plugins.toggle_plugin(name, enabled)
            self.config.harness.plugins = self._plugins._plugins
            self._register_plugin_mcp_servers()
            self.context_builder.plugin_manager = self._plugins
            self._runtime._save_runtime_overrides()
        return ChatResult(reply=result)

    @_locked
    def plugin_rollback(self, steps: int = 1) -> ChatResult:
        result = self._plugins.rollback(steps)
        self.config.harness.plugins = self._plugins._plugins
        self._register_plugin_mcp_servers()
        self.context_builder.plugin_manager = self._plugins
        self._runtime._save_runtime_overrides()
        return ChatResult(reply=result)

    @_locked
    def plugin_sandbox(
        self, module: str, plugin: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run a candidate plugin module through start/stop in a subprocess."""
        data: dict[str, Any] = {"name": "sandbox", "module": module, **(plugin or {})}
        cfg = PluginConfig(**data)
        cmd = [
            sys.executable,
            "-m",
            "diploid_agent.plugin_sandbox",
            "--module",
            module,
            "--name",
            cfg.name,
            "--chat-id",
            "0",
            "--prompt-slot",
            cfg.prompt_slot,
        ]
        if cfg.state_file:
            cmd.extend(["--state-file", cfg.state_file])
        if cfg.config:
            cmd.extend(["--config-json", _json.dumps(cfg.config, ensure_ascii=False)])
        for p in self.config.harness.plugin_paths:
            if p.exists():
                cmd.extend(["--plugin-path", str(p)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        try:
            output = _json.loads(result.stdout.splitlines()[-1])
        except (IndexError, _json.JSONDecodeError) as exc:
            output = {"ok": False, "error": f"Invalid sandbox output: {result.stdout!r} ({exc})"}
        if not output.get("ok") and self._incidents is not None:
            self._incidents.record(
                plugin=cfg.name,
                phase="sandbox",
                error=output.get("error", "unknown"),
                action="rejected",
            )
        return output

    @_locked
    def plugin_create(
        self,
        name: str,
        module: str | None = None,
        prompt_slot: str = "self_state",
        state_file: str | None = None,
        mcp_server: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Scaffold a new plugin module on disk, sandbox it, and return a ready config."""
        target_module = module or name
        if not target_module.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Unsafe plugin module name: {target_module}")
        if target_module.count(".") or target_module.startswith("/"):
            raise ValueError(f"Plugin module name must be a bare package name: {target_module}")

        plugin_root = self.config.harness.plugin_paths[0].expanduser()
        plugin_root.mkdir(parents=True, exist_ok=True)
        if str(plugin_root) not in sys.path:
            sys.path.append(str(plugin_root))

        plugin_dir = plugin_root / target_module
        if plugin_dir.exists():
            raise ValueError(f"Plugin directory already exists: {plugin_dir}")
        plugin_dir.mkdir(parents=True)

        init_path = plugin_dir / "__init__.py"
        init_path.write_text(
            f'"""{target_module} plugin for diploid-agent."""\n\n'
            f"from __future__ import annotations\n\n"
            f"from typing import Any\n\n"
            f"from diploid_agent.config import PluginConfig\n"
            f"from diploid_agent.plugins.base import StatePlugin\n\n\n"
            f"class Plugin(StatePlugin):\n"
            f'    """A minimal state plugin."""\n\n'
            f"    def __init__(\n"
            f"        self,\n"
            f"        config: PluginConfig,\n"
            f"        chat_id: str,\n"
            f"        sessions_root: Any,\n"
            f"        runtime: Any = None,\n"
            f"    ) -> None:\n"
            f"        super().__init__(config, chat_id, sessions_root, runtime=runtime)\n\n"
            f"    def prompt_block(self, max_chars: int | None = None) -> str | None:\n"
            f"        return None\n",
            encoding="utf-8",
        )

        plugin_config: dict[str, Any] = {
            "name": name,
            "enabled": True,
            "module": target_module,
            "prompt_slot": prompt_slot,
            "prompt_order": 100,
            "max_prompt_chars": 0,
        }
        if state_file:
            plugin_config["state_file"] = state_file
        if mcp_server:
            plugin_config["mcp_server"] = mcp_server
        if config:
            plugin_config["config"] = config

        sandbox_result = self.plugin_sandbox(target_module, plugin_config)
        if not sandbox_result.get("ok"):
            # Don't leave a broken scaffold behind.
            try:
                shutil.rmtree(plugin_dir)
            except OSError:
                pass
            error = sandbox_result.get("error", "unknown")
            raise ValueError(f"Sandbox failed for {target_module}: {error}")

        return plugin_config

    def incidents(self) -> list[dict[str, Any]]:
        return self._incidents.recent()

    def incidents_for_plugin(self, name: str) -> list[dict[str, Any]]:
        return self._incidents.for_plugin(name)

    @_locked
    def record_incident(
        self,
        plugin: str,
        phase: str,
        error: str,
        action: str = "",
        chat_id: str = "",
    ) -> ChatResult:
        self._incidents.record(
            plugin=plugin,
            phase=phase,
            error=error,
            action=action,
            chat_id=chat_id,
        )
        return ChatResult(reply="incident recorded")
