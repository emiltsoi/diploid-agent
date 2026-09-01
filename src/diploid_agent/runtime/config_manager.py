"""Runtime configuration management and live override persistence."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from diploid_agent.config import (
    Config,
    ConfigPersistenceError,
    NotificationsConfig,
    PluginConfig,
    TaskConfig,
    TelegramConfig,
    TimerConfig,
    WakerConfig,
)
from diploid_agent.models import RuntimeStatus
from diploid_agent.plan.models import PlanStatus

logger = logging.getLogger(__name__)

_T = TypeVar("_T", bound=BaseModel)


class RuntimeConfigManager:
    """Live runtime configuration loading, updating, and persistence."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._runtime_overrides_path = (
            Path(runtime.config.harness.session_store_path).expanduser().parent
            / "runtime-overrides.yaml"
        )
        self._loaded_overrides: dict[str, Any] | None = None

    @property
    def config(self) -> Config:
        return self._runtime.config

    @property
    def _lock(self):
        return self._runtime._lock

    @property
    def store_path(self) -> Path:
        return self._runtime.store_path

    @property
    def _plugins(self):
        return self._runtime._plugins

    @property
    def _runtime_metrics(self):
        return self._runtime._runtime_metrics

    @property
    def _chat_store(self):
        return self._runtime._chat_store

    def get_status(self) -> RuntimeStatus:
        """Return the current runtime daemon status."""
        now = time.time()
        plans = self._runtime.plan_manager.list_plans()
        active_plans = [p.name for p in plans if p.status == PlanStatus.ACTIVE]
        return RuntimeStatus(
            instance_id=self._runtime.instance_id,
            started_at=self._runtime.instance_started_at,
            uptime_seconds=now - self._runtime.instance_started_at,
            event_bus_running=self._runtime.event_bus.running,
            timer_running=self._runtime.timer_service.running,
            task_engine_active=self._runtime.task_engine.is_running(),
            plan_count=len(plans),
            pending_wake_count=self._runtime.wake_queue.due_count(now=now),
            active_chat_count=len(self._runtime._active_turns),
            plan_active=bool(active_plans),
            active_plans=active_plans,
        )

    def get_task_config(self) -> TaskConfig:
        """Return the current live task configuration."""
        return self.config.harness.task

    def _update_config_section(
        self,
        current: _T,
        new: _T,
        *,
        success: str,
        post: Callable[[], None] | None = None,
        error: str = "Config updated in memory but persistence failed",
    ) -> str:
        """Apply a partial update to a config section and persist overrides."""
        with self._lock:
            for field in new.model_fields_set:
                setattr(current, field, getattr(new, field))
            if post is not None:
                post()
            if not self._runtime._save_runtime_overrides():
                raise ConfigPersistenceError(error)
            return success

    def update_task_config(self, task_config: TaskConfig) -> str:
        """Update the live task configuration, resize the worker pool, and persist.

        Only the fields that were explicitly set on ``task_config`` are mutated,
        so partial updates (e.g., just ``workers``) do not clobber other live
        settings like ``acp_model`` or ``enabled_types``. The update is applied
        in-place on the existing ``TaskConfig`` so the ``TaskEngine`` and
        ``AgentRuntime`` share the same object.

        Raises:
            ConfigPersistenceError: If the in-memory update succeeds but the
                runtime-overrides file cannot be written.
        """
        return self._update_config_section(
            self.config.harness.task,
            task_config,
            success="Task config updated",
            post=self._runtime.task_engine.reconfigure,
            error="Task config updated in memory but persistence failed",
        )

    def _load_runtime_overrides(self) -> None:
        """Load persisted runtime config overrides."""
        if not self._runtime_overrides_path.exists():
            return
        try:
            data = yaml.safe_load(self._runtime_overrides_path.read_text()) or {}
        except (OSError, ValueError, yaml.YAMLError):
            logger.exception(
                "Failed to load runtime overrides from %s", self._runtime_overrides_path
            )
            return
        if not isinstance(data, dict):
            logger.warning(
                "Runtime overrides file %s is not a dict; skipping", self._runtime_overrides_path
            )
            return
        self._loaded_overrides = data
        self._load_override("task", TaskConfig)
        self._load_override("waker", WakerConfig)
        self._load_override("timer", TimerConfig)
        self._load_override("notifications", NotificationsConfig)
        self._load_override("telegram", TelegramConfig)
        self._load_plugins_override(data.get("plugins"))

    def _load_override(self, key: str, model_cls: type) -> None:
        """Load one top-level override section if it is a valid mapping."""
        data = self._loaded_overrides
        if data is None:
            return
        section = data.get(key)
        if section is None:
            return
        if not isinstance(section, dict):
            logger.warning("Runtime overrides '%s' entry is not a dict; skipping", key)
            return
        try:
            existing = getattr(self.config.harness, key)
            if isinstance(existing, model_cls):
                # Merge the override values into the existing section so fields
                # that are missing from the runtime-overrides file keep the
                # defaults from harness.yaml. Re-validate to skip malformed
                # overrides just like a fresh model construction would.
                merged = {**existing.model_dump(), **section}
                setattr(self.config.harness, key, model_cls(**merged))
            else:
                setattr(self.config.harness, key, model_cls(**section))
        except (ValueError, TypeError):
            logger.exception("Invalid %s override in %s", key, self._runtime_overrides_path)

    def _load_plugins_override(self, section: Any) -> None:
        """Load the plugins override list, if it is a valid list of mappings."""
        if section is None:
            return
        if not isinstance(section, list):
            logger.warning("Runtime overrides 'plugins' entry is not a list; skipping")
            return
        plugins: list[PluginConfig] = []
        for item in section:
            if not isinstance(item, dict):
                logger.warning("Runtime overrides 'plugins' item is not a dict; skipping")
                continue
            try:
                plugins.append(PluginConfig(**item))
            except (ValueError, TypeError):
                logger.exception("Invalid plugin override in %s", self._runtime_overrides_path)
        if plugins:
            self.config.harness.plugins = plugins

    def _save_runtime_overrides(self) -> bool:
        """Persist runtime config overrides to disk atomically."""
        with self._lock:
            try:
                self._runtime_overrides_path.parent.mkdir(parents=True, exist_ok=True)
                overrides = {
                    "task": self.config.harness.task.model_dump(exclude_none=True, mode="json"),
                    "waker": self.config.harness.waker.model_dump(exclude_none=True, mode="json"),
                    "timer": self.config.harness.timer.model_dump(exclude_none=True, mode="json"),
                    "notifications": self.config.harness.notifications.model_dump(
                        exclude_none=True, mode="json"
                    ),
                    "telegram": self.config.harness.telegram.model_dump(
                        exclude_none=True, mode="json"
                    ),
                    "plugins": [
                        p.model_dump(exclude_none=True, mode="json")
                        for p in self.config.harness.plugins
                    ],
                }
                tmp_path = self._runtime_overrides_path.with_suffix(".yaml.tmp")
                tmp_path.write_text(
                    yaml.safe_dump(overrides, sort_keys=False, default_flow_style=False)
                )
                os.replace(tmp_path, self._runtime_overrides_path)
                return True
            except OSError:
                logger.exception(
                    "Failed to save runtime overrides to %s", self._runtime_overrides_path
                )
                return False

    def get_notifications_config(self) -> NotificationsConfig:
        """Return the current live notifications configuration."""
        return self.config.harness.notifications

    def update_notifications_config(self, notifications_config: NotificationsConfig) -> str:
        """Update the live notifications configuration and persist.

        Only explicitly set fields are mutated, so partial updates do not
        clobber other live settings. The notifier is recreated so changes take
        effect immediately.

        Raises:
            ConfigPersistenceError: If the in-memory update succeeds but the
                runtime-overrides file cannot be written.
        """
        return self._update_config_section(
            self.config.harness.notifications,
            notifications_config,
            success="Notifications config updated",
            post=lambda: setattr(self._runtime, "notifier", self._runtime._create_notifier()),
            error="Notifications config updated in memory but persistence failed",
        )

    def get_timer_config(self) -> TimerConfig:
        """Return the current live timer configuration."""
        return self.config.harness.timer

    def update_timer_config(self, timer_config: TimerConfig) -> str:
        """Update the live timer configuration and persist.

        Only explicitly set fields are mutated, so partial updates do not
        clobber other live settings.

        Raises:
            ConfigPersistenceError: If the in-memory update succeeds but the
                runtime-overrides file cannot be written.
        """
        return self._update_config_section(
            self.config.harness.timer,
            timer_config,
            success="Timer config updated",
            error="Timer config updated in memory but persistence failed",
        )

    def get_waker_config(self) -> WakerConfig:
        """Return the current live waker configuration."""
        return self.config.harness.waker

    def update_waker_config(self, waker_config: WakerConfig) -> str:
        """Update the live waker configuration and persist.

        Only explicitly set fields are mutated, so partial updates do not
        clobber other live settings.

        Raises:
            ConfigPersistenceError: If the in-memory update succeeds but the
                runtime-overrides file cannot be written.
        """
        return self._update_config_section(
            self.config.harness.waker,
            waker_config,
            success="Waker config updated",
            error="Waker config updated in memory but persistence failed",
        )

    def get_telegram_config(self) -> TelegramConfig:
        """Return the current live Telegram configuration."""
        return self.config.harness.telegram

    def update_telegram_config(self, telegram_config: TelegramConfig) -> str:
        """Update the live Telegram configuration and persist.

        Only explicitly set fields are mutated, so partial updates do not
        clobber other live settings. The notifier is recreated so changes to
        the token or enabled flag take effect immediately.

        Raises:
            ConfigPersistenceError: If the in-memory update succeeds but the
                runtime-overrides file cannot be written.
        """
        return self._update_config_section(
            self.config.harness.telegram,
            telegram_config,
            success="Telegram config updated",
            post=lambda: setattr(self._runtime, "notifier", self._runtime._create_notifier()),
            error="Telegram config updated in memory but persistence failed",
        )

    def get_plugins_config(self) -> list[PluginConfig]:
        """Return the current live plugin list."""
        return list(self.config.harness.plugins)

    def update_plugins_config(self, plugins: list[PluginConfig]) -> str:
        """Update the live plugin list and persist.

        Plugins are matched by name and only fields that were explicitly set in
        the request are mutated. New plugins are appended. The PluginManager is
        reconfigured with the merged list.

        Raises:
            ConfigPersistenceError: If the in-memory update succeeds but the
                runtime-overrides file cannot be written.
        """
        with self._lock:
            by_name: dict[str, PluginConfig] = {p.name: p for p in self.config.harness.plugins}
            for plugin in plugins:
                if plugin.name in by_name:
                    existing = by_name[plugin.name]
                    for field in plugin.model_fields_set:
                        setattr(existing, field, getattr(plugin, field))
                else:
                    by_name[plugin.name] = plugin
            merged = list(by_name.values())
            self.config.harness.plugins = merged
            self._plugins.reconfigure(merged)
            self._runtime._register_plugin_mcp_servers()
            self._runtime.context_builder.plugin_manager = self._runtime._plugins
            if not self._runtime._save_runtime_overrides():
                raise ConfigPersistenceError(
                    "Plugins config updated in memory but persistence failed"
                )
            return "Plugins config updated"

    def get_config(self) -> dict[str, Any]:
        """Return the current live runtime configuration (excluding secrets)."""
        with self._lock:
            data = self.config.model_dump(mode="json", exclude={"secrets"})
            # Redact nested credentials that can be loaded from environment variables.
            telegram = data.get("harness", {}).get("telegram")
            if isinstance(telegram, dict):
                telegram["token"] = "***"
            hindsight = data.get("harness", {}).get("memory", {}).get("hindsight")
            if isinstance(hindsight, dict):
                hindsight["api_key"] = "***"
            return data

    def update_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a partial runtime configuration update.

        Supported top-level keys: ``telegram`` and ``plugins``. Other keys are
        ignored. The updated live configuration is returned.

        Raises:
            ConfigPersistenceError: If the in-memory update succeeds but the
                runtime-overrides file cannot be written.
        """
        with self._lock:
            if "telegram" in patch:
                self.update_telegram_config(TelegramConfig(**patch["telegram"]))
            if "plugins" in patch:
                self.update_plugins_config([PluginConfig(**p) for p in patch["plugins"]])
            return self.get_config()
