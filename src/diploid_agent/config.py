"""Configuration and secret loading for diploid-agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from diploid_agent.plan.models import TaskType


class Secrets(BaseModel):
    windsurf_api_key: str | None = Field(None, alias="WINDSURF_API_KEY", repr=False)
    harness_api_key: str | None = Field(None, alias="HARNESS_API_KEY", repr=False)

    @field_validator("windsurf_api_key", "harness_api_key")
    @classmethod
    def strip_secret(cls, v: str | None) -> str | None:
        return v.strip() if v is not None else None


class EngineConfig(BaseModel):
    """Engine configuration."""

    provider: str = "diploid"
    bin: str = "~/.local/bin/devin"
    start_args: list[str] | None = None
    model: str = "swe-1-7"
    context_window: int | None = None
    permission_mode: str = "dangerous"
    timeout: float = 900.0
    soft_timeout: float | None = 600.0  # seconds before auto-cancel + partial reply; None disables
    acp_startup_timeout: float = 30.0  # seconds to wait for `devin acp` initialize handshake
    acp_watchdog_interval: float = 10.0  # seconds between ACP transport watchdog checks
    acp_watchdog_timeout: float = (
        120.0  # seconds without ACP prompt output before the watchdog kills the child
    )
    acp_control_timeout: float = (
        120.0  # seconds without a control-call response before the transport is reset
    )
    continuation_triggers: list[str] = Field(
        default_factory=lambda: ["continue", "go on", "proceed", "resume"]
    )

    @field_validator("bin")
    @classmethod
    def expand_bin(cls, v: str) -> str:
        return os.path.expanduser(v)


DiploidConfig = EngineConfig  # backward-compatible alias


class PersonaConfig(BaseModel):
    name: str
    profile_root: Path
    fleet_root: Path | None = None
    identity_class: str = "worker"  # worker | graduating | family
    memory_filename: str = "MEMORY.md"
    knowledge_ids: list[str] = Field(default_factory=list)

    @field_validator("profile_root", "fleet_root")
    @classmethod
    def expand_paths(cls, v: Path | None) -> Path | None:
        return v.expanduser() if v is not None else None


class TelegramConfig(BaseModel):
    enabled: bool = False
    webhook_port: int = 8080
    token: str | None = None
    stream_thoughts: bool = False
    stream_chunk_interval: float = 2.0
    intermediate_messages: bool = True
    intermediate_idle: float = 3.0
    intermediate_min_chars: int = 20
    min_telegram_interval: float = 1.0
    min_edit_message_interval: float = 2.0
    message_format: Literal["plain", "markdown_v2"] = "plain"
    code_style: Literal["inline", "box"] = "inline"


class MetricsConfig(BaseModel):
    expose_in_prompt: bool = False
    max_recent_turns: int = 100


class McpServerConfig(BaseModel):
    """One MCP server definition. Only stdio is validated against the ACP shape."""

    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: list[str] = Field(default_factory=list)
    disabled: bool = False

    @field_validator("args", "env", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any) -> list[str]:
        return v if isinstance(v, list) else []


class McpConfig(BaseModel):
    """MCP defaults for the harness."""

    servers: list[McpServerConfig] = Field(default_factory=list)
    default_enabled: list[str] = Field(default_factory=list)


class SkillsConfig(BaseModel):
    """Skill discovery defaults."""

    shared_root: Path = Path("personas/shared")
    default_enabled: list[str] = Field(default_factory=list)
    default_lazy: bool = False  # if True, no skill is active until triggered or enabled
    allow_chat_creation: bool = True

    @field_validator("shared_root")
    @classmethod
    def expand_shared_root(cls, v: Path) -> Path:
        return v.expanduser()


class HindsightConfig(BaseModel):
    base_url: str = "http://localhost:8888"
    bank: str | None = None
    api_key: str | None = None
    timeout: float = 120.0
    max_recall_tokens: int = 1500
    recall_min_scores: dict[str, float] = Field(
        default_factory=lambda: {"semantic": 0.25, "reranker": 0.5}
    )
    prefer_observations: bool = True
    async_writes: bool = True
    fallback_to_file: bool = True
    spool_path: Path | None = None


class MemoryConfig(BaseModel):
    backend: str = "file"  # file | hindsight
    n_turns_summarization: int | None = None
    max_chat_memory_chars: int = 8192
    max_recall_chars: int | None = None  # extra client-side cap on recall results
    max_persona_memory_chars: int = 16384
    max_reply_quote_chars: int = 2048
    max_bot_reply_quote_chars: int = 240
    short_term_turns: int = 10
    short_term_strategy: str = "raw"  # raw | smart
    min_short_term_turns: int = 2
    max_short_term_chars: int = 4096
    include_short_term: bool = True
    short_term_summary_cache_days: int = 7
    hindsight: HindsightConfig = Field(default_factory=HindsightConfig)


class PluginConfig(BaseModel):
    """One per-chat state plugin."""

    name: str
    enabled: bool = True
    module: str | None = None
    state_file: str | None = None
    mcp_server: McpServerConfig | None = None
    skill: str | None = None
    skill_path: Path | None = None
    prompt_slot: str = "persona_state"
    first_prompt_only: bool = False
    prompt_order: int = 100
    max_prompt_chars: int = 1024
    prompt_template: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("skill_path")
    @classmethod
    def _expand_skill_path(cls, v: Path | None) -> Path | None:
        return v.expanduser() if v is not None else None


class ConfigPersistenceError(RuntimeError):
    """Raised when a live config update is applied in memory but cannot be persisted."""


class NotificationsConfig(BaseModel):
    """Outbound notification configuration."""

    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = True
    webhook_url: str | None = Field(default=None, min_length=1)

    @field_validator("webhook_url", mode="before")
    @classmethod
    def _normalize_webhook_url(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        return v if v else None


class WakerConfig(BaseModel):
    """Wake/dispatch retry configuration."""

    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = False
    interval_seconds: float = Field(default=5.0, gt=0, le=86400)
    max_retries: int = Field(default=3, ge=0, le=100)
    retry_after: float = Field(default=30.0, gt=0, le=86400)
    lease_seconds: float = Field(default=300.0, gt=0, le=86400)


class PlanConfig(BaseModel):
    """Plan persistence and layout configuration."""

    root: Path = Path("plans")
    store_filename: str = "plans.jsonl"

    @field_validator("root")
    @classmethod
    def expand_root(cls, v: Path) -> Path:
        return v.expanduser()


class TaskConfig(BaseModel):
    """Background task worker configuration."""

    model_config = ConfigDict(validate_assignment=True)

    workers: int = Field(default=4, ge=1, le=64)
    shell_timeout: float = Field(default=60.0, gt=0, le=86400)
    enabled_types: list[str] = Field(default_factory=lambda: ["shell", "noop", "acp"])
    acp_timeout: float | None = Field(default=None, gt=0, le=86400)
    acp_model: str | None = Field(default=None, min_length=1)

    @field_validator("enabled_types")
    @classmethod
    def _check_enabled_types(cls, v: list[str]) -> list[str]:
        valid = {t.value for t in TaskType}
        invalid = [t for t in v if t not in valid]
        if invalid:
            raise ValueError(f"invalid task types: {invalid}; must be one of {sorted(valid)}")
        return v

    @field_validator("acp_model")
    @classmethod
    def _strip_acp_model(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("acp_model cannot be empty or whitespace-only")
        return v


class TimerConfig(BaseModel):
    """Background wake queue polling configuration."""

    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = True
    interval_seconds: float = Field(default=5.0, gt=0, le=86400)
    lease_seconds: float = Field(default=300.0, gt=0, le=86400)
    max_retries: int = Field(default=5, ge=0, le=100)
    retry_after_seconds: float = Field(default=30.0, gt=0, le=86400)


class ConversationBudget(BaseModel):
    """Per-conversation token budget."""

    enabled: bool = False
    max_total_tokens: int = 100_000
    warning_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    hard_cap: bool = True


class RoutingConfig(BaseModel):
    """Lane-based model routing and conversation budget."""

    enabled: bool = False
    fallback_model: str | None = None
    lanes: dict[str, str] = Field(default_factory=dict)
    lane_keywords: dict[str, list[str]] = Field(default_factory=dict)
    budget: ConversationBudget = Field(default_factory=ConversationBudget)


class HarnessConfig(BaseModel):
    sessions_root: Path = Path("sessions")
    session_store_path: Path = Path("sessions.jsonl")
    dispatch_store_path: Path | None = None
    wake_store_path: Path | None = None
    instance_ttl_seconds: float = 60.0
    listen_host: str = "127.0.0.1"
    listen_port: int = 4003
    session_prune_enabled: bool = True
    session_prune_days: int = 14
    plugin_paths: list[Path] = Field(default_factory=lambda: [Path("~/.devin/plugins")])
    plugins: list[PluginConfig] = Field(default_factory=list)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    plan: PlanConfig = Field(default_factory=PlanConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    waker: WakerConfig = Field(default_factory=WakerConfig)
    timer: TimerConfig = Field(default_factory=TimerConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    @model_validator(mode="after")
    def _set_store_paths(self) -> HarnessConfig:
        if self.dispatch_store_path is None:
            self.dispatch_store_path = self.session_store_path.parent / "dispatch_store.jsonl"
        if self.wake_store_path is None:
            self.wake_store_path = self.session_store_path.parent / "wake_queue.jsonl"
        return self

    @field_validator("plugin_paths")
    @classmethod
    def _expand_plugin_paths(cls, v: list[Path]) -> list[Path]:
        return [p.expanduser() for p in v]

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_body(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("body") is not None:
            raise ValueError(
                "the legacy body plugin is no longer supported. "
                "Move body configuration to harness.plugins (or remove it to use the defaults)."
            )
        return data


class Config(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    engine: EngineConfig = Field(
        default_factory=EngineConfig,
        alias="diploid",
        serialization_alias="engine",
    )
    persona: PersonaConfig
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    secrets: Secrets | None = None

    @classmethod
    def load(
        cls,
        config_path: Path,
        secrets_path: Path | None = None,
    ) -> Config:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        if secrets_path is None:
            secrets_path = config_path.parent / "secrets.env"
        secrets_data = _load_dotenv(secrets_path)

        # TELEGRAM_BOT_TOKEN can live in the environment, the config, or secrets.env.
        # Precedence: environment > config > secrets.env.
        telegram_token = (
            os.environ.get("TELEGRAM_BOT_TOKEN")
            or data.get("harness", {}).get("telegram", {}).get("token")
            or secrets_data.pop("TELEGRAM_BOT_TOKEN", None)
        )
        if telegram_token:
            data.setdefault("harness", {})["telegram"] = {
                **data.get("harness", {}).get("telegram", {}),
                "token": telegram_token,
            }

        windsurf_api_key = secrets_data.pop("WINDSURF_API_KEY", None) or os.environ.get(
            "WINDSURF_API_KEY"
        )
        harness_api_key = secrets_data.pop("HARNESS_API_KEY", None) or os.environ.get(
            "HARNESS_API_KEY"
        )

        secrets: dict[str, str] = data.get("secrets") or {}
        if windsurf_api_key:
            secrets["WINDSURF_API_KEY"] = windsurf_api_key
        if harness_api_key:
            secrets["HARNESS_API_KEY"] = harness_api_key
        if secrets:
            data["secrets"] = secrets

        return cls(**data)


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env parser (no external dependencies)."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        result[key] = value
    return result
