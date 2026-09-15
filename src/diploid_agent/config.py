"""Configuration and secret loading for diploid-agent."""

from __future__ import annotations

import os
import re
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
    timeout: float | None = 900.0  # hard prompt deadline; None waits until /stop or completion
    soft_timeout: float | None = 600.0  # seconds before auto-cancel + partial reply; None disables
    acp_startup_timeout: float = 30.0  # seconds to wait for `devin acp` initialize handshake
    acp_watchdog_interval: float = 10.0  # seconds between ACP transport watchdog checks
    acp_watchdog_timeout: float = (
        120.0  # seconds without a control-call response before the transport is reset
    )
    acp_control_timeout: float = (
        120.0  # seconds without a control-call response before the transport is reset
    )
    acp_max_restarts: int = 3  # transport-error restarts allowed in the backoff window
    acp_restart_backoff_window: float = 300.0  # seconds for the transport-error budget
    acp_max_mcp_restarts: int = 5  # MCP-change restarts allowed in the backoff window
    acp_max_user_restarts: int = 5  # user-requested restarts allowed in the backoff window
    acp_resume_enabled: bool = (
        True  # use ACP session/resume (or load) instead of prompt rehydration
    )
    acp_resume_max_retries: int = 1  # retries per resume method (resume or load)
    acp_resume_retry_base_seconds: float = 0.5
    acp_resume_retry_max_seconds: float = 5.0
    acp_resume_timeout: float = (
        120.0  # total budget for ACP session resume/load incl. config re-apply
    )
    acp_resume_after_restart_timeout: float = (
        15.0  # resume budget when the transport was just restarted (session/load from disk)
    )
    acp_silence_warn_after: float = (
        600.0  # seconds of in-flight-prompt stdout silence before a lifecycle warning
    )
    acp_timeout_auto_resend: bool = (
        False  # if True, automatically resend a hard-timeout turn; if False, ask first
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


class AuthorshipConfig(BaseModel):
    """Per-plugin toggles for self-wake, self-inference, felt authorship, cron.

    The master on/off switch is the plugin's own ``enabled`` flag.  These
    settings live inside ``PluginConfig.config`` so the contract is owned by
    the authorship plugin, not the runtime.
    """

    self_wake_enabled: bool = False
    self_inference_enabled: bool = False
    felt_authorship_enabled: bool = False
    cron_enabled: bool = False
    restart_enabled: bool = False
    user_override: list[str] = Field(default_factory=list)

    @field_validator("user_override", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any) -> list[str]:
        return v if isinstance(v, list) else []


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
    attachments_enabled: bool = True
    attachments_max_bytes: int = 20_000_000
    attachments_dirname: str = "inbox"
    stt_provider: Literal["none", "faster-whisper", "command"] = "none"
    stt_model: str = "small"
    stt_command: str = ""
    tts_provider: Literal["none", "piper", "command"] = "none"
    tts_model_path: str = ""
    tts_command: str = ""
    tts_max_chars: int = 800


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
    # Consolidation scope sent as each retain item's observation_scopes.
    # "" = server default ("combined": every distinct tag set becomes its own
    # scope, so volatile session:N tags fragment observations per session).
    # "chat" = one scope per chat ([["chat:<chat_id>"]]) — dedups across
    # sessions while keeping observations visible to the chat-scoped recall
    # filter. "shared" = single global untagged scope (breaks tag-filtered
    # recall of observations — only use with unfiltered recall).
    observation_scope: str = ""


class MemoryConfig(BaseModel):
    backend: str = "file"  # file | hindsight
    summary_timeout: float | None = Field(
        default=600.0,
        gt=0,
        le=86400,
        description="Hard ACP timeout for memory summarization calls.",
    )
    summary_soft_timeout: float | None = Field(
        default=30.0,
        gt=0,
        le=86400,
        description="Soft ACP timeout before cancel for memory summarization calls.",
    )
    n_turns_summarization: int | None = None
    max_chat_memory_chars: int = 8192
    max_recall_chars: int | None = None  # extra client-side cap on recall results
    max_persona_memory_chars: int = 16384
    max_reply_quote_chars: int = 2048
    max_bot_reply_quote_chars: int = 240
    short_term_turns: int = 10
    short_term_strategy: str = "smart"  # raw | smart
    min_short_term_turns: int = 2
    max_short_term_chars: int = 6144
    include_short_term: bool = True
    short_term_summary_cache_days: int = 7
    precompute_short_term_summary: bool = True
    precompute_short_term_summary_min_turns: int = 5
    recall_on_follow_up: bool = False  # whether to run long-term recall on follow-ups
    fresh_recall_triggers: list[str] = Field(
        default_factory=lambda: [
            "do you remember",
            "what did we",
            "where did we",
            "what were we",
            "where were we",
            "did we discuss",
            "did we agree",
            "did we decide",
            "remind me",
            "where did we leave off",
            "what just happened",
            "where are we",
            "as we discussed",
            "as i said",
            "last time",
            "earlier",
            "continue from",
            "pick up where",
            "recall",
            "what did i say",
            "what did we say",
        ]
    )
    fresh_recall_max_chars: int = 1024
    fresh_recall_max_results: int = 3
    fresh_auto_recall_max_chars: int = 512
    fresh_auto_recall_max_results: int = 1
    max_compact_promoted_chars: int = 512
    auto_promote_enabled: bool = True
    auto_promote_tags: list[str] = Field(
        default_factory=lambda: [
            "preference",
            "plan",
            "watchpoint",
            "decision",
            "agreement",
            "fact",
        ]
    )
    auto_promote_triggers: list[str] = Field(
        default_factory=lambda: [
            "i prefer",
            "we agreed",
            "we decided",
            "watchpoint",
            "this matters",
            "remember that i",
        ]
    )
    max_promoted_lines: int = 20
    max_compact_chat_memory_chars: int = 256
    max_compact_persona_memory_chars: int = 512
    max_compact_short_term_chars: int = 512
    # Retain pipeline: slice working narration off turn pairs and bundle
    # several turns per retained document for better extraction signal.
    retain_final_segment: bool = False  # retain only the post-last-tool reply segment
    retain_min_final_chars: int = 200  # fall back to the full reply when the segment is shorter
    retain_bundle_turns: int = 1  # turn pairs bundled per retained document (1 = per-turn retain)
    # Speaker labels and context passed to the long-term memory backend.
    # Hindsight uses these to attribute facts to the right person.
    retain_user_prefix: str = "User"
    retain_assistant_prefix: str = "Assistant"
    retain_context: str = ""
    # Cap for the "Recent turns" tail carried into a *new* ACP session prompt
    # (fresh/compact session-boundary builds). The new session starts with an
    # empty context window, so this can be much larger than the compact caps —
    # it is what preserves the thread of the previous session.
    new_session_tail_max_chars: int = 4096
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
    prompt_slot: str = "self_state"
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
    outbox_delivery: bool = False
    mesh_telegram_float: bool = False

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
    enabled_types: list[str] = Field(default_factory=lambda: ["shell", "noop", "acp", "subagent"])
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
    # Agent-initiated self-wakes (reason=self_wake) are rate-limited so an
    # agent cannot loop or spam itself: at most ``self_wake_max_pending``
    # queued per chat, no faster than one enqueue per
    # ``self_wake_min_interval_seconds``, and no farther out than
    # ``self_wake_max_delay_seconds``.
    self_wake_max_pending: int = Field(default=3, ge=0, le=100)
    self_wake_min_interval_seconds: float = Field(default=300.0, ge=0, le=86400)
    self_wake_max_delay_seconds: float = Field(default=604800.0, gt=0, le=31536000)


_CRON_JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class CronScheduleSpec(BaseModel):
    """When a cron job fires. Exactly one field must be set."""

    cron: str | None = None  # 5-field cron expression (croniter)
    every_seconds: float | None = None
    at_daily: str | None = None  # "HH:MM" local wall-clock

    @field_validator("at_daily")
    @classmethod
    def _check_at_daily(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not re.fullmatch(r"[0-2]\d:[0-5]\d", v):
            raise ValueError("at_daily must be 'HH:MM' (local wall-clock)")
        hour = int(v.split(":")[0])
        if hour > 23:
            raise ValueError("at_daily hour must be 00-23")
        return v

    @model_validator(mode="after")
    def _exactly_one(self) -> CronScheduleSpec:
        set_fields = [
            name
            for name in ("cron", "every_seconds", "at_daily")
            if getattr(self, name) is not None
        ]
        if len(set_fields) != 1:
            raise ValueError("schedule requires exactly one of cron / every_seconds / at_daily")
        return self


class CronTriggerSpec(BaseModel):
    """An event condition that fires a job instead of a schedule.

    ``file`` watches a path's mtime — the path must resolve under the
    persona dir or the owning chat's session dir (``session:<rel>``
    selects the session dir explicitly; operator-global files may also
    reach under ``$HOME``). ``body`` edge-fires when a field in the owning
    chat's ``chat_body_state.json`` crosses ``op``/``value``.

    ``cooldown_seconds`` bounds refires (default: ``min_interval_seconds``).
    """

    type: Literal["file", "body"]
    path: str | None = None  # file: path under an allowed root
    field: str | None = None  # body: chat_body_state.json key
    op: Literal[">", ">=", "<", "<=", "==", "!="] | None = None  # body
    value: float | str | bool | None = None  # body
    cooldown_seconds: float | None = Field(default=None, gt=0, le=86400)

    @model_validator(mode="after")
    def _fields_for_type(self) -> CronTriggerSpec:
        if self.type == "file":
            if not (self.path or "").strip():
                raise ValueError("file trigger requires path")
            if self.field is not None or self.op is not None or self.value is not None:
                raise ValueError("file trigger takes only path (+ cooldown_seconds)")
        else:
            if not (self.field or "").strip():
                raise ValueError("body trigger requires field")
            if self.op is None or self.value is None:
                raise ValueError("body trigger requires op and value")
            if self.path is not None:
                raise ValueError("body trigger takes only field/op/value (+ cooldown_seconds)")
        return self


class CronCallSpec(BaseModel):
    """What a cron job runs: a subprocess script or a phantom LLM turn."""

    type: Literal["script", "llm"]
    command: str | None = None  # script
    cwd: str | None = None  # default: persona dir
    persona: str | None = None  # llm — default: this service's persona
    prompt: str | None = None  # llm
    model: str | None = None  # llm model override
    timeout_seconds: float | None = None

    @model_validator(mode="after")
    def _required_fields(self) -> CronCallSpec:
        if self.type == "script" and not (self.command or "").strip():
            raise ValueError("script call requires command")
        if self.type == "llm" and not (self.prompt or "").strip():
            raise ValueError("llm call requires prompt")
        return self


class CronJobSpec(BaseModel):
    """One declarative scheduled job from a crons.yaml file."""

    id: str
    enabled: bool = True
    schedule: CronScheduleSpec | None = None
    trigger: CronTriggerSpec | None = None
    call: CronCallSpec
    delivery: Literal["silent", "digest", "turn"] = "silent"
    chat_id: str | None = None  # owning chat; persona jobs default to theirs
    overlap: Literal["skip", "queue"] = "skip"
    catchup: Literal["once", "skip"] = "once"
    max_consecutive_failures: int = Field(default=3, ge=1, le=100)

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not _CRON_JOB_ID_RE.fullmatch(v):
            raise ValueError("job id must be a slug: [a-z0-9][a-z0-9_-]*")
        return v

    @model_validator(mode="after")
    def _exactly_one_driver(self) -> CronJobSpec:
        if (self.schedule is None) == (self.trigger is None):
            raise ValueError("job requires exactly one of schedule / trigger")
        return self


class CronFileSpec(BaseModel):
    """Parsed contents of one crons.yaml file."""

    jobs: list[CronJobSpec] = Field(default_factory=list)


class CronConfig(BaseModel):
    """Declarative cron scheduler configuration (``harness.cron.*``)."""

    enabled: bool = True
    tick_seconds: float = Field(default=5.0, gt=0, le=3600)
    global_file: Path | None = Path("config/crons.yaml")
    persona_filename: str = "crons.yaml"
    state_path: Path | None = None  # default: next to wake_store_path
    max_jobs_per_persona: int = Field(default=8, ge=0, le=100)
    max_jobs_global: int = Field(default=16, ge=0, le=500)
    min_interval_seconds: float = Field(default=300.0, ge=0, le=86400)
    max_llm_timeout_seconds: float = Field(default=600.0, gt=0, le=86400)
    max_script_timeout_seconds: float = Field(default=300.0, gt=0, le=86400)
    turn_delivery_max_per_day: int = Field(default=4, ge=0, le=100)
    results_dirname: str = "cron"  # sessions/<chat>/cron/
    digest_max_jobs: int = Field(default=8, ge=1, le=100)
    phantom_persona_max_chars: int = Field(default=6000, ge=0)
    phantom_memory_max_chars: int = Field(default=4000, ge=0)
    phantom_promoted_max_chars: int = Field(default=1500, ge=0)
    # Extra filesystem roots a file trigger may watch, for persona AND global
    # jobs — the operator's door for shared spaces outside the persona/session
    # confinement (e.g. a common-room directory on a shared mount).
    trigger_allowed_roots: list[Path] = Field(default_factory=list)

    @field_validator("global_file", "state_path")
    @classmethod
    def _expand_cron_paths(cls, v: Path | None) -> Path | None:
        return v.expanduser() if v is not None else None

    @field_validator("trigger_allowed_roots")
    @classmethod
    def _expand_trigger_roots(cls, v: list[Path]) -> list[Path]:
        return [p.expanduser() for p in v]


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


class MeshConfig(BaseModel):
    """Mesh peer configuration."""

    enabled: bool = False
    agent_name: str | None = None
    private_key_path: Path | None = None
    vault_path: Path | None = None
    registry_url: str | None = None
    registry_pin: str | None = None
    allow_insecure_registry: bool = False
    route: str = "receive"
    sign_timestamp: bool = True
    allow_loopback: bool = False
    chat_mapping: Literal["per_sender", "single", "session"] = "per_sender"
    fallback_chat_id: str = "mesh:inbox"
    chat_map: dict[str, str] = Field(default_factory=dict)
    auto_join: bool = False
    ingress_module: str = "diploid_mesh.ingress"
    mcp_enabled: bool = True
    replay_window_ttl: float = 300.0
    replay_window_size: int = 10000
    rate_limit_per_minute: int = 0
    outbox_enabled: bool = False
    outbox_dir: Path | None = None
    delivery_retries: int = 3
    delivery_backoff: float = 1.0
    delivery_timeout: float = 10.0
    max_sends_per_turn: int = Field(default=3, ge=0)
    max_message_in_turn_suggestion: int = Field(default=2, ge=0)

    @field_validator("private_key_path", "vault_path", "outbox_dir")
    @classmethod
    def _expand_path(cls, v: Path | None) -> Path | None:
        return v.expanduser() if v is not None else None

    @field_validator("registry_url")
    @classmethod
    def _strip_registry_url(cls, v: str | None) -> str | None:
        return v.rstrip("/") if v else None


class PromptBlocksConfig(BaseModel):
    """Per-persona prompt slot allowlist/denylist and caps."""

    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    caps: dict[str, int] = Field(default_factory=dict)


class HarnessConfig(BaseModel):
    sessions_root: Path = Path("sessions")
    session_store_path: Path = Path("sessions.jsonl")
    dispatch_store_path: Path | None = None
    wake_store_path: Path | None = None
    instance_ttl_seconds: float = 60.0
    listen_host: str = "127.0.0.1"
    listen_port: int = 4003
    # Soul re-injection thresholds.  When the context window is under pressure,
    # the harness re-injects cheap identity slots; at high pressure it also
    # re-injects memory and may start a fresh ACP session.
    reinject_soul_threshold: float = 0.7  # cumulative context pressure
    reinject_soul_input_threshold: float = 0.6  # last-turn input pressure
    reinject_soul_full_threshold: float = 0.9  # force full soul + fresh session
    reinject_soul_turns: int = 20  # fallback turn budget when window is unknown
    proactive_new_session_threshold: float = (
        0.85  # estimated prompt ratio that forces a fresh session
    )
    pressure_handoff_enabled: bool = (
        True  # grant one bounded turn to author a handoff before a pressure rebuild
    )
    proactive_input_buffer_factor: float = 1.2  # multiplier for last-turn tokens when estimating
    proactive_soul_token_budget: int = 500  # cheap fresh-soul token budget for proactive sizing
    wake_context_token_budget: int = 0  # 0 = disabled; soft cap for first-turn compact prompts
    proactive_calibration_enabled: bool = True  # live-calibrate chars/token from last-turn metrics
    proactive_calibration_min_prompt_chars: int = 100  # minimum prompt length to trust calibration
    compact_plugin_max_chars: int = 200  # cap for plugin prompt blocks in compact/fresh mode
    interrupted_turn_message_cap: int = (
        2048  # cap for the partial message in an interrupted-turn anchor
    )
    interrupted_turn_thought_cap: int = (
        512  # cap for the partial thought in an interrupted-turn anchor
    )
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
    cron: CronConfig = Field(default_factory=CronConfig)
    # Units an agent-initiated restart may name (the authorship
    # ``restart_enabled`` toggle still gates it). Empty means "own unit
    # only" — the persona's ``<name>.service``.
    restart_allowed_units: list[str] = Field(default_factory=list)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    prompt_blocks: PromptBlocksConfig = Field(default_factory=PromptBlocksConfig)

    @model_validator(mode="after")
    def _set_store_paths(self) -> HarnessConfig:
        if self.dispatch_store_path is None:
            self.dispatch_store_path = self.session_store_path.parent / "dispatch_store.jsonl"
        if self.wake_store_path is None:
            self.wake_store_path = self.session_store_path.parent / "wake_queue.jsonl"
        if self.cron.state_path is None:
            self.cron.state_path = self.session_store_path.parent / "cron_state.jsonl"
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
                "the legacy body field is no longer supported. "
                "Move that configuration to harness.plugins (or remove it to use the defaults)."
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
