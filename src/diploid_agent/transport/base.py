"""Base Transport protocol."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from diploid_agent.config import NotificationsConfig, TaskConfig, TimerConfig, WakerConfig


@dataclass
class InboundMessage:
    """A normalized user message from any transport."""

    chat_id: str
    text: str
    reply_to: str | None = None
    reply_to_is_bot: bool | None = None
    reply_to_message_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutboundMessage:
    """A normalized message to send back to the user."""

    chat_id: str
    text: str
    reply_to_message_id: int | None = None
    notice: str | None = None


class RuntimeAPI(abc.ABC):
    """The runtime surface a Transport talks to.

    ConversationHarness and later AgentRuntime both satisfy this.
    """

    @abc.abstractmethod
    def process(
        self,
        chat_id: str,
        user_message: str,
        *,
        model: str | None = None,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
    ) -> Any:
        """Process a user message and return a ChatResult."""

    @abc.abstractmethod
    def continue_turn(self, dispatch_id: str, result: str) -> Any:
        """Resume after a background dispatch."""

    @abc.abstractmethod
    def stop(self, chat_id: str) -> Any:
        """Cancel a running turn."""

    @abc.abstractmethod
    def restart(self, chat_id: str) -> Any:
        """Kill the ACP child and start a fresh transport for this chat."""

    @abc.abstractmethod
    def graceful_service_restart(
        self,
        chat_id: str,
        service: str,
        reason: str = "",
    ) -> Any:
        """Schedule a graceful systemd restart of the named service."""

    @abc.abstractmethod
    def turn_status(self, chat_id: str, wait: float = 0.0) -> Any:
        """Return the current streaming state."""

    @abc.abstractmethod
    def status(self, chat_id: str) -> Any:
        """Return chat status."""

    @abc.abstractmethod
    def list_sessions(self, chat_id: str) -> Any:
        """List chat sessions."""

    @abc.abstractmethod
    def new_session(self, chat_id: str, model: str | None = None) -> Any:
        """Start a new session."""

    @abc.abstractmethod
    def resume_session(self, chat_id: str, session_number: int) -> Any:
        """Resume a session."""

    @abc.abstractmethod
    def branch_session(self, chat_id: str, session_number: int) -> Any:
        """Branch a session."""

    @abc.abstractmethod
    def switch_model(self, chat_id: str, model: str) -> Any:
        """Switch the model for a chat."""

    @abc.abstractmethod
    def dispatch(self, chat_id: str, context: str | None = None) -> Any:
        """Create a dispatch."""

    @abc.abstractmethod
    def wake(
        self,
        chat_id: str,
        event_id: str | None = None,
        reason: str | None = None,
        silent: bool | None = None,
    ) -> Any:
        """Wake a chat from a queued event."""

    @abc.abstractmethod
    def list_models(self) -> list[str]:
        """List supported models."""

    @abc.abstractmethod
    def get_metrics(self, chat_id: str | None = None) -> Any:
        """Return metrics."""

    def health(self) -> dict[str, Any]:
        """Return the current health of the runtime and its dependencies."""
        return {"status": "ok"}

    def get_prometheus_metrics(self) -> str:
        """Return metrics in Prometheus exposition format."""
        return ""

    def get_config(self) -> dict[str, Any]:
        """Return the current live runtime configuration."""
        raise NotImplementedError

    def update_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a partial runtime configuration update."""
        raise NotImplementedError

    def get_telegram_config(self) -> Any:
        """Return the current live Telegram configuration."""
        raise NotImplementedError

    @abc.abstractmethod
    def plan_create(
        self,
        name: str,
        description: str = "",
        chat_id: str | None = None,
        tasks: list[Any] | None = None,
    ) -> Any:
        """Create a new plan with the given tasks."""

    @abc.abstractmethod
    def plan_task_start(self, plan_id: str, task_id: str | None = None) -> Any:
        """Start a ready task in a plan."""

    @abc.abstractmethod
    def plan_task_done(
        self,
        plan_id: str,
        task_id: str,
        result: str = "",
        log: str = "",
    ) -> Any:
        """Manually mark a plan task as done."""

    @abc.abstractmethod
    def get_task_config(self) -> TaskConfig:
        """Return the current live task configuration."""

    @abc.abstractmethod
    def update_task_config(self, task_config: TaskConfig) -> str:
        """Update the live task configuration."""

    @abc.abstractmethod
    def get_waker_config(self) -> WakerConfig:
        """Return the current live waker configuration."""

    @abc.abstractmethod
    def update_waker_config(self, waker_config: WakerConfig) -> str:
        """Update the live waker configuration."""

    @abc.abstractmethod
    def get_timer_config(self) -> TimerConfig:
        """Return the current live timer configuration."""

    @abc.abstractmethod
    def update_timer_config(self, timer_config: TimerConfig) -> str:
        """Update the live timer configuration."""

    @abc.abstractmethod
    def get_notifications_config(self) -> NotificationsConfig:
        """Return the current live notifications configuration."""

    @abc.abstractmethod
    def update_notifications_config(self, notifications_config: NotificationsConfig) -> str:
        """Update the live notifications configuration."""

    def register_ingress_handler(self, protocol: str, handler: Any) -> None:
        """Register a protocol-specific inbound HTTP handler."""
        raise NotImplementedError

    def handle_ingress(self, protocol: str, request: Any) -> Any:
        """Dispatch an inbound HTTP request to the registered handler."""
        raise NotImplementedError


class Transport(abc.ABC):
    """A channel for receiving user messages and sending replies."""

    @abc.abstractmethod
    def start(self, runtime: RuntimeAPI) -> None:
        """Bind the transport to a runtime and begin listening."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop the transport."""

    @abc.abstractmethod
    def send(self, message: OutboundMessage) -> Any:
        """Send an outbound message and return any delivery metadata."""
