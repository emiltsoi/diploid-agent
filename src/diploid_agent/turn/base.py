"""Shared base for the Turn* collaborator classes.

Each turn component is bound to a ``TurnController`` and reaches the runtime
and sibling components through it. ``TurnComponent`` holds the accessors that
were previously copied into every component.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diploid_agent.turn.controller import TurnController


class TurnComponent:
    """Runtime/controller accessors shared by the Turn* components."""

    def __init__(self, controller: TurnController) -> None:
        self.controller = controller

    @property
    def runtime(self) -> Any:
        return self.controller.runtime

    @property
    def acp_client(self) -> Any:
        return getattr(self.runtime, "acp_client", None)

    @property
    def _lock(self) -> Any:
        return self.runtime._lock

    @property
    def _chat_store(self) -> Any:
        return self.runtime._chat_store

    @property
    def _prompts(self) -> Any:
        return self.runtime._prompts

    @property
    def _mcp_skills(self) -> Any:
        return self.runtime._mcp_skills

    @property
    def _outbox(self) -> Any:
        return self.runtime._outbox

    @property
    def _runtime_metrics(self) -> Any:
        return self.runtime._runtime_metrics

    @property
    def _planning(self) -> Any:
        return self.runtime._planning

    @property
    def _subagent(self) -> Any:
        return self.runtime._subagent

    @property
    def context_builder(self) -> Any:
        return self.runtime.context_builder

    @property
    def engine(self) -> Any:
        return self.runtime.engine

    @property
    def session(self) -> Any:
        return self.controller.session

    @property
    def rehydrate(self) -> Any:
        return self.controller.rehydrate

    @property
    def _dispatch(self) -> Any:
        return self.controller._dispatch
