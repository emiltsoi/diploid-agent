"""TurnSession: per-chat session lifecycle helpers."""

from __future__ import annotations

import functools
import logging
import time
from typing import TYPE_CHECKING, Any

from diploid_agent.engine import TurnRequest
from diploid_agent.models import ChatResult, SessionRecord
from diploid_agent.plugins.contexts import (
    SessionActiveContext,
    SessionArchiveContext,
    SessionClearContext,
    SessionStartContext,
)

if TYPE_CHECKING:
    from diploid_agent.turn.controller import TurnController

logger = logging.getLogger(__name__)


def _locked(method):
    """Run a TurnSession method under the runtime RLock."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.runtime._lock:
            return method(self, *args, **kwargs)

    return wrapper


class TurnSession:
    """Session management for a single chat."""

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
    def context_builder(self) -> Any:
        return self.runtime.context_builder

    @property
    def engine(self) -> Any:
        return self.runtime.engine

    def _can_resume_record(self, chat_id: str, record: SessionRecord, use_model: str) -> bool:
        """Return True if the ACP session for this record can be resumed."""
        if not record or not record.session_id:
            return False
        if record.model and record.model != use_model:
            return False
        if record.last_stop_reason == "timeout":
            return False
        current_mcp = sorted(self.runtime._active_mcp_server_names(chat_id))
        record_mcp = sorted(record.enabled_mcp_servers or [])
        if current_mcp != record_mcp:
            return False
        current_skills = sorted(self.runtime._active_skill_names(chat_id))
        record_skills = sorted(record.enabled_skills or [])
        return current_skills == record_skills

    def _finalize_session_activation(
        self,
        chat_id: str,
        record: SessionRecord,
        reply: str,
        system_message: str,
        reply_prefix: str,
        notice: str | None = None,
    ) -> ChatResult:
        """Record the turn, append the record, notify plugins, and return the result."""
        record.turn_number += 1
        self.runtime._memory_manager(chat_id).record_turn(
            user_message=f"[system: {system_message}]",
            reply=reply,
            model=record.model,
            session_number=record.session_number,
            turn_number=record.turn_number,
            notice=notice,
        )

        transition = self.runtime._check_chat_memory_transition(chat_id, record)
        if transition:
            notice = transition if notice is None else f"{notice}\n\n{transition}"

        self.runtime._append_record(record)
        self.runtime._prune_and_compact(chat_id)
        self.runtime._plugins.after_session_active(
            chat_id,
            SessionActiveContext(chat_id=chat_id, record=record),
        )
        return ChatResult(
            reply=f"{reply_prefix}\n\n{reply.strip()}",
            notice=notice,
            session_id=record.session_id,
            session_number=record.session_number,
            turn_number=record.turn_number,
        )

    def _start_fresh_session(
        self,
        chat_id: str,
        *,
        desired_model: str,
        kind: str,
        label: str,
        system_message: str,
        reply_prefix: str,
        old_model: str | None = None,
        clear_active: bool = False,
        plugin_overrides: dict[str, bool] | None = None,
    ) -> ChatResult:
        """Archive any active session and start a fresh ACP session for the chat."""
        record = self.runtime._active_record(chat_id)
        current_model = self.runtime._model(record)
        if record and desired_model == current_model and kind == "switch_model":
            return ChatResult(reply=f"Already using model `{desired_model}` for this chat.")

        use_model = desired_model or current_model

        if record:
            self.runtime._plugins.before_session_archive(
                chat_id,
                SessionArchiveContext(chat_id=chat_id, old_record=record),
            )
            self.runtime._archive_active_session(chat_id, record)

        if clear_active:
            self.runtime._plugins.before_session_clear(
                chat_id,
                SessionClearContext(chat_id=chat_id, old_record=record),
            )
            self.runtime._clear_active_session(chat_id)

        user_message = (
            "Continue the conversation as your true self. "
            "Acknowledge that you are ready to continue. "
            "Do not claim to know the name of the model serving you. "
            "Do not sign your reply."
        )
        session_number = self.runtime._next_session_number(chat_id)
        start_ctx = self.runtime._plugins.before_session_start(
            chat_id,
            SessionStartContext(
                chat_id=chat_id,
                kind=kind,
                user_message=user_message,
                model=use_model,
                session_number=session_number,
                old_model=old_model,
                skill_names=set(self.runtime._active_skill_names(chat_id)),
                mcp_servers=self.runtime._active_mcp_servers(chat_id),
            ),
        )
        if isinstance(start_ctx, ChatResult):
            return start_ctx
        use_model = start_ctx.model or use_model
        user_message = start_ctx.user_message

        pctx = self.runtime.context_builder.build_first(
            chat_id,
            user_message,
            record,
            model=use_model,
            skill_names=start_ctx.skill_names,
            mcp_names=None,
        )
        prompt = pctx.prompt
        notice = pctx.notice
        memory_flags = pctx.memory_flags
        use_model = pctx.model or use_model

        result, session_id = self.runtime._call_unlocked(
            self.runtime._start_new_session,
            chat_id,
            prompt,
            use_model,
            mcp_servers=start_ctx.mcp_servers or self.runtime._active_mcp_servers(chat_id),
            skill_names=start_ctx.skill_names or set(self.runtime._active_skill_names(chat_id)),
        )
        reply = result.reply

        new_record = self.runtime._create_record(
            chat_id,
            session_number,
            session_id,
            use_model,
            reply,
            memory_flags,
            label=label,
        )
        new_record.enabled_mcp_servers = self.runtime._active_mcp_server_names(chat_id)
        new_record.enabled_skills = sorted(self.runtime._active_skill_names(chat_id))
        if plugin_overrides is not None:
            new_record.plugin_overrides = plugin_overrides
        elif record is not None:
            new_record.plugin_overrides = record.plugin_overrides

        self.runtime._chat_state(chat_id).sessions[new_record.session_number] = new_record
        return self._finalize_session_activation(
            chat_id,
            new_record,
            reply,
            system_message,
            reply_prefix,
            notice=notice,
        )

    @_locked
    def switch_model(self, chat_id: str, model: str) -> ChatResult:
        """Switch the model for a chat by starting a fresh Devin session."""
        record = self.runtime._active_record(chat_id)
        current_model = self.runtime._model(record)
        return self._start_fresh_session(
            chat_id,
            desired_model=model,
            kind="switch_model",
            label=f"switched to {model}",
            system_message=f"switched to model {model}",
            reply_prefix=f"Now running on model `{model}`.",
            old_model=current_model,
        )

    @_locked
    def new_session(self, chat_id: str, model: str | None = None) -> ChatResult:
        """Start a fresh ACP session for a chat, clearing the active context."""
        record = self.runtime._active_record(chat_id)
        plugin_overrides = record.plugin_overrides if record else None
        return self._start_fresh_session(
            chat_id,
            desired_model=model or self.runtime._model(record),
            kind="new",
            label="new session",
            system_message="new session started",
            reply_prefix="New session started.",
            clear_active=True,
            plugin_overrides=plugin_overrides,
        )

    @_locked
    def resume_session(self, chat_id: str, session_number: int) -> ChatResult:
        """Resume an archived session as the active one."""
        state = self.runtime._chat_state(chat_id)
        source = state.sessions.get(session_number)
        if source is None:
            return ChatResult(reply=f"Session {session_number} not found for this chat.")

        active = self.runtime._active_record(chat_id)
        if active and active.session_number == session_number:
            return ChatResult(reply=f"Session {session_number} is already active.")

        # Archive the current active session before replacing it.
        if active:
            self.runtime._plugins.before_session_archive(
                chat_id,
                SessionArchiveContext(chat_id=chat_id, old_record=active),
            )
            self.runtime._archive_active_session(chat_id, active)

        # Copy the source session into the active directory.
        self.runtime._copy_session_dir(
            self.runtime._archive_dir(chat_id, session_number), self.runtime._chat_dir(chat_id)
        )

        source_mcp_names = (
            source.enabled_mcp_servers
            if source.enabled_mcp_servers is not None
            else sorted(self.runtime.mcp.default_enabled_names())
        )
        source_skill_names = (
            set(source.enabled_skills)
            if source.enabled_skills is not None
            else self.runtime._active_skill_names(chat_id)
        )

        resumed_id: str | None = None
        reply = ""
        memory_flags: dict[str, bool] = {}
        use_model = source.model

        if self.runtime.config.engine.acp_resume_enabled:
            try:
                logger.debug("Attempting ACP session resume for %s", source.session_id)
                resumed_id = self.runtime._call_unlocked(
                    self.runtime.engine.resume_session,
                    source.session_id,
                    cwd=self.runtime._chat_dir(chat_id),
                    model=source.model,
                    mcp_servers=self.runtime.mcp.enabled_servers(chat_id, source_mcp_names),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to resume ACP session %s: %s", source.session_id, exc)

        if not resumed_id:
            # Fall back to the legacy probe/rehydrate path.
            try:
                alive = self.runtime._call_unlocked(
                    self.runtime.engine.session_alive, source.session_id
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to probe ACP session %s: %s", source.session_id, exc)
                alive = False

            if alive:
                source.updated_at = time.time()
                source.cwd = str(self.runtime._chat_dir(chat_id))
                source.enabled_mcp_servers = source_mcp_names
                source.enabled_skills = sorted(source_skill_names)
                self.runtime.skills.sync_to_chat(
                    chat_id, self.runtime._chat_dir(chat_id), source_skill_names
                )
                self.runtime._chat_state(chat_id).sessions[source.session_number] = source
                reply = "Resumed existing session."
            else:
                user_message = "Continue the conversation as your true self."
                start_ctx = self.runtime._plugins.before_session_start(
                    chat_id,
                    SessionStartContext(
                        chat_id=chat_id,
                        kind="resume",
                        user_message=user_message,
                        model=source.model,
                        session_number=source.session_number,
                        skill_names=source_skill_names,
                        mcp_servers=self.runtime.mcp.enabled_servers(chat_id, source_mcp_names),
                    ),
                )
                if isinstance(start_ctx, ChatResult):
                    return start_ctx
                use_model = start_ctx.model or source.model
                user_message = start_ctx.user_message

                pctx = self.runtime.context_builder.build_first(
                    chat_id,
                    user_message,
                    active,
                    model=use_model,
                    skill_names=start_ctx.skill_names,
                    mcp_names=None,
                    rehydrated=True,
                )
                prompt = pctx.prompt
                memory_flags = pctx.memory_flags
                use_model = pctx.model or use_model
                result, session_id = self.runtime._call_unlocked(
                    self.runtime._start_new_session,
                    chat_id,
                    prompt,
                    use_model,
                    mcp_servers=start_ctx.mcp_servers
                    or self.runtime.mcp.enabled_servers(chat_id, source_mcp_names),
                    skill_names=start_ctx.skill_names or source_skill_names,
                )
                reply = result.reply
                source.session_id = session_id
                source.updated_at = time.time()
                source.cwd = str(self.runtime._chat_dir(chat_id))
                source.chat_memory_exceeded = memory_flags.get("chat_memory_exceeded", False)
                source.persona_memory_exceeded = memory_flags.get("persona_memory_exceeded", False)
                source.enabled_mcp_servers = source_mcp_names
                source.enabled_skills = sorted(source_skill_names)
                source.model = use_model
        else:
            # ACP resume succeeded; send a follow-up on the resumed session.
            user_message = "Continue the conversation as your true self."
            start_ctx = self.runtime._plugins.before_session_start(
                chat_id,
                SessionStartContext(
                    chat_id=chat_id,
                    kind="resume",
                    user_message=user_message,
                    model=source.model,
                    session_number=source.session_number,
                    skill_names=source_skill_names,
                    mcp_servers=self.runtime.mcp.enabled_servers(chat_id, source_mcp_names),
                ),
            )
            if isinstance(start_ctx, ChatResult):
                return start_ctx
            use_model = start_ctx.model or source.model
            user_message = start_ctx.user_message

            pctx = self.runtime.context_builder.build_follow_up(
                chat_id,
                user_message,
                source,
                skill_names=source_skill_names,
            )
            prompt = pctx.prompt
            memory_flags = pctx.memory_flags
            use_model = pctx.model or use_model
            request = TurnRequest(
                prompt=prompt,
                cwd=self.runtime._chat_dir(chat_id),
                model=use_model,
                mcp_servers=None,
                soft_timeout=self.runtime.config.engine.soft_timeout,
            )
            result = self.runtime.call_engine_unlocked(
                self.runtime.engine.prompt,
                request,
                session_id=resumed_id,
            )
            reply = result.reply
            source.session_id = resumed_id
            source.updated_at = time.time()
            source.cwd = str(self.runtime._chat_dir(chat_id))
            source.chat_memory_exceeded = memory_flags.get("chat_memory_exceeded", False)
            source.persona_memory_exceeded = memory_flags.get("persona_memory_exceeded", False)
            source.enabled_mcp_servers = source_mcp_names
            source.enabled_skills = sorted(source_skill_names)
            source.model = use_model
            self.runtime.skills.sync_to_chat(
                chat_id, self.runtime._chat_dir(chat_id), source_skill_names
            )

        return self._finalize_session_activation(
            chat_id,
            source,
            reply,
            "resumed session",
            f"Resumed session {session_number}.",
        )

    @_locked
    def branch_session(self, chat_id: str, session_number: int) -> ChatResult:
        """Branch from an archived session, creating a new active session."""
        state = self.runtime._chat_state(chat_id)
        source = state.sessions.get(session_number)
        if source is None:
            return ChatResult(reply=f"Session {session_number} not found for this chat.")

        active = self.runtime._active_record(chat_id)
        if active:
            self.runtime._plugins.before_session_archive(
                chat_id,
                SessionArchiveContext(chat_id=chat_id, old_record=active),
            )
            self.runtime._archive_active_session(chat_id, active)

        # Copy the source session into the active directory.
        self.runtime._copy_session_dir(
            self.runtime._archive_dir(chat_id, session_number), self.runtime._chat_dir(chat_id)
        )

        user_message = "Continue the conversation as your true self."
        new_number = self.runtime._next_session_number(chat_id)
        start_ctx = self.runtime._plugins.before_session_start(
            chat_id,
            SessionStartContext(
                chat_id=chat_id,
                kind="branch",
                user_message=user_message,
                model=source.model,
                session_number=new_number,
                skill_names=set(source.enabled_skills)
                if source.enabled_skills is not None
                else self.runtime._active_skill_names(chat_id),
                mcp_servers=self.runtime.mcp.enabled_servers(
                    chat_id,
                    source.enabled_mcp_servers
                    if source.enabled_mcp_servers is not None
                    else sorted(self.runtime.mcp.default_enabled_names()),
                ),
            ),
        )
        if isinstance(start_ctx, ChatResult):
            return start_ctx
        use_model = start_ctx.model or source.model
        user_message = start_ctx.user_message

        source_mcp_names = (
            [s["name"] for s in start_ctx.mcp_servers]
            if start_ctx.mcp_servers is not None
            else source.enabled_mcp_servers
            if source.enabled_mcp_servers is not None
            else sorted(self.runtime.mcp.default_enabled_names())
        )
        source_skill_names = (
            start_ctx.skill_names
            if start_ctx.skill_names is not None
            else set(source.enabled_skills)
            if source.enabled_skills is not None
            else self.runtime._active_skill_names(chat_id)
        )

        resumed_id: str | None = None
        if self.runtime.config.engine.acp_resume_enabled and self._can_resume_record(
            chat_id, source, use_model
        ):
            try:
                logger.debug(
                    "Attempting ACP session resume for branch of %s", source.session_id
                )
                resumed_id = self.runtime._call_unlocked(
                    self.runtime.engine.resume_session,
                    source.session_id,
                    cwd=self.runtime._chat_dir(chat_id),
                    model=use_model,
                    mcp_servers=start_ctx.mcp_servers
                    or self.runtime.mcp.enabled_servers(chat_id, source_mcp_names),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to resume ACP session for branch %s: %s", chat_id, exc
                )

        if resumed_id:
            pctx = self.runtime.context_builder.build_follow_up(
                chat_id,
                user_message,
                source,
                skill_names=source_skill_names,
            )
            prompt = pctx.prompt
            notice = pctx.notice
            memory_flags = pctx.memory_flags
            use_model = pctx.model or use_model
            request = TurnRequest(
                prompt=prompt,
                cwd=self.runtime._chat_dir(chat_id),
                model=use_model,
                mcp_servers=None,
                soft_timeout=self.runtime.config.engine.soft_timeout,
            )
            result = self.runtime.call_engine_unlocked(
                self.runtime.engine.prompt,
                request,
                session_id=resumed_id,
            )
            session_id = resumed_id
            reply = result.reply
        else:
            pctx = self.runtime.context_builder.build_first(
                chat_id,
                user_message,
                active,
                model=use_model,
                skill_names=start_ctx.skill_names,
                mcp_names=None,
            )
            prompt = pctx.prompt
            notice = pctx.notice
            memory_flags = pctx.memory_flags
            use_model = pctx.model or use_model
            result, session_id = self.runtime._call_unlocked(
                self.runtime._start_new_session,
                chat_id,
                prompt,
                use_model,
                mcp_servers=start_ctx.mcp_servers
                or self.runtime.mcp.enabled_servers(chat_id, source_mcp_names),
                skill_names=start_ctx.skill_names or source_skill_names,
            )
            reply = result.reply

        new_record = self.runtime._create_record(
            chat_id,
            new_number,
            session_id,
            use_model,
            reply,
            memory_flags,
            parent=session_number,
            label=f"branch of {session_number}",
        )
        new_record.enabled_mcp_servers = source_mcp_names
        new_record.enabled_skills = sorted(source_skill_names)
        new_record.plugin_overrides = source.plugin_overrides
        self.runtime._chat_state(chat_id).sessions[new_record.session_number] = new_record
        return self._finalize_session_activation(
            chat_id,
            new_record,
            reply,
            "branched session",
            f"Branched from session {session_number}.",
            notice=notice,
        )
