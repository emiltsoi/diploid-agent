"""Post-resume external-turn reconciliation tests."""

from __future__ import annotations

import threading

from diploid_agent.models import SessionRecord
from diploid_agent.turn.reconcile import ExternalTurn, TurnReconcile


def _update(kind: str, text: str) -> dict:
    return {"sessionUpdate": kind, "content": [{"type": "text", "text": text}]}


def test_reconstruct_turns_pairs_chunks() -> None:
    updates = [
        _update("user_message_chunk", "hello "),
        _update("user_message_chunk", "world"),
        _update("agent_thought_chunk", "thinking"),
        _update("agent_message_chunk", "hi "),
        _update("agent_message_chunk", "there"),
        _update("tool_call", "ignored"),
        _update("user_message_chunk", "next"),
        _update("agent_message_chunk", "reply2"),
    ]
    turns = TurnReconcile.reconstruct_turns(updates)
    assert [(t.user, t.assistant) for t in turns] == [
        ("hello world", "hi there"),
        ("next", "reply2"),
    ]


def test_reconstruct_turns_keeps_unanswered_tail() -> None:
    updates = [
        _update("user_message_chunk", "q1"),
        _update("agent_message_chunk", "a1"),
        _update("user_message_chunk", "interrupted"),
    ]
    turns = TurnReconcile.reconstruct_turns(updates)
    assert len(turns) == 2
    assert turns[1] == ExternalTurn(user="interrupted", assistant="")


class _FakeBackend:
    def __init__(self, entries=None):
        self.entries = entries or []
        self.appended = []

    def load_transcript(self):
        return list(self.entries)

    def append_transcript(self, user_message, assistant_reply, *, external=False):
        self.appended.append((user_message, assistant_reply, external))
        self.entries.append({"role": "user", "content": user_message})
        self.entries.append({"role": "assistant", "content": assistant_reply})


class _FakeEngine:
    def __init__(self, updates):
        self._updates = updates

    def drain_session_replay(self, session_id):
        out, self._updates = self._updates, []
        return out


class _FakePlugins:
    def __init__(self):
        self.external = []

    def on_external_turns(self, chat_id, turns):
        self.external.append(turns)


class _FakeRuntime:
    def __init__(self, backend, updates):
        self.engine = _FakeEngine(updates)
        self._backend = backend
        self._plugins = _FakePlugins()
        self.lifecycle_log = None
        self._lock = threading.RLock()

    def call_engine_unlocked(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def _memory_manager(self, chat_id):
        return type("MM", (), {"backend": self._backend})()


class _FakeController:
    def __init__(self, runtime):
        self.runtime = runtime


def _record() -> SessionRecord:
    return SessionRecord(
        chat_id="iris:devin",
        session_number=1,
        session_id="liberating-sulfur",
        model="swe-2-high",
        persona="iris",
        cwd=".",
        created_at=0.0,
        updated_at=0.0,
        turn_number=6,
    )


def test_reconcile_adopts_foreign_turns() -> None:
    backend = _FakeBackend(
        entries=[
            {"role": "user", "content": "t1"},
            {"role": "assistant", "content": "a1"},
        ]
    )
    updates = [
        _update("user_message_chunk", "t1"),
        _update("agent_message_chunk", "a1"),
        _update("user_message_chunk", "desktop says hi"),
        _update("agent_message_chunk", "desktop reply"),
    ]
    runtime = _FakeRuntime(backend, updates)
    recon = TurnReconcile(_FakeController(runtime))

    report = recon.reconcile_after_resume("iris:devin", "liberating-sulfur", _record())

    assert report is not None and report.count == 1
    assert backend.appended == [("desktop says hi", "desktop reply", True)]
    assert runtime._plugins.external == [
        [{"user": "desktop says hi", "assistant": "desktop reply"}]
    ]

    # Second drain with nothing new — idempotent, no double-append.
    report2 = recon.reconcile_after_resume("iris:devin", "liberating-sulfur", _record())
    assert report2 is None
    assert len(backend.appended) == 1


def test_reconcile_count_fallback_when_tail_unmatched() -> None:
    backend = _FakeBackend(
        entries=[
            {"role": "user", "content": "edited-or-truncated"},
            {"role": "assistant", "content": "a1"},
        ]
    )
    updates = [
        _update("user_message_chunk", "t1"),
        _update("agent_message_chunk", "a1"),
        _update("user_message_chunk", "foreign"),
        _update("agent_message_chunk", "freply"),
        _update("user_message_chunk", "foreign2"),
        _update("agent_message_chunk", "freply2"),
    ]
    runtime = _FakeRuntime(backend, updates)
    recon = TurnReconcile(_FakeController(runtime))

    report = recon.reconcile_after_resume("iris:devin", "liberating-sulfur", _record())

    assert report is not None and report.count == 2
    assert report.aligned is False


def test_reconcile_no_replay_returns_none() -> None:
    runtime = _FakeRuntime(_FakeBackend(), [])
    recon = TurnReconcile(_FakeController(runtime))
    assert recon.reconcile_after_resume("iris:devin", "s", _record()) is None
