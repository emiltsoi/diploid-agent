"""Activity mirror: non-Telegram chats mirror results to the observer chat."""

import threading
from types import SimpleNamespace

from diploid_agent.models import ChatResult
from diploid_agent.runtime.outbox import RuntimeOutbox


class _Recorder:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, chat_id, text, *, reply_to_message_id=None):
        self.sent.append((chat_id, text))


def _outbox(mirror: int | None, notifier: _Recorder, *, outbox_delivery: bool = False) -> RuntimeOutbox:
    config = SimpleNamespace(
        harness=SimpleNamespace(
            notifications=SimpleNamespace(
                enabled=True,
                webhook_url=None,
                outbox_delivery=outbox_delivery,
                mesh_telegram_float=False,
                mirror_chat_id=mirror,
            ),
            telegram=SimpleNamespace(token="t"),
        )
    )
    return RuntimeOutbox(
        config=config,
        metrics=None,
        store={},
        lock=threading.RLock(),
        notifier_fn=lambda: notifier,
    )


def test_mesh_chat_mirrors_to_observer() -> None:
    rec = _Recorder()
    ob = _outbox(7945905361, rec)
    ob.emit_activity_mirror("mesh:vesper", ChatResult(reply="pong"))
    assert rec.sent == [("7945905361", "[mesh:vesper] pong")]


def test_telegram_origin_and_mirror_chat_not_mirrored() -> None:
    rec = _Recorder()
    ob = _outbox(7945905361, rec)
    ob.emit_activity_mirror("7945905361", ChatResult(reply="own turn"))
    ob.emit_activity_mirror("12345", ChatResult(reply="other tg chat"))
    assert rec.sent == []


def test_mirror_unset_is_silent() -> None:
    rec = _Recorder()
    ob = _outbox(None, rec)
    ob.emit_activity_mirror("mesh:vesper", ChatResult(reply="pong"))
    assert rec.sent == []


def test_async_delivery_path_also_mirrors() -> None:
    rec = _Recorder()
    ob = _outbox(7945905361, rec)
    ob._deliver_chat_result("mesh:vesper", ChatResult(reply="wake reply"))
    assert ("7945905361", "[mesh:vesper] wake reply") in rec.sent


def test_outbox_delivery_path_mirrors_exactly_once() -> None:
    """notify=True + outbox_delivery: _deliver_chat_result mirrors internally;
    the process() gate skips the second emit — the queue holds exactly one
    mirror for the observer chat."""
    rec = _Recorder()
    ob = _outbox(7945905361, rec, outbox_delivery=True)
    ob._deliver_chat_result("mesh:vesper", ChatResult(reply="wake reply"))
    popped = [
        (cid, r.reply)
        for cid, r in iter(lambda: ob.outbox_pop(return_chat_id=True), None)
    ]
    assert popped == [
        ("7945905361", "[mesh:vesper] wake reply"),
        ("mesh:vesper", "wake reply"),
    ]
