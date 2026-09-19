"""Tests for outbound notifications."""

import json
from unittest.mock import MagicMock

import httpx

from diploid_agent.notifier import NoopNotifier, TelegramNotifier, WebhookNotifier


def test_noop_notifier_returns_none() -> None:
    n = NoopNotifier()
    assert n.send("chat-1", "hello") is None


def test_telegram_notifier_calls_send_message(monkeypatch) -> None:
    client = MagicMock()
    client.post.return_value.json.return_value = {"ok": True, "result": {"message_id": 42}}
    client.post.return_value.raise_for_status = MagicMock()

    notifier = TelegramNotifier("test-token", client=client)
    result = notifier.send("12345", "Task done.")

    assert client.post.call_count == 1
    args, kwargs = client.post.call_args
    url = args[0]
    assert url == "https://api.telegram.org/bottest-token/sendMessage"
    assert kwargs["data"]["chat_id"] == "12345"
    assert kwargs["data"]["text"] == "Task done."
    assert result == 42


def test_webhook_notifier_posts_json(monkeypatch) -> None:
    client = MagicMock()
    client.post.return_value.raise_for_status = MagicMock()

    notifier = WebhookNotifier("http://example.com/webhook", client=client)
    notifier.send("chat-1", "hello")

    args, kwargs = client.post.call_args
    url = args[0]
    assert url == "http://example.com/webhook"
    assert kwargs["json"]["chat_id"] == "chat-1"
    assert kwargs["json"]["text"] == "hello"


def test_telegram_notifier_typing(monkeypatch) -> None:
    client = MagicMock()
    client.post.return_value.json.return_value = {"ok": True, "result": True}
    client.post.return_value.raise_for_status = MagicMock()

    notifier = TelegramNotifier("test-token", client=client)
    notifier.typing("12345")

    args, kwargs = client.post.call_args
    url = args[0]
    assert url == "https://api.telegram.org/bottest-token/sendChatAction"
    assert kwargs["data"]["chat_id"] == "12345"
    assert kwargs["data"]["action"] == "typing"


def test_telegram_notifier_retries_read_timeout(monkeypatch) -> None:
    client = MagicMock()
    client.post.side_effect = [
        httpx.ReadTimeout("read timeout"),
        MagicMock(
            json=MagicMock(return_value={"ok": True, "result": {"message_id": 42}}),
            raise_for_status=MagicMock(),
        ),
    ]
    monkeypatch.setattr("diploid_agent.notifier.time.sleep", lambda *_: None)

    notifier = TelegramNotifier("test-token", client=client)
    result = notifier.send("12345", "Task done.")

    assert client.post.call_count == 2
    assert result == 42


def test_telegram_notifier_retries_502_and_logs_body(monkeypatch) -> None:
    client = MagicMock()
    bad_response = MagicMock()
    bad_response.status_code = 502
    bad_response.json = MagicMock(return_value={"ok": False, "description": "Bad Gateway"})
    good_response = MagicMock()
    good_response.json = MagicMock(return_value={"ok": True, "result": {"message_id": 7}})
    good_response.raise_for_status = MagicMock()
    client.post.side_effect = [
        httpx.HTTPStatusError("502", request=MagicMock(), response=bad_response),
        good_response,
    ]
    monkeypatch.setattr("diploid_agent.notifier.time.sleep", lambda *_: None)

    notifier = TelegramNotifier("test-token", client=client)
    result = notifier.send("12345", "Task done.")

    assert client.post.call_count == 2
    assert result == 7


def test_telegram_notifier_default_client_uses_30s_timeout(monkeypatch) -> None:
    client = MagicMock()
    client.post.return_value.json.return_value = {"ok": True, "result": {"message_id": 1}}
    client.post.return_value.raise_for_status = MagicMock()
    monkeypatch.setattr("diploid_agent.notifier.httpx.Client", lambda **kw: client)
    notifier = TelegramNotifier("test-token")
    _ = notifier.send("12345", "hello")

    assert client is notifier.client
    assert client.post.call_count == 1


def _tg_response(payload: dict, status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.json = MagicMock(return_value=payload)
    if status >= 400:
        resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(f"{status}", request=MagicMock(), response=resp)
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


def test_update_task_board_sends_first_then_edits(tmp_path) -> None:
    client = MagicMock()
    client.post.return_value = _tg_response({"ok": True, "result": {"message_id": 42}})

    notifier = TelegramNotifier("test-token", client=client, state_dir=tmp_path)
    assert notifier.update_task_board("12345", "☐ a") is True
    assert notifier.update_task_board("12345", "◐ a") is True

    calls = client.post.call_args_list
    assert calls[0][0][0].endswith("/sendMessage")
    assert "parse_mode" not in calls[0][1]["data"]
    assert calls[1][0][0].endswith("/editMessageText")
    assert calls[1][1]["data"]["message_id"] == 42
    assert calls[1][1]["data"]["text"] == "◐ a"


def test_update_task_board_persists_id_across_restart(tmp_path) -> None:
    client = MagicMock()
    client.post.return_value = _tg_response({"ok": True, "result": {"message_id": 42}})

    TelegramNotifier("test-token", client=client, state_dir=tmp_path).update_task_board(
        "12345", "☐ a"
    )
    assert (tmp_path / "12345.board.json").exists()

    client2 = MagicMock()
    client2.post.return_value = _tg_response({"ok": True, "result": True})
    notifier2 = TelegramNotifier("test-token", client=client2, state_dir=tmp_path)
    assert notifier2.update_task_board("12345", "☑ a") is True

    call = client2.post.call_args_list[0]
    assert call[0][0].endswith("/editMessageText")
    assert call[1]["data"]["message_id"] == 42


def test_update_task_board_resends_and_rekeys_on_deleted_message(tmp_path) -> None:
    client = MagicMock()
    client.post.side_effect = [
        _tg_response({"ok": True, "result": {"message_id": 42}}),
        _tg_response(
            {
                "ok": False,
                "error_code": 400,
                "description": "Bad Request: message to edit not found",
            },
            status=400,
        ),
        _tg_response({"ok": True, "result": {"message_id": 99}}),
    ]

    notifier = TelegramNotifier("test-token", client=client, state_dir=tmp_path)
    notifier.update_task_board("12345", "☐ a")
    assert notifier.update_task_board("12345", "☑ a") is True

    calls = client.post.call_args_list
    assert calls[1][0][0].endswith("/editMessageText")
    assert calls[2][0][0].endswith("/sendMessage")
    assert json.loads((tmp_path / "12345.board.json").read_text())["message_id"] == 99


def test_update_task_board_not_modified_counts_as_ok(tmp_path) -> None:
    client = MagicMock()
    client.post.side_effect = [
        _tg_response({"ok": True, "result": {"message_id": 42}}),
        _tg_response(
            {
                "ok": False,
                "error_code": 400,
                "description": "Bad Request: message is not modified",
            },
            status=400,
        ),
    ]

    notifier = TelegramNotifier("test-token", client=client, state_dir=tmp_path)
    notifier.update_task_board("12345", "☐ a")
    assert notifier.update_task_board("12345", "☐ a") is True
    assert client.post.call_count == 2


def test_update_task_board_keeps_id_on_transient_error(tmp_path) -> None:
    client = MagicMock()
    client.post.side_effect = [
        _tg_response({"ok": True, "result": {"message_id": 42}}),
        _tg_response(
            {"ok": False, "error_code": 500, "description": "Internal Server Error"},
            status=500,
        ),
        _tg_response({"ok": True, "result": True}),
    ]

    notifier = TelegramNotifier("test-token", client=client, state_dir=tmp_path)
    notifier.update_task_board("12345", "☐ a")
    assert notifier.update_task_board("12345", "◐ a") is False
    assert notifier.update_task_board("12345", "◐ a") is True

    calls = client.post.call_args_list
    assert calls[1][0][0].endswith("/editMessageText")
    assert calls[2][0][0].endswith("/editMessageText")


def test_update_task_board_rejects_non_telegram_chat_id(tmp_path) -> None:
    client = MagicMock()
    notifier = TelegramNotifier("test-token", client=client, state_dir=tmp_path)

    assert notifier.update_task_board("mesh:vesper", "☐ a") is False
    assert notifier.update_task_board("not-a-chat", "☐ a") is False
    client.post.assert_not_called()
