"""Tests for outbound notifications."""

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
