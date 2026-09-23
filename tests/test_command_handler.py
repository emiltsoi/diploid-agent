"""Tests for CommandHandler's HTTP fallback: {chat_id} path templating."""

import httpx

from diploid_agent.transport.command_handler import CommandHandler


def _handler_with_mock_client() -> tuple[CommandHandler, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"reply": "ok"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return (
        CommandHandler(harness_url="http://harness", api_key="k", client=client),
        requests,
    )


def test_post_substitutes_chat_id_placeholder_in_path():
    handler, requests = _handler_with_mock_client()
    result = handler.call(
        method="new_session",
        chat_id="42",
        http_path="/new/{chat_id}",
    )
    assert result == {"reply": "ok"}
    assert requests[0].url.path == "/new/42"
    assert requests[0].method == "POST"


def test_get_substitutes_chat_id_placeholder_in_path():
    handler, requests = _handler_with_mock_client()
    handler.call(
        method="status",
        chat_id="42",
        http_path="/status/{chat_id}",
        http_method="GET",
    )
    assert requests[0].url.path == "/status/42"
