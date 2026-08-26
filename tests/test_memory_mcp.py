"""Tests for the diploid-memory MCP server."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import respx
from httpx import Response

from diploid_agent.memory_mcp import MemoryMcpServer


def _exchange(monkeypatch, messages, chat_id="chat-1", harness_url="http://127.0.0.1:4003"):
    server = MemoryMcpServer(chat_id, harness_url)
    stdin_lines = [json.dumps(m) for m in messages]
    stdin = io.StringIO("\n".join(stdin_lines) + "\n")
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    server.run()
    stdout.seek(0)
    return [json.loads(line) for line in stdout if line.strip()]


def test_memory_mcp_lists_tools(tmp_path: Path, monkeypatch) -> None:
    responses = _exchange(
        monkeypatch,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ],
    )
    assert len(responses) == 2
    tools = responses[1]["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == {"memory_recall", "memory_retain", "memory_promote", "memory_status"}


@respx.mock
def test_memory_mcp_recalls_via_http(tmp_path: Path, monkeypatch) -> None:
    route = respx.post("http://127.0.0.1:4003/recall").mock(
        return_value=Response(200, json={"reply": "Memory from previous turns:\n\nfoo"})
    )
    responses = _exchange(
        monkeypatch,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "memory_recall",
                    "arguments": {"query": "foo"},
                },
            },
        ],
    )
    assert len(responses) == 2
    assert not responses[1]["result"]["isError"]
    assert "foo" in responses[1]["result"]["content"][0]["text"]
    assert route.called


@respx.mock
def test_memory_mcp_retains_via_http(tmp_path: Path, monkeypatch) -> None:
    route = respx.post("http://127.0.0.1:4003/retain").mock(
        return_value=Response(200, json={"reply": "Retained."})
    )
    responses = _exchange(
        monkeypatch,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "memory_retain",
                    "arguments": {"content": "We agreed on tea.", "tags": ["agreement"]},
                },
            },
        ],
    )
    assert len(responses) == 2
    assert not responses[1]["result"]["isError"]
    assert route.called
    assert json.loads(route.calls.last.request.content) == {
        "chat_id": "chat-1",
        "content": "We agreed on tea.",
        "tags": ["agreement"],
    }


@respx.mock
def test_memory_mcp_promotes_via_http(tmp_path: Path, monkeypatch) -> None:
    route = respx.post("http://127.0.0.1:4003/promote").mock(
        return_value=Response(200, json={"reply": "Promoted to persona memory."})
    )
    responses = _exchange(
        monkeypatch,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "memory_promote",
                    "arguments": {"fact": "The user prefers tea."},
                },
            },
        ],
    )
    assert len(responses) == 2
    assert not responses[1]["result"]["isError"]
    assert route.called
    assert json.loads(route.calls.last.request.content) == {
        "chat_id": "chat-1",
        "message": "The user prefers tea.",
    }
