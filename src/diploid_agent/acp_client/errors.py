"""Typed ACP JSON-RPC exceptions and error classification."""

from __future__ import annotations

import json
from typing import Any


class AcpError(RuntimeError):
    """Base class for ACP JSON-RPC errors returned by the child process."""

    def __init__(self, method: str, error: dict[str, Any]) -> None:
        self.method = method
        self.code = error.get("code")
        self.message = error.get("message", "")
        self.data = error.get("data") or {}
        self.error = error
        super().__init__(f"ACP {method} failed: {error}")


class AcpTransportError(AcpError):
    """The ACP transport itself did not respond or the child process died."""

    def __init__(self, method: str, error: dict[str, Any] | None = None, msg: str = "") -> None:
        if error is None:
            error = {"message": msg}
        super().__init__(method, error)


class AcpSessionStaleError(AcpError):
    """The ACP session id is no longer valid on an otherwise healthy transport."""


class AcpModelError(AcpError):
    """The requested model is not available to the ACP child."""


class AcpMcpError(AcpError):
    """The MCP server configuration was rejected by the ACP child."""


def _acp_error_from_response(method: str, error: dict[str, Any]) -> AcpError:
    """Return the appropriate typed exception for a JSON-RPC error response.

    Classifier rules (from observed `devin acp` 3000.6.7 payloads):

    - ``-32602`` / ``Invalid params`` with an ``mcpServers`` / ``McpServer``
      mention in the data or message -> ``AcpMcpError``.
    - ``-32002`` / ``Resource not found``
      - ``data.uri`` / message starts with ``Model not found`` or mentions
        ``model`` -> ``AcpModelError``.
      - ``data.uri`` / message contains ``Session not found`` or a session id
        and does *not* look like a model error -> ``AcpSessionStaleError``.
    - Other ACP protocol or transport failures -> ``AcpError``.
    """
    code = error.get("code")
    message = (error.get("message") or "").lower()
    data = error.get("data") or {}
    data_str = json.dumps(data, default=str).lower()
    combined = f"{message} {data_str}"

    if code == -32602:
        if "mcp" in combined or "mcpservers" in combined:
            return AcpMcpError(method, error)
        return AcpError(method, error)

    if code == -32002:
        # The `data.uri` field is sometimes the human-readable detail string.
        data_uri = str(data.get("uri", "")).lower()
        detail = data_uri or message

        # `devin acp 3000.6.7` reports a stale/non-existent session as
        # "Model not found: <model>. Available models: " with an *empty* model
        # list. A genuine "model not found" has a non-empty available-models list.
        available_models_str = ""
        if "available models:" in detail:
            available_models_str = detail.split("available models:", 1)[1].strip(" \"\'[]").strip()

        looks_like_model_error = (
            detail.startswith("model not found")
            or ("model" in detail and "not found" in detail)
        )
        if looks_like_model_error:
            if available_models_str == "":
                return AcpSessionStaleError(method, error)
            return AcpModelError(method, error)

        if detail.startswith("session not found") or (
            "session" in detail and "not found" in detail
        ):
            return AcpSessionStaleError(method, error)

        # Generic resource not found that does not name a session is probably a
        # model or other configuration error, not a stale session.
        if "model" in detail and available_models_str != "":
            return AcpModelError(method, error)
        if "session" in detail or (method and method.startswith("session/")):
            return AcpSessionStaleError(method, error)

    return AcpError(method, error)
