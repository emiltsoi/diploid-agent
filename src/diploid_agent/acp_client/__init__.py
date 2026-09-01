"""ACP client: drive an ACP-compatible agent binary over stdio JSON-RPC."""

from diploid_agent.acp_client.client import (
    AcpClient,
    AcpError,
    AcpMcpError,
    AcpModelError,
    AcpPromptResult,
    AcpSessionStaleError,
    AcpTransportError,
    _acp_error_from_response,
    _load_windsurf_api_key,
    _normalize_model,
    _Prompt,
    _resolve_agent_bin,
)

__all__ = [
    "AcpClient",
    "AcpError",
    "AcpMcpError",
    "AcpModelError",
    "AcpPromptResult",
    "AcpSessionStaleError",
    "AcpTransportError",
    "_Prompt",
    "_acp_error_from_response",
    "_load_windsurf_api_key",
    "_normalize_model",
    "_resolve_agent_bin",
]
