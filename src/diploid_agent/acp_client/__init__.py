"""ACP client: drive an ACP-compatible agent binary over stdio JSON-RPC."""

from diploid_agent.acp_client.client import AcpClient
from diploid_agent.acp_client.errors import (
    AcpError,
    AcpMcpError,
    AcpModelError,
    AcpSessionStaleError,
    AcpTransportError,
    _acp_error_from_response,
)
from diploid_agent.acp_client.lifecycle import AcpLifecycleLog, AcpRestartHistory
from diploid_agent.acp_client.types import AcpPromptResult, _Prompt
from diploid_agent.acp_client.utils import (
    _load_windsurf_api_key,
    _normalize_model,
    _resolve_agent_bin,
)

__all__ = [
    "AcpClient",
    "AcpError",
    "AcpLifecycleLog",
    "AcpMcpError",
    "AcpModelError",
    "AcpPromptResult",
    "AcpRestartHistory",
    "AcpSessionStaleError",
    "AcpTransportError",
    "_Prompt",
    "_acp_error_from_response",
    "_load_windsurf_api_key",
    "_normalize_model",
    "_resolve_agent_bin",
]
