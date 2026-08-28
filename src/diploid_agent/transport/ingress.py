"""Pluggable ingress extension point for diploid-agent.

A transport/comm protocol (e.g. mesh) can register an IngressHandler. The
harness calls the handler for any request routed to /ingress/{protocol} or
to the protocol's well-known aliases.
"""

from __future__ import annotations

import abc
import importlib
import logging
from typing import Any

from fastapi import Request, Response

logger = logging.getLogger(__name__)


class IngressHandler(abc.ABC):
    """Abstract base for a protocol-specific inbound request handler."""

    @abc.abstractmethod
    async def handle(self, request: Request) -> Response:
        """Handle an inbound request and return a Response."""
        ...


def load_ingress_handler(module_path: str, runtime: Any | None = None) -> IngressHandler:
    """Load an IngressHandler factory from a module and call it.

    The module must export either:
      - an `Ingress` or `IngressHandler` class (instantiated with no args,
        or with `runtime` if its constructor accepts it)
      - a `create_handler()` or `create_handler(runtime)` function
    """
    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        raise ImportError(f"Could not load ingress module {module_path!r}: {exc}") from exc

    import inspect

    for attr in ("Ingress", "IngressHandler"):
        cls = getattr(mod, attr, None)
        if cls is not None and callable(cls) and issubclass(cls, IngressHandler):
            sig = inspect.signature(cls)
            params = list(sig.parameters.keys())
            # For classes, inspect.signature omits `self` and shows only __init__ args.
            if runtime is not None and len(params) == 1 and params[0] == "runtime":
                return cls(runtime)
            return cls()

    factory = getattr(mod, "create_handler", None)
    if callable(factory):
        sig = inspect.signature(factory)
        params = list(sig.parameters.keys())
        if runtime is not None and len(params) == 1 and params[0] == "runtime":
            return factory(runtime)
        return factory()

    raise ImportError(
        f"Ingress module {module_path!r} must export Ingress, IngressHandler, or create_handler"
    )
