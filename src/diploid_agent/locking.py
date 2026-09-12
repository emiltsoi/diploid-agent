"""Shared method decorator for running component methods under the runtime RLock."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any


def locked(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run the decorated method while holding ``self._lock``.

    Every class this decorates must expose ``_lock`` (attribute or property)
    returning the shared ``threading.RLock`` — by convention the
    ``AgentRuntime`` lock.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper
