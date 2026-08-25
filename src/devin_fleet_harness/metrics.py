"""Lightweight, thread-safe Prometheus-style metrics collector."""

from __future__ import annotations

import threading
from collections import defaultdict


def _escape_label_value(value: object) -> str:
    """Escape a Prometheus label value."""
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("\n", "\\n")
    return f'"{text}"'


class MetricsCollector:
    """A minimal Prometheus-compatible metrics registry.

    Supports counters and gauges with optional labels. No external dependencies.
    """

    def __init__(self, prefix: str = "harness") -> None:
        self.prefix = prefix
        self._lock = threading.Lock()
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._gauges: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(dict)

    def inc(self, name: str, value: float = 1.0, **labels: object) -> None:
        """Increment a counter metric."""
        label_key = self._label_key(labels)
        with self._lock:
            self._counters[name][label_key] += value

    def set(self, name: str, value: float, **labels: object) -> None:
        """Set a gauge metric."""
        label_key = self._label_key(labels)
        with self._lock:
            self._gauges[name][label_key] = value

    def get(self, name: str, **labels: object) -> float:
        """Return the current value of a counter or gauge."""
        label_key = self._label_key(labels)
        with self._lock:
            if name in self._counters:
                return self._counters[name].get(label_key, 0.0)
            return self._gauges[name].get(label_key, 0.0)

    def render(self) -> str:
        """Render all metrics in Prometheus exposition format."""
        lines: list[str] = []
        with self._lock:
            for name, values in sorted(self._counters.items()):
                full_name = f"{self.prefix}_{name}"
                lines.append(f"# HELP {full_name} Total count")
                lines.append(f"# TYPE {full_name} counter")
                for label_key, value in sorted(values.items()):
                    labels_str = self._format_labels(label_key)
                    lines.append(f"{full_name}{labels_str} {value}")
            for name, values in sorted(self._gauges.items()):
                full_name = f"{self.prefix}_{name}"
                lines.append(f"# HELP {full_name} Current value")
                lines.append(f"# TYPE {full_name} gauge")
                for label_key, value in sorted(values.items()):
                    labels_str = self._format_labels(label_key)
                    lines.append(f"{full_name}{labels_str} {value}")
        return "\n".join(lines) + "\n"

    def _label_key(self, labels: dict[str, object]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((k, str(v)) for k, v in labels.items()))

    def _format_labels(self, label_key: tuple[tuple[str, str], ...]) -> str:
        if not label_key:
            return ""
        parts = [f"{k}={_escape_label_value(v)}" for k, v in label_key]
        return "{" + ",".join(parts) + "}"
