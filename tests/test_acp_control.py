"""Tests for the stable ACP control socket (Issue C: socket resurrection).

The control socket lives at a deterministic per-service path so a child's
baked ``DIPLOID_CONTROL_SOCKET`` survives transport generations and process
restarts. ``ensure_listening()`` re-binds after any ``close()``, and a
connect-probe keeps a new binder from stealing a *live* socket.
"""

import json
import os
import socket
import stat
import threading
import time
import uuid
from pathlib import Path

import pytest

import diploid_agent.acp_client.control as control_mod
from diploid_agent.acp_client.control import (
    ControlListener,
    ControlSocketInUseError,
)


def _name() -> str:
    return f"test-ctl-{uuid.uuid4().hex[:8]}.service"


def _listener(name: str, calls: list) -> ControlListener:
    return ControlListener(
        service_name=name,
        on_service_restart=lambda service, reason: calls.append((service, reason)),
        control_timeout=30.0,
        watchdog_timeout=30.0,
    )


def _send_restart(path: Path, service: str, reason: str = "test") -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(5.0)
        s.connect(str(path))
        s.sendall(
            json.dumps({"action": "restart_service", "service": service, "reason": reason}).encode(
                "utf-8"
            )
        )
        return json.loads(s.recv(1024).decode("utf-8"))


def test_stable_path_and_dir_permissions() -> None:
    """The socket lives at /tmp/diploid-ctl-<service>/control.sock with 0700."""
    name = _name()
    listener = _listener(name, [])
    try:
        path = listener.socket_path
        assert path.name == "control.sock"
        assert path.parent.name == f"diploid-ctl-{name}"
        mode = stat.S_IMODE(path.parent.stat().st_mode)
        assert mode == 0o700
        assert listener.env()["DIPLOID_CONTROL_SOCKET"] == str(path)
        assert path.exists()
    finally:
        listener.close()


def test_close_then_ensure_listening_rebinds_same_path() -> None:
    """bind -> close -> ensure_listening() re-binds at the same stable path."""
    name = _name()
    calls: list = []
    listener = _listener(name, calls)
    try:
        path = listener.socket_path
        listener.close()
        assert not path.exists()

        listener.ensure_listening()
        assert listener.socket_path == path
        assert path.exists()

        # A restart request still reaches the handler after the re-bind.
        ack = _send_restart(path, name, reason="rebind")
        assert ack["status"] == "ok"
        assert ack["pid"] == os.getpid()
        deadline = time.time() + 2.0
        while not calls and time.time() < deadline:
            time.sleep(0.02)
        assert calls == [(name, "rebind")]
    finally:
        listener.close()


def test_env_path_is_stable_across_generations() -> None:
    """env() reports the same path no matter how often the listener rebinds."""
    name = _name()
    listener = _listener(name, [])
    try:
        first = listener.env()["DIPLOID_CONTROL_SOCKET"]
        listener.close()
        listener.ensure_listening()
        second = listener.env()["DIPLOID_CONTROL_SOCKET"]
        assert first == second
    finally:
        listener.close()


def test_stale_socket_file_is_rebound() -> None:
    """A dead socket file at the stable path is unlinked and rebound."""
    name = _name()
    first = _listener(name, [])
    path = first.socket_path
    first.close()

    # Simulate a crashed owner: a leftover file nobody listens on.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

    calls: list = []
    second = _listener(name, calls)
    try:
        assert second.socket_path == path
        ack = _send_restart(path, name)
        assert ack["status"] == "ok"
    finally:
        second.close()


def test_foreign_live_socket_is_not_stolen(monkeypatch) -> None:
    """A connect-probe to a live foreign listener aborts instead of stealing."""
    monkeypatch.setattr(control_mod, "_PROBE_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(control_mod, "_PROBE_LIVE_RETRY_SECONDS", 0.3)
    monkeypatch.setattr(control_mod, "_PROBE_LIVE_RETRY_INTERVAL", 0.05)

    name = _name()
    path = Path(control_mod.tempfile.gettempdir()) / f"diploid-ctl-{name}" / "control.sock"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)

    # A "foreign" owner: a live listener that accepts but never answers the
    # ping, and keeps draining its backlog so probes can always connect.
    foreign = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    foreign.bind(str(path))
    foreign.listen(4)
    foreign.settimeout(0.1)
    draining = threading.Event()

    def _drain() -> None:
        while not draining.is_set():
            try:
                conn, _ = foreign.accept()
            except (TimeoutError, OSError):
                continue
            conn.close()

    drain_thread = threading.Thread(target=_drain, daemon=True)
    drain_thread.start()
    try:
        listener = _listener(name, [])
        try:
            # __init__ soft-fails; the explicit ensure_listening() must abort.
            with pytest.raises(ControlSocketInUseError):
                listener.ensure_listening()
            # The foreign socket still owns the path — we never bound or unlinked.
            assert listener._bound_stat is None
            assert listener._control_listener_thread is None
            assert path.exists()
        finally:
            listener.close()
    finally:
        draining.set()
        drain_thread.join(timeout=1.0)
        foreign.close()
        path.unlink(missing_ok=True)


def test_in_process_listener_shares_live_socket() -> None:
    """A second in-process listener for the same service shares the live socket.

    This is the per-task ACP engine case: the child's env points at the stable
    path, which is served by the primary listener carrying the same callback.
    """
    name = _name()
    calls_a: list = []
    calls_b: list = []
    first = _listener(name, calls_a)
    second = _listener(name, calls_b)
    try:
        assert second.socket_path == first.socket_path
        assert second.env()["DIPLOID_CONTROL_SOCKET"] == str(first.socket_path)
        assert second._control_listener_thread is None

        ack = _send_restart(first.socket_path, name, reason="shared")
        assert ack["status"] == "ok"
        deadline = time.time() + 2.0
        while not calls_a and time.time() < deadline:
            time.sleep(0.02)
        assert calls_a == [(name, "shared")]
        assert calls_b == []
    finally:
        second.close()
        first.close()


def test_close_does_not_remove_stable_dir() -> None:
    """The per-service dir persists across restarts; only the socket is unlinked."""
    name = _name()
    listener = _listener(name, [])
    socket_dir = listener.socket_path.parent
    listener.close()
    assert socket_dir.is_dir()
    assert not listener.socket_path.exists()


def test_no_callback_means_no_listener() -> None:
    """Without a restart callback the listener stays inert (and unbound)."""
    listener = ControlListener(
        service_name=_name(),
        on_service_restart=None,
        control_timeout=30.0,
        watchdog_timeout=30.0,
    )
    try:
        assert listener._control_listener_thread is None
        # ensure_listening() is a no-op without a callback.
        listener.ensure_listening()
        assert listener._control_listener_thread is None
    finally:
        listener.close()
