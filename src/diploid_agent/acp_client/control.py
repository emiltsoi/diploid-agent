"""ACP control socket listener for restart requests from the ACP subprocess."""

from __future__ import annotations

import json
import logging
import socket
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class ControlListener:
    """Listen on a private Unix socket for restart requests from the ACP subprocess.

    The fake systemctl wrapper installed in the isolated ACP HOME sends
    JSON-RPC-ish restart requests here instead of running real systemctl.
    """

    def __init__(
        self,
        service_name: str,
        on_service_restart: Callable[[str, str], None] | None,
        control_timeout: float,
        watchdog_timeout: float,
    ) -> None:
        self._service_name = service_name
        self._on_service_restart = on_service_restart
        self._control_timeout = control_timeout
        self._watchdog_timeout = watchdog_timeout
        self._control_socket_dir = Path(tempfile.mkdtemp(prefix="acp-ctl-"))
        self._control_socket_path = self._control_socket_dir / "control.sock"
        self._control_listener_running = False
        self._control_listener_thread: threading.Thread | None = None
        self._control_socket: socket.socket | None = None
        self._start()

    @property
    def socket_path(self) -> Path:
        return self._control_socket_path

    def call_timeout(self) -> float:
        """Cap control-call waits at the watchdog threshold to keep stalls short."""
        return max(60.0, min(self._control_timeout, self._watchdog_timeout))

    def _start(self) -> None:
        """Create and bind the control socket, then start the listener thread."""
        if self._on_service_restart is None:
            return
        if self._control_listener_thread is not None and self._control_listener_thread.is_alive():
            return
        self._control_listener_running = True
        sock = self._bind()
        if sock is None:
            return
        self._control_socket = sock
        self._control_listener_thread = threading.Thread(
            target=self._listen,
            args=(sock,),
            daemon=True,
        )
        self._control_listener_thread.start()

    def _bind(self) -> socket.socket | None:
        """Create and bind the control socket used by the fake systemctl wrapper."""
        try:
            self._control_socket_path.parent.mkdir(parents=True, exist_ok=True)
            if self._control_socket_path.exists():
                self._control_socket_path.unlink()
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(self._control_socket_path))
            sock.listen(1)
            sock.settimeout(1.0)
            return sock
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to bind ACP control socket: %s", exc)
            return None

    def _listen(self, sock: socket.socket) -> None:
        """Listen for restart requests from the fake systemctl wrapper."""
        try:
            while self._control_listener_running:
                try:
                    conn, _ = sock.accept()
                except TimeoutError:
                    continue
                with conn:
                    try:
                        data = b""
                        while True:
                            chunk = conn.recv(4096)
                            if not chunk:
                                break
                            data += chunk
                            if len(chunk) < 4096:
                                break
                        if not data:
                            continue
                        msg = json.loads(data.decode("utf-8"))
                        action = msg.get("action")
                        service = msg.get("service") or self._service_name or "unknown.service"
                        reason = msg.get("reason", "")
                        if action == "restart_service" and self._on_service_restart is not None:
                            self._on_service_restart(service, reason)
                        conn.sendall(json.dumps({"status": "ok"}).encode("utf-8"))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("ACP control socket request failed: %s", exc)
        finally:
            try:
                sock.close()
            except OSError:
                pass
            try:
                if self._control_socket_path.exists():
                    self._control_socket_path.unlink()
            except OSError:
                pass

    def close(self) -> None:
        """Signal the control listener to stop and clean up its socket."""
        self._control_listener_running = False
        if self._control_listener_thread is not None and self._control_listener_thread.is_alive():
            self._control_listener_thread.join(timeout=2.0)
        self._control_listener_thread = None
        if self._control_socket is not None:
            try:
                self._control_socket.close()
            except OSError:
                pass
            self._control_socket = None
        try:
            if self._control_socket_path.exists():
                self._control_socket_path.unlink()
        except OSError:
            pass
        try:
            self._control_socket_dir.rmdir()
        except OSError:
            pass

    def env(self) -> dict[str, str]:
        """Return environment variables needed by the ACP subprocess wrapper."""
        return {
            "DIPLOID_CONTROL_SOCKET": str(self._control_socket_path),
            "DIPLOID_SERVICE_NAME": self._service_name or "unknown.service",
        }
