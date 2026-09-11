"""ACP control socket listener for restart requests from the ACP subprocess."""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import stat
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from diploid_agent.acp_client.errors import AcpTransportError

logger = logging.getLogger(__name__)

_CONTROL_DIR_MODE = 0o700
_PROBE_TIMEOUT_SECONDS = 0.5
# Connect errors that prove nobody is listening on the path. Anything else —
# including a connect timeout or BlockingIOError on a full accept backlog,
# which both mean a live listener is present — is treated as "live" so a probe
# can never steal a socket a live listener still owns.
_PROBE_STALE_ERRORS = (ConnectionRefusedError, FileNotFoundError)
# A live foreign listener is retried briefly before we give up: during a
# `systemctl restart` overlap the old process can still hold the socket for a
# moment while systemd stops it.
_PROBE_LIVE_RETRY_SECONDS = 5.0
_PROBE_LIVE_RETRY_INTERVAL = 0.25


class ControlSocketInUseError(AcpTransportError):
    """The stable control-socket path is held by a live foreign listener."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            "session/new",
            msg=f"ACP control socket {path} is held by a live listener",
        )


class ControlListener:
    """Listen on a private Unix socket for restart requests from the ACP subprocess.

    The fake systemctl wrapper installed in the isolated ACP HOME sends
    JSON-RPC-ish restart requests here instead of running real systemctl.

    The socket lives at a stable per-service path
    (``/tmp/diploid-ctl-<service>/control.sock``) so the
    ``DIPLOID_CONTROL_SOCKET`` baked into a child's environment stays valid
    across transport generations *and* process restarts: a child spawned by an
    older generation still reaches the current listener.
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
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", service_name or "unknown.service")
        self._control_socket_dir = Path(tempfile.gettempdir()) / f"diploid-ctl-{safe_name}"
        self._control_socket_path = self._control_socket_dir / "control.sock"
        self._control_listener_running = False
        self._control_listener_thread: threading.Thread | None = None
        self._control_socket: socket.socket | None = None
        # (st_dev, st_ino) of the socket file this instance bound, so close()
        # never unlinks a socket rebound by a newer owner of the stable path.
        self._bound_stat: tuple[int, int] | None = None
        try:
            self._start()
        except ControlSocketInUseError as exc:
            # A foreign process still owns the path (e.g. the old process during
            # a systemd restart overlap). Do not fail construction over it;
            # ensure_listening() retries on the next transport start and fails
            # loudly there if the path is still held.
            logger.warning(
                "ACP control socket %s is held by a live listener; control "
                "channel stays down until it frees",
                self._control_socket_path,
                exc_info=exc,
            )

    @property
    def socket_path(self) -> Path:
        return self._control_socket_path

    def call_timeout(self) -> float:
        """Cap control-call waits at the watchdog threshold to keep stalls short."""
        return max(60.0, min(self._control_timeout, self._watchdog_timeout))

    def ensure_listening(self) -> None:
        """Re-bind the stable control socket if the listener is not running.

        Idempotent: a live in-process listener is kept. Raises
        ``ControlSocketInUseError`` when the path is held by a live *foreign*
        listener, so callers can abort the transport start instead of handing
        the child a socket owned by another process.
        """
        self._start()

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
            self._control_socket_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_control_dir_permissions()
            if self._control_socket_path.exists():
                state = self._probe_socket()
                # A live listener owned by *this* process (e.g. a per-task ACP
                # engine sharing the service name) already serves the path with
                # the same restart callback; no second bind is needed.
                if state == "live_self":
                    logger.info(
                        "ACP control socket %s already served in-process; sharing",
                        self._control_socket_path,
                    )
                    return None
                deadline = time.monotonic() + _PROBE_LIVE_RETRY_SECONDS
                while state == "live_foreign" and time.monotonic() < deadline:
                    time.sleep(_PROBE_LIVE_RETRY_INTERVAL)
                    state = self._probe_socket()
                if state == "live_self":
                    logger.info(
                        "ACP control socket %s already served in-process; sharing",
                        self._control_socket_path,
                    )
                    return None
                if state != "stale":
                    raise ControlSocketInUseError(self._control_socket_path)
                self._control_socket_path.unlink(missing_ok=True)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(self._control_socket_path))
            sock.listen(1)
            sock.settimeout(1.0)
            st = self._control_socket_path.lstat()
            self._bound_stat = (st.st_dev, st.st_ino)
            return sock
        except ControlSocketInUseError:
            raise
        except Exception as exc:
            logger.warning("Failed to bind ACP control socket", exc_info=exc)
            return None

    def _ensure_control_dir_permissions(self) -> None:
        """Enforce that the control socket directory is private to this user.

        ``mkdir`` with ``exist_ok=True`` does not set the mode on an existing
        path and the directory may be pre-created by a local attacker with lax
        permissions. Verify ownership and mode; raise if we cannot make it
        exclusively ours.
        """
        path = self._control_socket_dir
        try:
            st = path.lstat()
        except OSError as exc:
            raise ControlSocketInUseError(self._control_socket_path) from exc

        if st.st_uid != os.getuid():
            raise ControlSocketInUseError(self._control_socket_path)

        current_mode = stat.S_IMODE(st.st_mode)
        if current_mode != _CONTROL_DIR_MODE:
            try:
                os.chmod(path, _CONTROL_DIR_MODE)
            except OSError as exc:
                raise ControlSocketInUseError(self._control_socket_path) from exc
            try:
                st = path.lstat()
            except OSError as exc:
                raise ControlSocketInUseError(self._control_socket_path) from exc
            if stat.S_IMODE(st.st_mode) != _CONTROL_DIR_MODE:
                raise ControlSocketInUseError(self._control_socket_path)

    def _probe_socket(self) -> str:
        """Probe the socket path: ``stale``, ``live_self``, or ``live_foreign``.

        A connect failure means the socket is stale. A successful connect means
        *something* is listening; if the ack cannot be read or parsed we still
        report ``live_foreign`` rather than stealing a live socket's path.
        """
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(_PROBE_TIMEOUT_SECONDS)
            probe.connect(str(self._control_socket_path))
        except _PROBE_STALE_ERRORS:
            # Refused or vanished mid-race: nobody is listening.
            probe.close()
            return "stale"
        except OSError:
            # Inconclusive (e.g. EAGAIN / timeout on a full backlog — both mean
            # a live listener). Report live rather than risk stealing the path.
            probe.close()
            return "live_foreign"
        try:
            with probe:
                probe.sendall(json.dumps({"action": "control_ping"}).encode("utf-8"))
                data = probe.recv(1024)
            try:
                ack = json.loads(data.decode("utf-8")) if data else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                ack = {}
            if ack.get("pid") == os.getpid():
                return "live_self"
            return "live_foreign"
        except OSError:
            return "live_foreign"

    def _unlink_if_owned(self) -> None:
        """Unlink the socket file only if it is still the one this instance bound."""
        bound = self._bound_stat
        if bound is None:
            return
        try:
            st = self._control_socket_path.lstat()
        except OSError:
            self._bound_stat = None
            return
        if (st.st_dev, st.st_ino) != bound:
            # The stable path was rebound by a newer owner; not ours to remove.
            return
        try:
            self._control_socket_path.unlink()
        except OSError:
            pass
        self._bound_stat = None

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
                        conn.sendall(
                            json.dumps(
                                {
                                    "status": "ok",
                                    "pid": os.getpid(),
                                    "service": self._service_name,
                                }
                            ).encode("utf-8")
                        )
                    except Exception as exc:
                        logger.warning("ACP control socket request failed", exc_info=exc)
        finally:
            try:
                sock.close()
            except OSError:
                pass
            self._unlink_if_owned()

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
        self._unlink_if_owned()
        # The stable dir intentionally persists across process restarts.

    def env(self) -> dict[str, str]:
        """Return environment variables needed by the ACP subprocess wrapper."""
        return {
            "DIPLOID_CONTROL_SOCKET": str(self._control_socket_path),
            "DIPLOID_SERVICE_NAME": self._service_name or "unknown.service",
        }
