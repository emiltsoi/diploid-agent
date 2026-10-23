"""Direct coverage for the platform compat seam (diploid_agent._compat).

Every helper gets at least a smoke test plus the semantic contract that
matters to callers. Platform-specific assertions are guarded so the file
runs green on both POSIX and Windows.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from diploid_agent._compat import (
    IS_WINDOWS,
    atomic_replace,
    fcntl,
    getuid,
    kill_process_tree,
    pid_alive,
    safe_dir_name,
    short_temp_dir,
    subprocess_session_kwargs,
)


# --------------------------------------------------------------------------
# safe_dir_name
# --------------------------------------------------------------------------
def test_safe_dir_name_posix_mapping() -> None:
    """The historic ``/``→``_`` mapping holds everywhere — existing session
    dirs must not be renamed."""
    assert safe_dir_name("a/b") == "a_b"
    assert safe_dir_name("12345") == "12345"


@pytest.mark.skipif(not IS_WINDOWS, reason="NTFS rules only apply on Windows")
def test_safe_dir_name_windows_illegal_chars() -> None:
    assert safe_dir_name("mesh:hermes") == "mesh_hermes"
    assert safe_dir_name("a<b>c") == "a_b_c"
    assert safe_dir_name("q|x?y") == "q_x_y"
    # trailing dots/spaces are rejected by Explorer and most APIs
    assert safe_dir_name("room.") == "room"
    assert safe_dir_name("room. ") == "room"
    # all-illegal input still produces a usable name
    assert safe_dir_name(":") != ""


@pytest.mark.skipif(not IS_WINDOWS, reason="device names only matter on Windows")
def test_safe_dir_name_windows_reserved_names() -> None:
    for reserved in ("con", "CON", "nul", "aux", "prn", "com1", "COM9", "lpt3"):
        result = safe_dir_name(reserved)
        assert result.upper().split(".", 1)[0] not in {
            "CON",
            "NUL",
            "AUX",
            "PRN",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }, f"{reserved!r} mapped to reserved name {result!r}"
    # extension doesn't rescue a reserved basename: "con.txt" is illegal too
    assert not safe_dir_name("con.txt").upper().startswith("CON.")
    # non-reserved names containing reserved prefixes stay untouched
    assert safe_dir_name("converge") == "converge"
    assert safe_dir_name("com10") == "com10"  # only COM1-9 are reserved


# --------------------------------------------------------------------------
# flock / lockf
# --------------------------------------------------------------------------
def test_flock_roundtrip(tmp_path) -> None:
    path = tmp_path / "x.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def test_flock_preserves_fd_position(tmp_path) -> None:
    """fcntl.flock must not move the file position — a shim that seeks to a
    dedicated lock byte without restoring would corrupt caller I/O (and on
    Windows, grow the file into a sparse lockfile)."""
    path = tmp_path / "x.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        os.write(fd, b"payload-at-zero")
        os.lseek(fd, 7, os.SEEK_SET)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert os.lseek(fd, 0, os.SEEK_CUR) == 7
        fcntl.flock(fd, fcntl.LOCK_UN)
        assert os.lseek(fd, 0, os.SEEK_CUR) == 7
    finally:
        os.close(fd)
    # locking a dedicated out-of-band byte must not enlarge the payload file
    assert path.stat().st_size < 100


@pytest.mark.skipif(not IS_WINDOWS, reason="msvcrt shim specifics")
def test_flock_windows_contention_nb(tmp_path) -> None:
    """A second handle must fail LOCK_NB while the lock is held, succeed after release."""
    path = tmp_path / "x.lock"
    fd1 = os.open(path, os.O_RDWR | os.O_CREAT)
    fd2 = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd1, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(OSError):
            fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd1, fcntl.LOCK_UN)
        fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)  # now free — must not raise
    finally:
        os.close(fd1)
        os.close(fd2)


@pytest.mark.skipif(not IS_WINDOWS, reason="shim behavior; POSIX lockf defers to OS")
def test_lockf_length_zero_fails_loudly(tmp_path) -> None:
    """lock-to-EOF has no msvcrt equivalent — fail loudly rather than lock
    the wrong region silently."""
    fd = os.open(tmp_path / "x.lock", os.O_RDWR | os.O_CREAT)
    try:
        with pytest.raises(NotImplementedError):
            fcntl.lockf(fd, fcntl.LOCK_EX, 0)
    finally:
        os.close(fd)


@pytest.mark.skipif(not IS_WINDOWS, reason="shim behavior; POSIX lockf defers to OS")
def test_lockf_range_roundtrip(tmp_path) -> None:
    fd = os.open(tmp_path / "x.lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 4, 0, os.SEEK_SET)
        fcntl.lockf(fd, fcntl.LOCK_UN, 4, 0, os.SEEK_SET)
        assert os.lseek(fd, 0, os.SEEK_CUR) == 0  # position restored
    finally:
        os.close(fd)


# --------------------------------------------------------------------------
# process helpers
# --------------------------------------------------------------------------
def test_pid_alive_self() -> None:
    assert pid_alive(os.getpid()) is True


def test_pid_alive_dead() -> None:
    """A pid far outside the plausible range must report dead on both platforms."""
    assert pid_alive(0x3FFFFFFF) is False


def test_kill_process_tree(tmp_path) -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        **subprocess_session_kwargs(),
    )
    try:
        assert pid_alive(proc.pid)
        kill_process_tree(proc.pid, signal.SIGTERM if not IS_WINDOWS else None)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
    time.sleep(0.2)
    assert proc.poll() is not None


def test_subprocess_session_kwargs_shape() -> None:
    kwargs = subprocess_session_kwargs()
    if IS_WINDOWS:
        assert "creationflags" in kwargs
        assert "start_new_session" not in kwargs
    else:
        assert kwargs.get("start_new_session") is True


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def test_atomic_replace_success(tmp_path) -> None:
    src = tmp_path / "new"
    dst = tmp_path / "old"
    src.write_text("v2", encoding="utf-8")
    dst.write_text("v1", encoding="utf-8")
    atomic_replace(src, dst)
    assert dst.read_text(encoding="utf-8") == "v2"
    assert not src.exists()


def test_atomic_replace_missing_src_raises(tmp_path) -> None:
    with pytest.raises(OSError):
        atomic_replace(tmp_path / "nope", tmp_path / "dst")


def test_getuid_returns_int() -> None:
    uid = getuid()
    assert isinstance(uid, int)
    if IS_WINDOWS:
        assert uid == -1


def test_short_temp_dir_exists() -> None:
    assert short_temp_dir().is_dir()


# --------------------------------------------------------------------------
# control-channel degradation (no AF_UNIX)
# --------------------------------------------------------------------------
def test_control_listener_degrades_without_af_unix(monkeypatch, caplog) -> None:
    """Hosts without socket.AF_UNIX (some Windows builds) must not reach
    _bind/_probe_socket: ``Path()`` == ``Path(".")`` whose ``exists()`` is
    true, and the probe would die on ``socket.AF_UNIX`` AttributeError.
    Regression test for the boot-log AttributeError on win32."""
    import logging

    from diploid_agent.acp_client.control import ControlListener

    monkeypatch.delattr(socket, "AF_UNIX", raising=False)
    listener = ControlListener(
        service_name="test-degraded.service",
        on_service_restart=lambda service, reason: None,
        control_timeout=30.0,
        watchdog_timeout=30.0,
    )
    with caplog.at_level(logging.ERROR):
        listener.ensure_listening()  # must be a no-op, no bind attempt
    listener.close()
    assert listener.socket_path == Path()
    assert not any("Failed to bind" in r.message for r in caplog.records)
