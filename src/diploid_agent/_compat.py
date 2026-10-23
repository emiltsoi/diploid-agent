"""Platform compatibility shims — Unix-only APIs behind one import.

The goal is a single seam: call sites swap ``import fcntl`` (or the bare
``os.*`` call) for the equivalent here, and the module resolves to the native
implementation at import time. Everything outside this file stays platform-
agnostic and the diff stays small.

Windows coverage:
- ``fcntl.flock`` (advisory file locks)      → ``msvcrt.locking`` byte-range lock
- ``fcntl.lockf`` (byte-range locks)         → ``msvcrt.locking`` at range
- ``os.kill(pid, 0)`` liveness probe         → ``OpenProcess`` + exit code
- ``os.killpg`` process-group kill           → ``taskkill /T /F`` process tree
- ``os.getuid``                              → absent (caller must tolerate -1)
- subprocess session-group kwargs            → ``CREATE_NEW_PROCESS_GROUP``

Note on flock semantics: ``fcntl.flock`` is whole-file; ``msvcrt.locking``
locks a byte range. We lock a dedicated byte far past any real data — the
lock files carry a payload (owner pid, heartbeat) that other handles read
and write, and a Windows range lock would block them. Contention semantics
are preserved: every locker targets the same byte.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_DARWIN = sys.platform == "darwin"

_WIN_ILLEGAL_CHARS = '<>:"|?*'
# NTFS rejects these basenames outright, case-insensitive and with or
# without an extension ("con.txt" is as illegal as "con").
_WIN_RESERVED_NAMES = (
    {"CON", "NUL", "AUX", "PRN"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

__all__ = [
    "IS_DARWIN",
    "IS_WINDOWS",
    "atomic_replace",
    "fcntl",
    "getuid",
    "kill_process_tree",
    "pid_alive",
    "safe_dir_name",
    "short_temp_dir",
    "subprocess_session_kwargs",
]


# --------------------------------------------------------------------------
# fcntl.flock / lockf
# --------------------------------------------------------------------------
if IS_WINDOWS:
    import msvcrt

    # Lock a single dedicated byte far past any real data. Unlike advisory
    # flock, a Windows byte-range lock blocks *reads and writes* through
    # other handles — so locking the data region would break readers of the
    # lock file's payload (owner pid / heartbeat) while held. Contention is
    # unchanged: every locker targets the same byte.
    _LOCK_OFFSET = 0x7FFF0000

    _fcntl_flags = {
        "LOCK_SH": 1,
        "LOCK_EX": 2,
        "LOCK_NB": 4,
        "LOCK_UN": 8,
    }

    def _locking_op(operation: int) -> int:
        """Map LOCK_* flags to the msvcrt.locking mode.

        LOCK_EX/LOCK_SH both map to an exclusive lock (msvcrt has no shared
        mode); LOCK_NB chooses the non-blocking variant, which raises
        OSError on contention just like flock(LOCK_NB). Blocking LK_LOCK
        retries for ~10s rather than blocking forever — every critical
        section here is sub-second, so that ceiling is not reached in
        practice.
        """
        if operation & _fcntl_flags["LOCK_UN"]:
            return msvcrt.LK_UNLCK
        if operation & _fcntl_flags["LOCK_NB"]:
            return msvcrt.LK_NBLCK
        return msvcrt.LK_LOCK

    def _flock(fd: int, operation: int) -> None:
        """Approximate fcntl.flock with msvcrt.locking."""
        # msvcrt.locking acts at the current file position; POSIX flock does
        # not move it. Save and restore so a caller who forgets to re-seek
        # does not write at the dedicated lock byte.
        pos = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, _locking_op(operation), 1)
        finally:
            os.lseek(fd, pos, os.SEEK_SET)

    def _lockf(fd: int, operation: int, length: int, start: int, whence: int) -> None:
        """POSIX lockf: byte-range lock. ``length == 0`` (lock to EOF) has no
        msvcrt.locking equivalent and raises instead of silently locking the
        wrong region."""
        if not length:
            raise NotImplementedError(
                "fcntl.lockf(length=0) lock-to-EOF has no Windows equivalent"
            )
        pos = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            os.lseek(fd, start, whence)
            msvcrt.locking(fd, _locking_op(operation), length)
        finally:
            os.lseek(fd, pos, os.SEEK_SET)

    class _FcntlModule:
        LOCK_SH = _fcntl_flags["LOCK_SH"]
        LOCK_EX = _fcntl_flags["LOCK_EX"]
        LOCK_NB = _fcntl_flags["LOCK_NB"]
        LOCK_UN = _fcntl_flags["LOCK_UN"]

        @staticmethod
        def flock(fd: int, operation: int) -> None:
            _flock(fd, operation)

        @staticmethod
        def lockf(
            fd: int,
            operation: int,
            length: int = 0,
            start: int = 0,
            whence: int = os.SEEK_SET,
        ) -> None:
            _lockf(fd, operation, length, start, whence)

    fcntl = _FcntlModule()
else:
    import fcntl


# --------------------------------------------------------------------------
# process liveness
# --------------------------------------------------------------------------
def pid_alive(pid: int) -> bool:
    """True if a process with this pid exists (and isn't a zombie)."""
    if not IS_WINDOWS:
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True
    import ctypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


# --------------------------------------------------------------------------
# process-group kill
# --------------------------------------------------------------------------
def kill_process_tree(pid: int, sig: int | None = None) -> None:
    """Kill a process and its children.

    POSIX: killpg on the group the child was started in. Windows: taskkill
    /T walks the tree. On Windows only force-kill exists — there is no
    SIGTERM equivalent for foreign processes; callers wanting grace should
    terminate() first and use this as the hammer.
    """
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
        return
    os.killpg(pid, sig if sig is not None else signal.SIGKILL)


def subprocess_session_kwargs() -> dict:
    """Kwargs that put a spawned child in its own killable process group."""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def atomic_replace(src, dst, *, attempts: int = 10, delay: float = 0.05) -> None:
    """os.replace with transient Windows lock retry.

    On Windows the destination of an atomic replace may be held open for a
    few ms by an indexer/AV scan or a lagging reader — ``os.replace`` then
    raises ``PermissionError`` where POSIX would succeed. A short bounded
    retry absorbs that transient; a persistent lock still raises.
    """
    import time

    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if not IS_WINDOWS or i == attempts - 1:
                raise
            time.sleep(delay * (i + 1))


def getuid() -> int:
    """os.getuid() or -1 where it doesn't exist (Windows)."""
    return os.getuid() if hasattr(os, "getuid") else -1


def safe_dir_name(name: str) -> str:
    """Map a chat_id to a filesystem-safe directory name.

    POSIX keeps the historic ``/``→``_`` mapping so existing session dirs
    keep their names. Windows additionally strips NTFS-illegal characters —
    ``:`` alone makes ``mesh:hermes`` an ADS-syntax path — plus trailing
    dots/spaces and reserved device basenames (``CON``, ``COM3``, …).
    """
    safe = name.replace("/", "_")
    if IS_WINDOWS:
        safe = (
            "".join("_" if ch in _WIN_ILLEGAL_CHARS or ord(ch) < 32 else ch for ch in safe).rstrip(
                ". "
            )
            or "_"
        )
        if safe.split(".", 1)[0].upper() in _WIN_RESERVED_NAMES:
            safe = "_" + safe
    return safe


def short_temp_dir() -> Path:
    """A short temp base for AF_UNIX paths (sun_path is ~108 bytes).

    POSIX falls back to /tmp; Windows uses %TEMP% — already short enough
    for socket names in practice, and definitely inside the limit once
    the digest-fallback kicks in for longer paths.
    """
    if IS_WINDOWS:
        return Path(tempfile.gettempdir())
    return Path("/tmp")
