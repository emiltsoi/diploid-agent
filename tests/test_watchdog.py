import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import respx
from httpx import Response

sys.path.insert(0, str(Path(__file__).parent.parent / "probes"))
import diploid_harness_watchdog as watchdog


@respx.mock
def test_watchdog_rollback_before_restart() -> None:
    respx.get("http://127.0.0.1:4003/health").mock(
        side_effect=[
            Response(200, json={"status": "ok"}),
            Response(503, json={"status": "degraded"}),
            Response(503, json={"status": "degraded"}),
            Response(503, json={"status": "degraded"}),
            Response(200, json={"status": "ok"}),
        ]
    )
    rollback = respx.post("http://127.0.0.1:4003/config/rollback").respond(
        200, json={"reply": "ok"}
    )
    respx.post("http://127.0.0.1:4003/plugin-incidents").respond(
        200, json={"reply": "incident recorded"}
    )
    with patch("subprocess.run") as mock_run:
        calls = 0
        original_sleep = time.sleep
        main_thread = threading.current_thread()

        def short_sleep(s):
            nonlocal calls
            if threading.current_thread() is not main_thread:
                original_sleep(s)
                return
            calls += 1
            if calls > 6:
                raise SystemExit
            original_sleep(0)

        with patch("time.sleep", short_sleep):
            try:
                watchdog.main([])
            except SystemExit:
                pass
    assert rollback.called
    assert not mock_run.called
