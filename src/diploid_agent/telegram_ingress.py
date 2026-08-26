"""Re-export HTTP transport under the legacy module name."""

from diploid_agent.transport.http import HttpTransport, create_app, main

__all__ = ["HttpTransport", "create_app", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
