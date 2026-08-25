"""Re-export of the Telegram long-polling transport."""

from acp_fleet_harness.transport.telegram import (
    ChatInput,
    TelegramPoller,
    TelegramTransport,
    TurnWorker,
    main,
)

__all__ = [
    "ChatInput",
    "TelegramPoller",
    "TelegramTransport",
    "TurnWorker",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
