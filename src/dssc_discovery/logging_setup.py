"""Structured logging to console + file, configured once from Config.

Failures must be explicit and traceable, not silent — every module should
log through get_logger(__name__) rather than printing or staying quiet.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from dssc_discovery.config import get_config, resolve_path

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_ROOT_LOGGER_NAME = "dssc_discovery"
_configured = False


def _configure_root_logger() -> None:
    global _configured
    if _configured:
        return

    config = get_config()
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.handlers.clear()
    root.setLevel(config.log_level)
    root.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    log_dir = resolve_path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "dssc_discovery.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under 'dssc_discovery', configuring handlers on first use."""
    _configure_root_logger()
    if name == _ROOT_LOGGER_NAME or name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def reset_logging() -> None:
    """Test-only: drop handlers and force reconfiguration on the next get_logger() call."""
    global _configured
    logging.getLogger(_ROOT_LOGGER_NAME).handlers.clear()
    _configured = False
