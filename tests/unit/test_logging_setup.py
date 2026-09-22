import logging

import pytest

from dssc_discovery.config import clear_config_cache
from dssc_discovery.logging_setup import get_logger, reset_logging


@pytest.fixture(autouse=True)
def _clean_state():
    reset_logging()
    clear_config_cache()
    yield
    reset_logging()
    clear_config_cache()


def _flush_handlers():
    for handler in logging.getLogger("dssc_discovery").handlers:
        handler.flush()


def test_get_logger_writes_to_configured_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DSSC_LOG_DIR", str(tmp_path))
    clear_config_cache()

    logger = get_logger("test_module")
    logger.info("hello from test")
    _flush_handlers()

    log_file = tmp_path / "dssc_discovery.log"
    assert log_file.exists()
    assert "hello from test" in log_file.read_text(encoding="utf-8")


def test_logger_name_is_namespaced(tmp_path, monkeypatch):
    monkeypatch.setenv("DSSC_LOG_DIR", str(tmp_path))
    clear_config_cache()

    logger = get_logger("chemistry.uvvis")
    assert logger.name == "dssc_discovery.chemistry.uvvis"


def test_level_filtering(tmp_path, monkeypatch):
    monkeypatch.setenv("DSSC_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("DSSC_LOG_LEVEL", "WARNING")
    clear_config_cache()

    logger = get_logger("test_module")
    logger.info("this should be filtered out")
    logger.warning("this should appear")
    _flush_handlers()

    content = (tmp_path / "dssc_discovery.log").read_text(encoding="utf-8")
    assert "this should be filtered out" not in content
    assert "this should appear" in content
