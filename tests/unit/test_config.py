import pytest
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from dssc_discovery.config import Config, clear_config_cache, get_config


@pytest.fixture(autouse=True)
def _clean_config_cache():
    clear_config_cache()
    yield
    clear_config_cache()


def test_default_config_loads_from_yaml():
    config = get_config()
    assert config.random_seed == 42
    assert config.database_url == "sqlite:///data/dssc_discovery_dev.db"
    assert config.mlflow_tracking_uri == "./mlruns"
    assert config.log_level == "INFO"


def test_get_config_is_cached():
    assert get_config() is get_config()


def test_env_var_overrides_yaml_default(monkeypatch):
    monkeypatch.setenv("DSSC_RANDOM_SEED", "123")
    monkeypatch.setenv("DSSC_LOG_LEVEL", "DEBUG")
    clear_config_cache()

    config = get_config()

    assert config.random_seed == 123
    assert config.log_level == "DEBUG"
    # untouched fields still come from the YAML default
    assert config.database_url == "sqlite:///data/dssc_discovery_dev.db"


def test_missing_required_setting_raises_clearly(monkeypatch):
    # Exercises the same underlying mechanism Config relies on: a field with
    # no default, no matching env var, and no entry in any config source
    # must fail loudly rather than silently defaulting to None/empty.
    class RequiresSecret(BaseSettings):
        model_config = SettingsConfigDict(env_prefix="DSSC_TEST_")
        api_key: str  # no default -> required

    monkeypatch.delenv("DSSC_TEST_API_KEY", raising=False)

    with pytest.raises(ValidationError, match="api_key"):
        RequiresSecret()
