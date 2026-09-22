"""Central, environment-overridable configuration.

Precedence (highest first): explicit constructor kwargs, environment
variables (prefixed ``DSSC_``), a local ``.env`` file, ``config/default.yaml``,
then the field defaults below.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DSSC_",
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        yaml_file=DEFAULT_CONFIG_PATH,
        extra="forbid",
    )

    database_url: str = Field(default="sqlite:///data/dssc_discovery_dev.db")
    mlflow_tracking_uri: str = Field(default="./mlruns")
    random_seed: int = Field(default=42)
    data_dir: Path = Field(default=Path("data"))
    log_level: str = Field(default="INFO")
    log_dir: Path = Field(default=Path("logs"))

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_settings = YamlConfigSettingsSource(settings_cls)
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_settings,
            file_secret_settings,
        )


@lru_cache
def get_config() -> Config:
    return Config()


def clear_config_cache() -> None:
    """Test-only: drop the cached Config so env/.env changes are re-read."""
    get_config.cache_clear()


def resolve_path(path: Path) -> Path:
    """Resolve a config path against the repository root if it isn't already absolute."""
    return path if path.is_absolute() else REPO_ROOT / path
