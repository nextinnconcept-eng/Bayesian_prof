"""Central, environment-overridable configuration.

Precedence (highest first): explicit constructor kwargs, environment
variables (prefixed ``DSSC_``), a local ``.env`` file, ``config/default.yaml``,
then the field defaults below.
"""

from __future__ import annotations

import subprocess
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
    # Chunk 8.3: real retraining refuses to run below this many INGESTED
    # real conditions per species (default = the initial DoE size).
    min_real_conditions_for_retraining: int = Field(default=40, ge=1)
    bo_batch_size: int = Field(default=12, ge=1)

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


def _run_git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_git_commit_hash() -> str:
    """The current commit's full SHA, or "unknown" if git isn't available
    or this isn't a git checkout (e.g. a packaged deployment) -- used as
    the "code version" component of reproducibility records (PLAN.txt
    Section M). Never raises: a missing git binary shouldn't break
    dataset/model versioning, just make that one field less informative.
    """
    return _run_git("rev-parse", "HEAD") or "unknown"


def is_git_working_tree_dirty() -> bool:
    """True if there are uncommitted changes at call time. A commit hash
    alone doesn't fully capture "what code produced this" if the working
    tree had uncommitted edits -- this flag makes that gap visible
    instead of silently implying a clean, fully-reproducible state.
    Defaults to True (assume dirty/unknown) if git status can't be
    determined at all, since that's the safer assumption for a
    reproducibility flag.
    """
    status = _run_git("status", "--porcelain")
    return True if status is None else bool(status)
