"""Database backup: a point-in-time copy taken before any operation that
could lose data (a destructive migration, a bulk delete, manual DB
surgery) -- PLAN.txt Section 17.

SQLite: an online backup via sqlite3's own backup API, which is safe even
with concurrent writers (unlike a raw file copy, which could capture a
half-written page mid-write).

PostgreSQL: a pg_dump hook, since the mature system targets Postgres
(PLAN.txt Section R). Implemented against pg_dump's documented CLI
interface but UNTESTED in this dev environment -- there is no Postgres
server available here to verify against. Flagged explicitly rather than
claimed as verified.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from dssc_discovery.config import get_config, resolve_path
from dssc_discovery.database.session import get_engine
from dssc_discovery.logging_setup import get_logger

logger = get_logger(__name__)

_SQLITE_URL_PREFIX = "sqlite:///"


class BackupError(Exception):
    """A backup could not be created. Callers must treat this as fatal
    for whatever destructive operation was about to run -- never proceed
    without a successful backup.
    """


def _default_backup_dir() -> Path:
    return resolve_path(Path("data")) / "backups"


def backup_sqlite(db_path: Path, backup_dir: Path | None = None) -> Path:
    """Online backup of a SQLite database file via sqlite3's backup API."""
    if not db_path.exists():
        raise BackupError(f"Cannot back up {db_path}: file does not exist")

    backup_dir = backup_dir or _default_backup_dir()
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"{db_path.stem}_{timestamp}.db"

    source = sqlite3.connect(str(db_path))
    try:
        dest = sqlite3.connect(str(backup_path))
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()

    logger.info("SQLite backup created: %s", backup_path)
    return backup_path


def backup_postgres(database_url: str, backup_dir: Path | None = None) -> Path:
    """pg_dump-based backup. UNTESTED: no Postgres server is available in
    this dev environment to verify against -- implemented against
    pg_dump's documented CLI interface for when the mature system targets
    Postgres, but treat this path as unverified until it's exercised
    against a real Postgres instance.
    """
    if shutil.which("pg_dump") is None:
        raise BackupError("pg_dump not found on PATH; cannot back up PostgreSQL database")

    backup_dir = backup_dir or _default_backup_dir()
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"postgres_backup_{timestamp}.sql"

    try:
        with open(backup_path, "wb") as backup_file:
            subprocess.run(
                ["pg_dump", "--dbname", database_url],
                stdout=backup_file,
                check=True,
            )
    except subprocess.CalledProcessError as exc:
        raise BackupError(f"pg_dump failed: {exc}") from exc

    logger.info("PostgreSQL backup created: %s", backup_path)
    return backup_path


def backup_database(backup_dir: Path | None = None) -> Path:
    """Back up the currently configured database (Config.database_url via
    get_engine()), dispatching by engine dialect. Raises BackupError on
    failure or on an unsupported dialect -- callers must not proceed with
    a destructive operation if this raises.
    """
    engine = get_engine()

    if engine.dialect.name == "sqlite":
        db_url = get_config().database_url
        if not db_url.startswith(_SQLITE_URL_PREFIX):
            raise BackupError(f"Unsupported sqlite URL form for backup: {db_url}")
        raw_path = db_url[len(_SQLITE_URL_PREFIX) :]
        if raw_path == ":memory:":
            raise BackupError("Cannot back up an in-memory SQLite database")
        return backup_sqlite(resolve_path(Path(raw_path)), backup_dir)

    if engine.dialect.name == "postgresql":
        return backup_postgres(str(engine.url), backup_dir)

    raise BackupError(f"No backup implementation for dialect {engine.dialect.name!r}")
