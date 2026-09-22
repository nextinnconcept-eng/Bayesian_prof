"""SQLAlchemy engine/session factory, reading the DB URL from Config."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from dssc_discovery.config import get_config, resolve_path

_SQLITE_PREFIX = "sqlite:///"


def resolve_database_url(database_url: str) -> str:
    """Resolve a relative sqlite:/// path against the repo root, so the
    engine works regardless of the process's current working directory.
    In-memory URLs, already-absolute paths, and non-sqlite URLs pass
    through unchanged.
    """
    if not database_url.startswith(_SQLITE_PREFIX):
        return database_url

    raw_path = database_url[len(_SQLITE_PREFIX) :]
    if raw_path == ":memory:" or Path(raw_path).is_absolute():
        return database_url

    absolute_path = resolve_path(Path(raw_path))
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    return f"{_SQLITE_PREFIX}{absolute_path.as_posix()}"


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _enable_sqlite_foreign_keys(dbapi_connection: object, connection_record: object) -> None:
    # SQLite has foreign key enforcement OFF by default per connection --
    # verified empirically that an invalid FK silently succeeds without
    # this. Postgres enforces FKs natively and doesn't have this pragma.
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        config = get_config()
        url = resolve_database_url(config.database_url)
        _engine = create_engine(url)
        if _engine.dialect.name == "sqlite":
            event.listen(_engine, "connect", _enable_sqlite_foreign_keys)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine())
    return _session_factory


def reset_engine_cache() -> None:
    """Test-only: dispose and drop the cached engine/session factory, so a
    changed DSSC_DATABASE_URL (or config cache clear) takes effect.
    """
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
