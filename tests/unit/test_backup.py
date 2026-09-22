"""Tests for database backup (Chunk 1.8). Only the SQLite path is
exercised -- there is no Postgres server in this dev environment, so
backup_postgres() is implemented but deliberately left unverified (its
own docstring says so).
"""

from __future__ import annotations

import sqlite3

import pytest

from dssc_discovery.database.backup import BackupError, backup_sqlite


def _make_sqlite_db(path, value: str = "hello") -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
    conn.execute("INSERT INTO t (val) VALUES (?)", (value,))
    conn.commit()
    conn.close()


def test_backup_sqlite_creates_a_file(tmp_path):
    db_path = tmp_path / "source.db"
    _make_sqlite_db(db_path)

    backup_path = backup_sqlite(db_path, backup_dir=tmp_path / "backups")

    assert backup_path.exists()
    assert backup_path.parent == tmp_path / "backups"


def test_backup_sqlite_is_a_real_independent_copy(tmp_path):
    db_path = tmp_path / "source.db"
    _make_sqlite_db(db_path, value="original")

    backup_path = backup_sqlite(db_path, backup_dir=tmp_path / "backups")

    # mutate the source AFTER backup -- the backup must not reflect it
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE t SET val = 'mutated' WHERE id = 1")
    conn.commit()
    conn.close()

    backup_conn = sqlite3.connect(str(backup_path))
    rows = backup_conn.execute("SELECT val FROM t").fetchall()
    backup_conn.close()
    assert rows == [("original",)]


def test_backup_sqlite_raises_on_missing_source(tmp_path):
    with pytest.raises(BackupError, match="does not exist"):
        backup_sqlite(tmp_path / "does_not_exist.db", backup_dir=tmp_path / "backups")


def test_backup_sqlite_creates_backup_dir_if_missing(tmp_path):
    db_path = tmp_path / "source.db"
    _make_sqlite_db(db_path)
    backup_dir = tmp_path / "nested" / "backups"
    assert not backup_dir.exists()

    backup_sqlite(db_path, backup_dir=backup_dir)

    assert backup_dir.exists()


def test_repeated_backups_get_distinct_filenames(tmp_path):
    import time

    db_path = tmp_path / "source.db"
    _make_sqlite_db(db_path)
    backup_dir = tmp_path / "backups"

    first = backup_sqlite(db_path, backup_dir=backup_dir)
    time.sleep(1.1)  # timestamp resolution is 1 second
    second = backup_sqlite(db_path, backup_dir=backup_dir)

    assert first != second
    assert first.exists() and second.exists()
