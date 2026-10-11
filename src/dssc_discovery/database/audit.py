"""Audit logging and append-only enforcement.

Two independent mechanisms, both wired as SQLAlchemy events registered
once at import time (register_audit_hooks(), called at the bottom of this
module) -- so protection doesn't depend on the caller remembering to
enable it, regardless of whether the engine/session was built via
dssc_discovery.database.session.get_engine() or constructed directly (as
every test fixture in this project does).

1. Every insert/update/delete on any mapped table is logged to
   audit_log (Session.before_flush event) -- who/what/when. The *why*
   lives in the domain model's own provenance fields (operator, notes);
   this table is about detecting and reconstructing changes, not
   explaining intent.
2. Raw/measured-data tables (RawSpectrum, DerivedDescriptor,
   ElectrochemicalMeasurement, PhotovoltaicMeasurement,
   StabilityMeasurement) reject UPDATE outright (a per-mapper
   before_update event) -- PLAN.txt Section 12: raw data is never
   overwritten. A correction is a new record with its own provenance,
   not a silent edit to history.
"""

from __future__ import annotations

import json
import uuid as uuid_module
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.orm import Mapped, Session, mapped_column

from dssc_discovery.database.models import (
    GUID,
    Base,
    DerivedDescriptorORM,
    ElectrochemicalMeasurementORM,
    PhotovoltaicMeasurementORM,
    RawSpectrumORM,
    StabilityMeasurementORM,
    UTCDateTime,
)
from dssc_discovery.logging_setup import get_logger

logger = get_logger(__name__)


class AppendOnlyViolationError(Exception):
    """Raised when code attempts to UPDATE a row in an append-only table."""


class AuditLogORM(Base):
    __tablename__ = "audit_log"

    id: Mapped[uuid_module.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid_module.uuid4
    )
    table_name: Mapped[str] = mapped_column(sa.String(100))
    record_id: Mapped[str] = mapped_column(sa.String(36))
    operation: Mapped[str] = mapped_column(sa.String(20))  # "insert" | "update" | "delete"
    changed_fields: Mapped[str] = mapped_column(sa.Text, default="")  # JSON {field: [old, new]}
    changed_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=lambda: datetime.now(UTC))


_APPEND_ONLY_ORM_CLASSES = (
    RawSpectrumORM,
    DerivedDescriptorORM,
    ElectrochemicalMeasurementORM,
    PhotovoltaicMeasurementORM,
    StabilityMeasurementORM,
)

_hooks_registered = False


def _reject_update(mapper: object, connection: object, target: object) -> None:
    raise AppendOnlyViolationError(
        f"{type(target).__name__} rows are append-only (raw/measured data, "
        f"PLAN.txt Section 12): record id {getattr(target, 'id', '?')} cannot be "
        f"updated. Insert a new corrected record instead of editing this one."
    )


def _summarize_changes(obj: object) -> str:
    changes: dict[str, list[object]] = {}
    for attr in sa.inspect(obj).attrs:  # type: ignore[union-attr]
        history = attr.history
        if not history.has_changes():
            continue
        old = history.deleted[0] if history.deleted else None
        new = history.added[0] if history.added else None
        changes[attr.key] = [old, new]
    return json.dumps(changes, default=str)


def _add_audit_entry(session: Session, obj: object, operation: str) -> None:
    table = getattr(type(obj), "__tablename__", type(obj).__name__)
    record_id = str(getattr(obj, "id", ""))
    changed_fields = _summarize_changes(obj) if operation == "update" else ""
    session.add(
        AuditLogORM(
            table_name=table,
            record_id=record_id,
            operation=operation,
            changed_fields=changed_fields,
        )
    )


def _is_audited(obj: object) -> bool:
    return isinstance(obj, Base) and not isinstance(obj, AuditLogORM)


def _before_flush(session: Session, flush_context: object, instances: object) -> None:
    # Snapshot first: session.new/dirty/deleted must not be mutated while
    # iterating, and adding audit rows below must not re-audit themselves.
    # The listener is registered on Session globally, so it also sees
    # sessions of other libraries in the same process (e.g. MLflow's own
    # SQLite tracking store, Chunk 4.7) -- only this schema's objects are
    # audited; anything else is none of our business and its database has
    # no audit_log table to write to.
    new_objects = [o for o in session.new if _is_audited(o)]
    dirty_objects = [
        o
        for o in session.dirty
        if _is_audited(o) and session.is_modified(o, include_collections=False)
    ]
    deleted_objects = [o for o in session.deleted if _is_audited(o)]

    for obj in new_objects:
        _add_audit_entry(session, obj, "insert")
    for obj in dirty_objects:
        _add_audit_entry(session, obj, "update")
    for obj in deleted_objects:
        _add_audit_entry(session, obj, "delete")


def register_audit_hooks() -> None:
    """Idempotent: safe to call more than once (only the first call
    actually registers anything), so re-importing this module or calling
    it defensively from multiple entry points never double-registers.
    """
    global _hooks_registered
    if _hooks_registered:
        return

    event.listen(Session, "before_flush", _before_flush)
    for orm_cls in _APPEND_ONLY_ORM_CLASSES:
        event.listen(orm_cls, "before_update", _reject_update)

    _hooks_registered = True
    logger.debug("audit + append-only hooks registered")


register_audit_hooks()
