"""Raw instrument file storage: a managed, checksummed file-reference
pattern for RawSpectrum.raw_data_reference (PLAN.txt Section 12: raw data
is never overwritten, and derived descriptors must stay traceable to it).

register_raw_file() copies a raw instrument file into managed storage
under Config.data_dir/raw/<modality>/, computes its SHA-256, and returns
a RawDataReference (path + checksum) whose to_reference_string() is
exactly what RawSpectrum.raw_data_reference stores -- no DB schema change
needed, since the field is already a single string (PLAN.txt's own
roadmap calls for "path/hash stored in RawSpectrum.raw_data_reference").

resolve_raw_file() does the reverse: given a stored reference, returns
the file's bytes after verifying the checksum still matches what was
recorded at registration -- raising rather than silently returning
altered or corrupted data.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from dssc_discovery.config import get_config, resolve_path
from dssc_discovery.domain.enums import MeasurementType
from dssc_discovery.logging_setup import get_logger

logger = get_logger(__name__)


class RawFileStorageError(Exception):
    """Raised when a raw file can't be registered, is missing on
    resolve, fails checksum verification (altered/corrupted since
    registration), or its reference string is malformed.
    """


@dataclass(frozen=True)
class RawDataReference:
    path: str  # relative to Config.data_dir, forward-slash form
    sha256: str

    def to_reference_string(self) -> str:
        return json.dumps({"path": self.path, "sha256": self.sha256})

    @classmethod
    def from_reference_string(cls, reference: str) -> RawDataReference:
        try:
            data = json.loads(reference)
            return cls(path=data["path"], sha256=data["sha256"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RawFileStorageError(f"Malformed raw_data_reference: {reference!r}") from exc


def _data_dir() -> Path:
    return resolve_path(get_config().data_dir)


def _raw_storage_root() -> Path:
    return _data_dir() / "raw"


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_raw_file(
    source_path: Path, measurement_type: MeasurementType, dest_filename: str | None = None
) -> RawDataReference:
    """Copy source_path into managed storage under
    data/raw/<measurement_type>/, compute its checksum, and return a
    RawDataReference. Copies (never moves) -- registration must never
    risk the researcher's original file. Refuses to overwrite an
    already-registered destination: raw data is immutable, so
    re-registering under the same filename is treated as an error, not
    a silent replace.
    """
    if not source_path.exists():
        raise RawFileStorageError(f"Source file does not exist: {source_path}")

    dest_dir = _raw_storage_root() / measurement_type.value
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / (dest_filename or source_path.name)

    if dest_path.exists():
        raise RawFileStorageError(
            f"Refusing to overwrite already-registered raw file: {dest_path} "
            f"(raw data is immutable -- register under a different filename if "
            f"this is genuinely a separate acquisition)"
        )

    shutil.copy2(source_path, dest_path)
    checksum = _sha256_of_file(dest_path)

    relative_path = dest_path.relative_to(_data_dir()).as_posix()
    reference = RawDataReference(path=relative_path, sha256=checksum)
    logger.info("registered raw file: %s (sha256=%s...)", relative_path, checksum[:12])
    return reference


def resolve_raw_file(reference_string: str) -> bytes:
    """Return the raw file's bytes after verifying its checksum still
    matches what was recorded at registration. Raises
    RawFileStorageError if the file is missing or the checksum no
    longer matches.
    """
    reference = RawDataReference.from_reference_string(reference_string)
    full_path = _data_dir() / reference.path

    if not full_path.exists():
        raise RawFileStorageError(f"Raw file missing: {full_path}")

    actual_checksum = _sha256_of_file(full_path)
    if actual_checksum != reference.sha256:
        raise RawFileStorageError(
            f"Checksum mismatch for {full_path}: expected {reference.sha256}, got "
            f"{actual_checksum} -- file has been altered or corrupted since it was "
            f"registered"
        )

    return full_path.read_bytes()
