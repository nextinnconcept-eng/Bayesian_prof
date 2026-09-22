"""CSV adapter over manual_entry's row-ingestion engine. Reads a CSV file
in the wide format (one row = one condition + photoanode batch + device +
PV measurement, see manual_entry.py) and delegates to ingest_rows() for
parsing, validation, and atomic persistence.
"""

from __future__ import annotations

import csv
from pathlib import Path

from sqlalchemy.orm import Session

from dssc_discovery.ingestion.manual_entry import IngestionReport, ingest_rows


def load_condition_device_pv_csv(path: Path, session: Session) -> IngestionReport:
    with open(path, newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return ingest_rows(rows, session)
