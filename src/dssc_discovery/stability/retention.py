"""Stability (retention %) as a second objective (Chunk 9.1).

The ageing-time lag (Section Q risk) is the central issue: a condition
can have PCE data months before its devices finish ageing. So retention
is NEVER imputed and rows are NEVER dropped for lacking it -- every row
carries an explicit stability_status:
- "measured": a StabilityMeasurement within tolerance of the target
  ageing duration exists (the closest one is used);
- "not_yet_aged": no such measurement yet; retention_percent is NaN.
Downstream code (multi_objective.py) uses only "measured" rows for the
retention surrogate and all rows for PCE.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from dssc_discovery.database.repositories import StabilityMeasurementRepository
from dssc_discovery.features.dataset_builder import build_ml_dataset

RETENTION_COLUMN = "retention_percent"
STATUS_COLUMN = "stability_status"
AGEING_COLUMN = "stability_ageing_h"
DEFAULT_TARGET_AGEING_H = 500.0


def retention_for_device(
    session: Session, device_id: UUID, target_ageing_h: float, tolerance_h: float
) -> tuple[float, float] | None:
    """(retention %, ageing h) of the measurement closest to the target
    ageing duration within tolerance, or None."""
    rows = StabilityMeasurementRepository(session).filter(device_id=device_id)
    candidates = [r for r in rows if abs(r.ageing_duration_h - target_ageing_h) <= tolerance_h]
    if not candidates:
        return None
    best = min(candidates, key=lambda r: (abs(r.ageing_duration_h - target_ageing_h), r.created_at))
    return best.retention_percent, best.ageing_duration_h


def build_multi_objective_dataset(
    session: Session,
    target_ageing_h: float = DEFAULT_TARGET_AGEING_H,
    tolerance_h: float = 50.0,
) -> pd.DataFrame:
    """build_ml_dataset() plus retention_percent / stability_status /
    stability_ageing_h. Same rows, same order."""
    data = build_ml_dataset(session)
    retention, ageing, status = [], [], []
    for device_id in data["device_id"]:
        found = retention_for_device(session, UUID(str(device_id)), target_ageing_h, tolerance_h)
        if found is None:
            retention.append(np.nan)
            ageing.append(np.nan)
            status.append("not_yet_aged")
        else:
            retention.append(found[0])
            ageing.append(found[1])
            status.append("measured")
    return data.assign(
        **{RETENTION_COLUMN: retention, AGEING_COLUMN: ageing, STATUS_COLUMN: status}
    )


def stability_coverage(data: pd.DataFrame) -> dict[str, int]:
    counts = data[STATUS_COLUMN].value_counts().to_dict()
    return {
        "measured": int(counts.get("measured", 0)),
        "not_yet_aged": int(counts.get("not_yet_aged", 0)),
    }
