"""Save and list DSSC Simulator runs (one table, no repository layer needed)."""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from dssc_discovery.database.models import DSSCSimulationRunORM
from dssc_discovery.dssc_model import DSSCParams, Performance, SingleDiodeParams


def save_run(
    session: Session,
    *,
    operator: str,
    label: str,
    model: str,
    params: DSSCParams,
    irradiance: float,
    area_cm2: float,
    perf: Performance,
    single_diode: SingleDiodeParams | None = None,
    uvvis: list[tuple[float, float]] | None = None,
    uvvis_wavelength_nm: float | None = None,
) -> DSSCSimulationRunORM:
    """Store one run: its inputs and the calculated figures of merit."""
    parameters = {"model_parameters": asdict(params)}
    if single_diode is not None:
        parameters["single_diode"] = asdict(single_diode)
    row = DSSCSimulationRunORM(
        operator=operator or "owner",
        is_synthetic=True,  # calculated, not measured
        label=label.strip(),
        model=model,
        temperature_k=params.T,
        irradiance_w_m2=irradiance,
        area_cm2=area_cm2,
        alpha_cm1=params.alpha,
        uvvis_wavelength_nm=uvvis_wavelength_nm,
        uvvis_json=json.dumps([list(pair) for pair in (uvvis or [])]),
        parameters_json=json.dumps(parameters),
        voc_v=perf.voc,
        jsc_ma_cm2=perf.jsc,
        isc_ma=perf.jsc * area_cm2,
        fill_factor=perf.ff,
        vmp_v=perf.vmp,
        imp_ma_cm2=perf.imp,
        pmax_mw_cm2=perf.pmax,
        efficiency_percent=perf.efficiency,
    )
    session.add(row)
    session.commit()
    return row


def runs_frame(session: Session) -> pd.DataFrame:
    """All saved runs, newest first, with readable column names."""
    rows = session.scalars(
        select(DSSCSimulationRunORM).order_by(DSSCSimulationRunORM.created_at.desc())
    ).all()
    return pd.DataFrame(
        [
            {
                "Saved": r.created_at.strftime("%Y-%m-%d %H:%M"),
                "Label": r.label,
                "Model": r.model,
                "T (K)": r.temperature_k,
                "Irradiance (W/m²)": r.irradiance_w_m2,
                "α (1/cm)": r.alpha_cm1,
                "Voc (V)": r.voc_v,
                "Jsc (mA/cm²)": r.jsc_ma_cm2,
                "Isc (mA)": r.isc_ma,
                "FF": r.fill_factor,
                "Pmax (mW/cm²)": r.pmax_mw_cm2,
                "η (%)": r.efficiency_percent,
            }
            for r in rows
        ]
    )
