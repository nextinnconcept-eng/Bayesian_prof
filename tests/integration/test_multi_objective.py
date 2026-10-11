"""M9 (Chunks 9.1-9.5): partial-stability dataset with explicit
missingness, hand-checked Pareto classification, multi-objective
acquisition, and routing through the same approval path."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from dssc_discovery.closed_loop.orchestrator import recommend_multi_objective_cycle
from dssc_discovery.database.repositories import (
    DeviceRepository,
    PhotovoltaicMeasurementRepository,
    StabilityMeasurementRepository,
)
from dssc_discovery.domain.enums import ClosedLoopState, IlluminationProtocol, SpeciesCode
from dssc_discovery.domain.measurements import PhotovoltaicMeasurement, StabilityMeasurement
from dssc_discovery.features.synthetic import generate_synthetic_dataset
from dssc_discovery.features.versioning import create_dataset_version
from dssc_discovery.models.tracking import log_surrogate_fit
from dssc_discovery.optimization.multi_objective import (
    fit_multi_objective,
    optimize_multi_objective,
)
from dssc_discovery.optimization.search_space import search_space_for
from dssc_discovery.stability.retention import build_multi_objective_dataset, stability_coverage
from dssc_discovery.visualization.pareto_plots import dominance_classification, plot_pareto_front
from tests.bo_helpers import FAST

SPACE = search_space_for(SpeciesCode.BETA_VULGARIS)


def _age_devices(session, devices, rng):
    pv_repo = PhotovoltaicMeasurementRepository(session)
    for device in devices:
        baseline = pv_repo.filter(device_id=device.id)[0]
        retention = float(rng.uniform(60, 95))
        aged_pce = baseline.power_conversion_efficiency_percent * retention / 100
        aged = pv_repo.add(
            PhotovoltaicMeasurement(
                device_id=device.id,
                illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
                measurement_date=date(2026, 12, 1),
                measurement_replicate_number=2,
                open_circuit_voltage_v=baseline.open_circuit_voltage_v,
                short_circuit_current_density_ma_cm2=baseline.short_circuit_current_density_ma_cm2
                * retention
                / 100,
                fill_factor=baseline.fill_factor,
                power_conversion_efficiency_percent=aged_pce,
                operator="tester",
                instrument="SYNTHETIC",
                is_synthetic=True,
            )
        )
        StabilityMeasurementRepository(session).add(
            StabilityMeasurement(
                device_id=device.id,
                baseline_measurement_id=baseline.id,
                timepoint_measurement_id=aged.id,
                ageing_duration_h=500.0,
                retention_percent=retention,
                operator="tester",
                instrument="SYNTHETIC",
                is_synthetic=True,
            )
        )


@pytest.fixture
def partial_stability(session):
    devices = generate_synthetic_dataset(session, n=20, seed=8)
    _age_devices(session, devices[:12], np.random.default_rng(0))
    session.commit()
    return build_multi_objective_dataset(session)


def test_partial_stability_is_explicit_not_imputed(partial_stability):
    data = partial_stability
    assert len(data) == 20
    assert stability_coverage(data) == {"measured": 12, "not_yet_aged": 8}
    pending = data[data["stability_status"] == "not_yet_aged"]
    assert pending["retention_percent"].isna().all()
    assert data.loc[data["stability_status"] == "measured", "retention_percent"].notna().all()


def test_pareto_classification_on_hand_checked_points():
    points = pd.DataFrame(
        {"pce": [5.0, 6.0, 4.0, 6.0, 3.0, np.nan], "ret": [90.0, 80.0, 85.0, 70.0, 95.0, 99.0]}
    )
    flags = dominance_classification(points, ("pce", "ret"))
    # (5,90),(6,80),(3,95) non-dominated; (4,85) dominated by (5,90); (6,70) by (6,80)
    assert list(flags.iloc[:5]) == [True, True, False, False, True]
    assert pd.isna(flags.iloc[5])
    fig = plot_pareto_front(points, objectives=("pce", "ret"))
    assert fig.axes[0].get_title()


def test_multi_objective_recommendations_route_through_approval(
    session, partial_stability, tmp_path
):
    surrogate = fit_multi_objective(partial_stability, SPACE, seed=0)
    assert len(surrogate.retention.train_Y) == 12  # only aged devices
    assert len(surrogate.pce.train_Y) == 20

    result = optimize_multi_objective(surrogate, batch_size=2, seed=0, **FAST)
    assert len(result.candidates) == len(result.retention) == 2
    for candidate in result.candidates:
        assert SPACE.violations(candidate.vector.model_dump(mode="json")) == []

    version = create_dataset_version(session, operator="tester")
    log_surrogate_fit(
        session, surrogate.pce, version, "tester", tracking_uri=str(tmp_path / "mlruns")
    )
    batch = recommend_multi_objective_cycle(
        session,
        surrogate,
        cycle_number=1,
        batch_size=2,
        seed=0,
        operator="tester",
        is_synthetic=True,
        acquisition_options=FAST,
    )
    session.commit()
    assert batch.bo_iteration.objective == "pce_retention"
    assert all(r.predicted_retention is not None for r in batch.recommendations)
    assert all(c.state == ClosedLoopState.PROPOSED for c in batch.conditions)
    assert "retention" in batch.recommendations[0].rationale_text


def test_retention_fit_refuses_too_few_aged_devices(session):
    devices = generate_synthetic_dataset(session, n=10, seed=9)
    _age_devices(session, devices[:2], np.random.default_rng(1))
    session.commit()
    with pytest.raises(ValueError, match="need >= 5"):
        fit_multi_objective(build_multi_objective_dataset(session), SPACE)
    assert DeviceRepository(session).list()
