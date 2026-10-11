"""One-click SYNTHETIC demo dataset for presenting the dashboard: a
3-cycle simulated closed loop (closed_loop/orchestrator.py), stability
measurements for about half the devices, three recommendations waiting
for approval, and one logged hypothesis. Every row is is_synthetic=True.

Refuses to run on a database that already holds any experimental
condition -- it must never mix fabricated rows into real data.
"""

from __future__ import annotations

from datetime import date

import numpy as np
from sqlalchemy.orm import Session

from dssc_discovery.closed_loop.orchestrator import (
    SIMULATION_OPERATOR,
    recommend_cycle,
    retrain,
    run_simulated_closed_loop,
)
from dssc_discovery.database.repositories import (
    DeviceRepository,
    ExperimentalConditionRepository,
    PhotovoltaicMeasurementRepository,
    StabilityMeasurementRepository,
)
from dssc_discovery.domain.enums import IlluminationProtocol, SpeciesCode
from dssc_discovery.domain.measurements import PhotovoltaicMeasurement, StabilityMeasurement
from dssc_discovery.evaluation.cv_harness import default_feature_specs, model_features
from dssc_discovery.explainability.hypothesis_log import log_hypothesis
from dssc_discovery.explainability.shap_tree import explain_tree_model
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.models.tree_models import RandomForestModel
from dssc_discovery.optimization.search_space import search_space_for

FAST_ACQUISITION = {"num_restarts": 2, "raw_samples": 32, "fallback_pool_size": 200}
SPECIES = SpeciesCode.BETA_VULGARIS


def build_demo_database(session: Session, *, seed: int = 0, tracking_uri: str | None = None) -> int:
    """Populate an EMPTY database with synthetic demo data; returns the
    number of measured devices. Takes roughly one to two minutes."""
    if ExperimentalConditionRepository(session).list():
        raise ValueError(
            "the database already has experiments; demo data only goes into an empty one"
        )

    run_simulated_closed_loop(
        session,
        SPECIES,
        n_cycles=3,
        initial_doe_size=20,
        batch_size=4,
        seed=seed,
        tracking_uri=tracking_uri,
        acquisition_options=FAST_ACQUISITION,
    )
    surrogate = retrain(
        session,
        search_space_for(SPECIES),
        seed=seed + 9,
        operator=SIMULATION_OPERATOR,
        tracking_uri=tracking_uri,
    )
    recommend_cycle(
        session,
        surrogate,
        cycle_number=4,
        batch_size=3,
        seed=seed + 9,
        operator=SIMULATION_OPERATOR,
        is_synthetic=True,
        acquisition_options=FAST_ACQUISITION,
    )
    session.commit()

    rng = np.random.default_rng(seed)
    pv_repo = PhotovoltaicMeasurementRepository(session)
    for device in DeviceRepository(session).list()[::2]:
        baseline = pv_repo.filter(device_id=device.id)[0]
        retention = float(rng.uniform(55, 95))
        aged = pv_repo.add(
            PhotovoltaicMeasurement(
                device_id=device.id,
                illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
                measurement_date=date.today(),
                measurement_replicate_number=2,
                open_circuit_voltage_v=baseline.open_circuit_voltage_v,
                short_circuit_current_density_ma_cm2=baseline.short_circuit_current_density_ma_cm2
                * retention
                / 100,
                fill_factor=baseline.fill_factor,
                power_conversion_efficiency_percent=baseline.power_conversion_efficiency_percent
                * retention
                / 100,
                operator=SIMULATION_OPERATOR,
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
                operator=SIMULATION_OPERATOR,
                instrument="SYNTHETIC",
                is_synthetic=True,
            )
        )

    data = build_ml_dataset(session)
    X = model_features(data)
    model = RandomForestModel(
        feature_specs=default_feature_specs(
            ["ph", "pigment_concentration_mm", "adsorption_time_min", "tio2_thickness_um"]
        ),
        n_estimators=50,
        random_state=seed,
    ).fit(X, data["power_conversion_efficiency_percent"])
    log_hypothesis(
        session,
        observation="Mid-range pH is associated with higher predicted PCE",
        candidate_mechanism="pH-dependent pigment speciation changes how the dye anchors to TiO2",
        confound_check="pH co-varies with solvent choice in the initial design",
        proposed_confirmatory_experiment="pH series at fixed solvent and pigment concentration",
        operator=SIMULATION_OPERATOR,
        explanation=explain_tree_model(model, X),
        is_synthetic=True,
    )
    session.commit()
    return len(data)
