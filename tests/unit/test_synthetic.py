"""Tests for the synthetic ground-truth PCE function and its Voc/FF/Jsc
decomposition (Chunk 3.5) -- pure functions, no database needed.

The core property under test: the deterministic ground truth has a
single, well-defined optimum inside this module's own sampling ranges,
stays in a realistic PCE span, and the Voc/FF/Jsc split it derives from
any target PCE reproduces that exact PCE via Voc*Jsc*FF/Pin -- i.e. the
decomposition is not just "close enough", it is exact by construction.
"""

from __future__ import annotations

from datetime import date
from uuid import uuid4

import numpy as np
import pytest

from dssc_discovery.domain.enums import IlluminationProtocol, SolventType, SurfaceTreatment
from dssc_discovery.domain.experimental_condition import ExperimentalConditionVector
from dssc_discovery.domain.measurements import PhotovoltaicMeasurement
from dssc_discovery.features.synthetic import (
    _SAMPLED_SOLVENTS,
    _SAMPLED_SURFACE_TREATMENTS,
    _SAMPLING_RANGES,
    _choice,
    _jsc_ma_cm2_for_pce,
    _sample_vector,
    _synthetic_fill_factor,
    _synthetic_voc_v,
    synthetic_pce,
)
from dssc_discovery.validation.rules import check_pce_consistency


def _vector(**overrides) -> ExperimentalConditionVector:
    kwargs = dict(
        solvent=SolventType.ETHANOL,
        solvent_ratio=0.7,
        ph=5.5,
        extraction_temperature_c=25.0,
        extraction_time_min=60.0,
        pigment_concentration_mm=12.0,
        adsorption_time_min=120.0,
        sensitization_concentration_mm=0.5,
        sensitization_temperature_c=25.0,
        tio2_thickness_um=14.0,
        sintering_temperature_c=450.0,
        sintering_time_min=30.0,
        surface_treatment=SurfaceTreatment.NONE,
    )
    kwargs.update(overrides)
    return ExperimentalConditionVector(**kwargs)


# --- synthetic_pce -----------------------------------------------------------


def test_pce_is_maximal_at_the_documented_optimum():
    optimum_pce = synthetic_pce(_vector())
    perturbations = [
        {"ph": 1.0},
        {"ph": 12.0},
        {"pigment_concentration_mm": 0.5},
        {"pigment_concentration_mm": 29.0},
        {"adsorption_time_min": 1.0},
        {"adsorption_time_min": 500.0},
        {"tio2_thickness_um": 1.0},
        {"tio2_thickness_um": 40.0},
    ]
    for overrides in perturbations:
        assert synthetic_pce(_vector(**overrides)) < optimum_pce


def test_pce_stays_within_a_realistic_range_across_random_sampling():
    rng = np.random.default_rng(7)
    values = [synthetic_pce(_sample_vector(rng)) for _ in range(2000)]
    assert min(values) > 0.5
    assert max(values) < 10.0


def test_pce_is_deterministic_for_identical_input():
    vector = _vector(ph=6.2, pigment_concentration_mm=9.0)
    assert synthetic_pce(vector) == synthetic_pce(vector)


# --- Voc / FF / Jsc decomposition --------------------------------------------


def test_voc_and_fill_factor_stay_within_documented_bounds():
    rng = np.random.default_rng(11)
    for _ in range(500):
        vector = _sample_vector(rng)
        assert 0.55 <= _synthetic_voc_v(vector) <= 0.75
        assert 0.55 <= _synthetic_fill_factor(vector) <= 0.75


@pytest.mark.parametrize("target_pce", [1.0, 3.5, 5.07, 8.4])
def test_jsc_decomposition_reproduces_the_target_pce_exactly(target_pce):
    vector = _vector()
    voc = _synthetic_voc_v(vector)
    ff = _synthetic_fill_factor(vector)
    jsc = _jsc_ma_cm2_for_pce(target_pce, voc, ff)

    pv = PhotovoltaicMeasurement(
        device_id=uuid4(),
        illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
        measurement_date=date.today(),
        open_circuit_voltage_v=voc,
        short_circuit_current_density_ma_cm2=jsc,
        fill_factor=ff,
        power_conversion_efficiency_percent=target_pce,
        operator="x",
        instrument="SYNTHETIC",
    )
    check_pce_consistency(pv, relative_tolerance=1e-6)  # must not raise
    assert jsc > 0


# --- sampling helpers ---------------------------------------------------------


def test_choice_helper_preserves_enum_type_unlike_raw_rng_choice():
    rng = np.random.default_rng(3)
    picked = _choice(rng, _SAMPLED_SOLVENTS)
    assert isinstance(picked, SolventType)


def test_sample_vector_stays_within_the_documented_sampling_ranges():
    rng = np.random.default_rng(42)
    for _ in range(300):
        vector = _sample_vector(rng)
        for field_name, (lower, upper) in _SAMPLING_RANGES.items():
            value = getattr(vector, field_name)
            assert lower <= value <= upper, f"{field_name}={value} outside [{lower}, {upper}]"
        assert vector.solvent in _SAMPLED_SOLVENTS
        assert vector.surface_treatment in _SAMPLED_SURFACE_TREATMENTS
