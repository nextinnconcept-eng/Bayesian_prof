"""Synthetic dataset generator (Chunk 3.5): produces fake-but-
structurally-valid ExperimentalCondition / PhotoanodeFabricationBatch /
Device / PhotovoltaicMeasurement rows with a KNOWN, non-physical
ground-truth PCE relationship, so ML/BO code (Chunks 4.x-6.x) can be
built and tested before real experimental throughput exists.

Every entity produced here has is_synthetic=True. create_dataset_version
(Chunk 3.3) raises SyntheticDatasetError rather than silently blending a
mix of synthetic and real rows into one snapshot -- real data must never
be contaminated by fabricated rows without anyone noticing.

Scientific-integrity note (PLAN.txt Section 5/17): synthetic_pce() is
NOT a physical model of any real DSSC mechanism. It is a smooth,
single-peaked function of four of the ExperimentalConditionVector's
design-time fields (pH, pigment concentration, adsorption time, TiO2
thickness), chosen only so a working ML/BO pipeline has *something*
learnable to recover end-to-end. The specific optimum location and
shape carry no scientific claim and must never be cited as if they did.
The Voc / fill-factor sub-formulas exist only to produce a
PhotovoltaicMeasurement quadruple that is internally consistent (passes
check_pce_consistency by construction, verified explicitly below, not
just assumed) -- they are not a claim about how Voc/FF actually vary
with these parameters either.

The per-parameter sampling ranges below are a temporary stand-in for
the real per-species operational search space (SearchSpace, Chunk 6.1,
not yet built) -- narrower than ExperimentalConditionVector's own broad
physical-plausibility bounds, chosen only so random sampling produces
realistic-looking conditions.
"""

from __future__ import annotations

import math
from datetime import date
from typing import TypeVar

import numpy as np
from sqlalchemy.orm import Session

from dssc_discovery.database.repositories import (
    DeviceRepository,
    ExperimentalConditionRepository,
    PhotoanodeFabricationBatchRepository,
    PhotovoltaicMeasurementRepository,
    SpeciesRepository,
)
from dssc_discovery.domain.device import Device, PhotoanodeFabricationBatch
from dssc_discovery.domain.enums import (
    CounterElectrodeType,
    DesignSource,
    ElectrolyteType,
    IlluminationProtocol,
    SolventType,
    SpeciesCode,
    SurfaceTreatment,
)
from dssc_discovery.domain.experimental_condition import (
    ExperimentalCondition,
    ExperimentalConditionVector,
)
from dssc_discovery.domain.measurements import PhotovoltaicMeasurement
from dssc_discovery.domain.species import Species
from dssc_discovery.validation.exceptions import DomainValidationError
from dssc_discovery.validation.rules import (
    ILLUMINATION_POWER_MW_CM2,
    check_duplicate_condition,
    check_pce_consistency,
)


class SyntheticDatasetError(DomainValidationError):
    """Raised when synthetic and real rows would otherwise be mixed
    within one MLDatasetVersion snapshot (Chunk 3.3) -- a dataset that
    silently blends fabricated and real measurements could contaminate
    a real scientific result without anyone noticing.
    """


# --- ground-truth PCE function ---------------------------------------------

_PCE_OPTIMUM_PH = 5.5
_PCE_OPTIMUM_CONCENTRATION_MM = 12.0
_PCE_OPTIMUM_ADSORPTION_MIN = 120.0
_PCE_OPTIMUM_THICKNESS_UM = 14.0


def _gaussian_bump(x: float, optimum: float, width: float) -> float:
    return math.exp(-((x - optimum) ** 2) / (2 * width**2))


def synthetic_pce(vector: ExperimentalConditionVector) -> float:
    """Deterministic, noise-free ground-truth PCE (%) for a given
    condition vector. Single global optimum at ph=5.5,
    pigment_concentration_mm=12.0, adsorption_time_min=120.0,
    tio2_thickness_um=14.0 -- chosen only to sit inside this module's
    own _SAMPLING_RANGES below, not derived from any real measurement.
    Empirically verified (see tests/unit/test_synthetic.py) to stay
    within roughly [1.0, 8.5]%, a realistic single-digit DSSC PCE span,
    safely inside PhotovoltaicMeasurement's [0, 100] field bound.
    """
    ph_term = _gaussian_bump(vector.ph, _PCE_OPTIMUM_PH, 2.0)
    conc_term = _gaussian_bump(vector.pigment_concentration_mm, _PCE_OPTIMUM_CONCENTRATION_MM, 6.0)
    adsorption_term = _gaussian_bump(vector.adsorption_time_min, _PCE_OPTIMUM_ADSORPTION_MIN, 60.0)
    thickness_term = _gaussian_bump(vector.tio2_thickness_um, _PCE_OPTIMUM_THICKNESS_UM, 6.0)
    interaction = 0.15 * ph_term * adsorption_term
    raw_score = ph_term + conc_term + adsorption_term + thickness_term + interaction
    return 1.0 + raw_score * 1.8


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _synthetic_voc_v(vector: ExperimentalConditionVector) -> float:
    """Bounded, weakly pH-dependent Voc (V) -- exists only to produce a
    plausible-looking Voc/Jsc/FF split, not a scientific claim (see
    module docstring)."""
    return _clamp(0.68 - 0.01 * (vector.ph - _PCE_OPTIMUM_PH), 0.55, 0.75)


def _synthetic_fill_factor(vector: ExperimentalConditionVector) -> float:
    """Bounded, weakly thickness-dependent FF -- same caveat as
    _synthetic_voc_v."""
    return _clamp(
        0.68 - 0.004 * abs(vector.tio2_thickness_um - _PCE_OPTIMUM_THICKNESS_UM), 0.55, 0.75
    )


def _jsc_ma_cm2_for_pce(pce_percent: float, voc_v: float, fill_factor: float) -> float:
    """Solve Jsc algebraically so Voc*Jsc*FF/Pin*100 == pce_percent
    exactly (Pin = the AM1.5G 1-sun power density check_pce_consistency
    itself uses) -- guarantees every generated row passes
    check_pce_consistency by construction, not by chance."""
    power_density = ILLUMINATION_POWER_MW_CM2[IlluminationProtocol.AM1_5G_1_SUN]
    return (pce_percent / 100.0) * power_density / (voc_v * fill_factor)


# --- sampling ranges (temporary stand-in for Chunk 6.1's SearchSpace) ------

_SAMPLING_RANGES: dict[str, tuple[float, float]] = {
    "solvent_ratio": (0.5, 1.0),
    "ph": (2.0, 9.0),
    "extraction_temperature_c": (20.0, 80.0),
    "extraction_time_min": (15.0, 180.0),
    "pigment_concentration_mm": (1.0, 30.0),
    "adsorption_time_min": (30.0, 300.0),
    "sensitization_concentration_mm": (0.1, 2.0),
    "sensitization_temperature_c": (20.0, 60.0),
    "tio2_thickness_um": (5.0, 25.0),
    "sintering_temperature_c": (400.0, 500.0),
    "sintering_time_min": (15.0, 60.0),
}
_SAMPLED_SOLVENTS: tuple[SolventType, ...] = (
    SolventType.WATER,
    SolventType.ETHANOL,
    SolventType.METHANOL,
    SolventType.ACETONE,
    SolventType.ACIDIFIED_ETHANOL,
    SolventType.MIXED,
)
_SAMPLED_SURFACE_TREATMENTS: tuple[SurfaceTreatment, ...] = (
    SurfaceTreatment.NONE,
    SurfaceTreatment.TICL4,
    SurfaceTreatment.MGO_COATING,
)
_NOISE_STD_PERCENT_DEFAULT = 0.3
_NOISY_PCE_BOUNDS = (0.05, 25.0)


_ChoiceT = TypeVar("_ChoiceT")


def _choice(rng: np.random.Generator, options: tuple[_ChoiceT, ...]) -> _ChoiceT:
    # Plain Python indexing, not rng.choice(options) directly: numpy
    # coerces a sequence of StrEnum members into a bare numpy str_
    # array, silently discarding the enum type -- verified this loses
    # SolventType/SurfaceTreatment before picking this approach.
    return options[int(rng.integers(0, len(options)))]


def _sample_vector(rng: np.random.Generator) -> ExperimentalConditionVector:
    return ExperimentalConditionVector(
        solvent=_choice(rng, _SAMPLED_SOLVENTS),
        solvent_ratio=float(rng.uniform(*_SAMPLING_RANGES["solvent_ratio"])),
        ph=float(rng.uniform(*_SAMPLING_RANGES["ph"])),
        extraction_temperature_c=float(rng.uniform(*_SAMPLING_RANGES["extraction_temperature_c"])),
        extraction_time_min=float(rng.uniform(*_SAMPLING_RANGES["extraction_time_min"])),
        pigment_concentration_mm=float(rng.uniform(*_SAMPLING_RANGES["pigment_concentration_mm"])),
        adsorption_time_min=float(rng.uniform(*_SAMPLING_RANGES["adsorption_time_min"])),
        sensitization_concentration_mm=float(
            rng.uniform(*_SAMPLING_RANGES["sensitization_concentration_mm"])
        ),
        sensitization_temperature_c=float(
            rng.uniform(*_SAMPLING_RANGES["sensitization_temperature_c"])
        ),
        tio2_thickness_um=float(rng.uniform(*_SAMPLING_RANGES["tio2_thickness_um"])),
        sintering_temperature_c=float(rng.uniform(*_SAMPLING_RANGES["sintering_temperature_c"])),
        sintering_time_min=float(rng.uniform(*_SAMPLING_RANGES["sintering_time_min"])),
        surface_treatment=_choice(rng, _SAMPLED_SURFACE_TREATMENTS),
    )


def generate_synthetic_condition_and_device(
    session: Session,
    vector: ExperimentalConditionVector,
    species_code: SpeciesCode,
    condition_code: str,
    device_code: str,
    operator: str,
    rng: np.random.Generator,
    noise_std_percent: float = _NOISE_STD_PERCENT_DEFAULT,
) -> Device:
    """Persist one synthetic ExperimentalCondition + PhotoanodeFabricationBatch
    + Device + PhotovoltaicMeasurement, all is_synthetic=True, for the
    given parameter vector. The recorded PCE is synthetic_pce(vector)
    plus Gaussian noise (std=noise_std_percent); Voc/Jsc/FF are
    constructed so the row passes check_pce_consistency, which is
    called explicitly before persisting as an independent check on the
    construction, not merely assumed correct.

    Caller must session.commit() -- repositories only flush (Chunk 1.7
    convention), so a batch of these can be committed atomically.
    """
    species_repo = SpeciesRepository(session)
    condition_repo = ExperimentalConditionRepository(session)
    photoanode_repo = PhotoanodeFabricationBatchRepository(session)
    device_repo = DeviceRepository(session)
    pv_repo = PhotovoltaicMeasurementRepository(session)

    if species_repo.get_one(code=species_code) is None:
        species_repo.add(Species(code=species_code, operator=operator, is_synthetic=True))

    condition = ExperimentalCondition(
        species_code=species_code,
        condition_code=condition_code,
        design_source=DesignSource.MANUAL,
        target_parameters=vector,
        operator=operator,
        is_synthetic=True,
    )
    check_duplicate_condition(condition_repo, condition)
    condition = condition_repo.add(condition)

    photoanode_batch = photoanode_repo.add(
        PhotoanodeFabricationBatch(
            batch_code=f"{condition_code}-FAB",
            tio2_thickness_um=vector.tio2_thickness_um,
            sintering_temperature_c=vector.sintering_temperature_c,
            sintering_time_min=vector.sintering_time_min,
            surface_treatment=vector.surface_treatment,
            operator=operator,
            is_synthetic=True,
        )
    )

    device = device_repo.add(
        Device(
            experimental_condition_id=condition.id,
            photoanode_fabrication_batch_id=photoanode_batch.id,
            device_code=device_code,
            replicate_number=1,
            active_area_cm2=0.25,
            electrolyte_type=ElectrolyteType.IODIDE_TRIIODIDE,
            counter_electrode_type=CounterElectrodeType.PLATINUM,
            operator=operator,
            is_synthetic=True,
        )
    )

    true_pce = synthetic_pce(vector)
    noisy_pce = _clamp(true_pce + float(rng.normal(0.0, noise_std_percent)), *_NOISY_PCE_BOUNDS)
    voc = _synthetic_voc_v(vector)
    fill_factor = _synthetic_fill_factor(vector)
    jsc = _jsc_ma_cm2_for_pce(noisy_pce, voc, fill_factor)

    pv_measurement = PhotovoltaicMeasurement(
        device_id=device.id,
        illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
        measurement_date=date.today(),
        open_circuit_voltage_v=voc,
        short_circuit_current_density_ma_cm2=jsc,
        fill_factor=fill_factor,
        power_conversion_efficiency_percent=noisy_pce,
        operator=operator,
        instrument="SYNTHETIC",
        is_synthetic=True,
    )
    check_pce_consistency(pv_measurement)
    pv_repo.add(pv_measurement)

    return device


def generate_synthetic_dataset(
    session: Session,
    n: int,
    seed: int,
    species_code: SpeciesCode = SpeciesCode.BETA_VULGARIS,
    operator: str = "synthetic-generator",
    condition_code_prefix: str | None = None,
    noise_std_percent: float = _NOISE_STD_PERCENT_DEFAULT,
) -> list[Device]:
    """Generate and persist n synthetic devices (with conditions,
    fabrication batches, and PV measurements) for one species, sampling
    parameter vectors uniformly at random within _SAMPLING_RANGES.

    Deterministic for a given seed: the same seed always reproduces the
    same dataset (PLAN.txt's reproducibility requirement), even though
    the recorded PCE includes noise -- the noise itself is drawn from
    the seeded generator, not from unseeded system randomness.

    Caller must session.commit().
    """
    if n <= 0:
        raise ValueError("n must be positive")

    rng = np.random.default_rng(seed)
    prefix = condition_code_prefix or f"{species_code.name}-SYN"
    devices = []
    for i in range(n):
        vector = _sample_vector(rng)
        condition_code = f"{prefix}-{i:04d}"
        device_code = f"{condition_code}-R1"
        device = generate_synthetic_condition_and_device(
            session,
            vector,
            species_code,
            condition_code,
            device_code,
            operator,
            rng,
            noise_std_percent=noise_std_percent,
        )
        devices.append(device)
    return devices
