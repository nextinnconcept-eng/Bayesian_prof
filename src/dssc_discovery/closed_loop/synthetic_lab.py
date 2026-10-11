"""Simulated fabrication + measurement for the M7 loop: writes synthetic
PhotoanodeFabricationBatch / Device / PhotovoltaicMeasurement rows for an
existing condition, using the same internally-consistent Voc/Jsc/FF
construction as the 3.5 generator (every row passes
check_pce_consistency). All rows are is_synthetic=True.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from dssc_discovery.database.repositories import (
    DeviceRepository,
    PhotoanodeFabricationBatchRepository,
    PhotovoltaicMeasurementRepository,
)
from dssc_discovery.domain.device import Device, PhotoanodeFabricationBatch
from dssc_discovery.domain.enums import (
    CounterElectrodeType,
    ElectrolyteType,
    IlluminationProtocol,
)
from dssc_discovery.domain.experimental_condition import ExperimentalCondition
from dssc_discovery.domain.measurements import PhotovoltaicMeasurement
from dssc_discovery.features.synthetic import (
    _jsc_ma_cm2_for_pce,
    _synthetic_fill_factor,
    _synthetic_voc_v,
)
from dssc_discovery.validation.rules import check_pce_consistency


def record_synthetic_devices(
    session: Session, condition: ExperimentalCondition, pce_values: list[float], operator: str
) -> list[Device]:
    if not condition.is_synthetic:
        raise ValueError("synthetic_lab only records measurements for synthetic conditions")
    vector = condition.target_parameters
    batch = PhotoanodeFabricationBatchRepository(session).add(
        PhotoanodeFabricationBatch(
            batch_code=f"{condition.condition_code}-FAB",
            tio2_thickness_um=vector.tio2_thickness_um,
            sintering_temperature_c=vector.sintering_temperature_c,
            sintering_time_min=vector.sintering_time_min,
            surface_treatment=vector.surface_treatment,
            operator=operator,
            is_synthetic=True,
        )
    )
    voc, fill_factor = _synthetic_voc_v(vector), _synthetic_fill_factor(vector)
    devices = []
    for replicate, pce in enumerate(pce_values, start=1):
        device = DeviceRepository(session).add(
            Device(
                experimental_condition_id=condition.id,
                photoanode_fabrication_batch_id=batch.id,
                device_code=f"{condition.condition_code}-R{replicate}",
                replicate_number=replicate,
                active_area_cm2=0.25,
                electrolyte_type=ElectrolyteType.IODIDE_TRIIODIDE,
                counter_electrode_type=CounterElectrodeType.PLATINUM,
                operator=operator,
                is_synthetic=True,
            )
        )
        measurement = PhotovoltaicMeasurement(
            device_id=device.id,
            illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
            measurement_date=date.today(),
            open_circuit_voltage_v=voc,
            short_circuit_current_density_ma_cm2=_jsc_ma_cm2_for_pce(pce, voc, fill_factor),
            fill_factor=fill_factor,
            power_conversion_efficiency_percent=pce,
            operator=operator,
            instrument="SYNTHETIC",
            is_synthetic=True,
        )
        check_pce_consistency(measurement)
        PhotovoltaicMeasurementRepository(session).add(measurement)
        devices.append(device)
    return devices
