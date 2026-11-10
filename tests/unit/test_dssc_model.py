"""DSSC Simulator: calculations (hand-checked against the source equations) and saving."""

from __future__ import annotations

import math

import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from dssc_discovery.database.models import Base
from dssc_discovery.dssc_model import (
    DSSCParams,
    SingleDiodeParams,
    absorption_coefficient,
    current_at_voltage,
    equations_curve,
    open_circuit_voltage,
    performance,
    short_circuit_current,
    single_diode_curve,
    voltage_at_current,
)
from dssc_discovery.dssc_runs import runs_frame, save_run

P = DSSCParams()


def test_beer_lambert():
    assert absorption_coefficient(2.1715, 1e-3) == pytest.approx(5001.0, rel=1e-3)
    with pytest.raises(ValueError):
        absorption_coefficient(1.0, 0.0)


def test_document_defaults_give_expected_values():
    isc = short_circuit_current(P)
    assert isc * 1e3 == pytest.approx(15.44, abs=0.05)  # mA/cm^2
    assert open_circuit_voltage(P, isc) == pytest.approx(0.282, abs=0.003)


def test_isc_scales_with_irradiance():
    assert short_circuit_current(P, 320) == pytest.approx(0.32 * short_circuit_current(P, 1000))


def test_equation_4_endpoints_and_equation_3_is_its_inverse():
    isc = short_circuit_current(P)
    voc = open_circuit_voltage(P, isc)
    assert current_at_voltage(P, 0.0, isc) == pytest.approx(isc)
    assert current_at_voltage(P, voc, isc) == pytest.approx(0.0, abs=1e-9)
    v = np.array([0.05, 0.15, 0.25])
    assert voltage_at_current(P, current_at_voltage(P, v, isc), isc) == pytest.approx(v)


def test_performance_is_consistent():
    v, i = equations_curve(P)
    perf = performance(v, i)
    assert perf.pin == 100.0
    assert 0 < perf.ff < 1
    assert perf.pmax == pytest.approx(perf.vmp * perf.imp)
    assert perf.ff == pytest.approx(perf.pmax / (perf.voc * perf.jsc))
    assert perf.efficiency == pytest.approx(perf.pmax)  # Pin = 100 mW/cm^2


def test_lower_light_lowers_current_power_and_voltage():
    full = performance(*equations_curve(P, 1000), 1000)
    low = performance(*equations_curve(P, 320), 320)
    assert low.jsc < full.jsc and low.voc < full.voc and low.pmax < full.pmax


def test_single_diode_satisfies_its_own_equation():
    sd = SingleDiodeParams()
    v, i = single_diode_curve(P, sd)
    vt = sd.n * P.k * P.T / P.q
    x = v + i * sd.rs
    residual = sd.iph - sd.i0 * np.expm1(x / vt) - x / sd.rsh - i
    assert np.max(np.abs(residual)) < 1e-9
    assert i[0] == pytest.approx(sd.iph, rel=1e-2)
    assert i[-1] == pytest.approx(0.0, abs=1e-8)
    perf = performance(v, i)
    assert 0 < perf.ff < 1 and perf.efficiency > 0


def test_single_diode_voc_rises_with_light_and_with_temperature():
    # I0 is held constant (see dssc_model docstring), so Voc = (nkT/q) ln(Iph/I0) grows with T,
    # as the document's Equation 2 also does. Real cells lose Voc with T because I0 grows.
    sd = SingleDiodeParams()
    assert (
        performance(*single_diode_curve(P, sd, 1000), 1000).voc
        > performance(*single_diode_curve(P, sd, 320), 320).voc
    )
    hot = DSSCParams(T=375.0)
    assert (
        performance(*single_diode_curve(hot, sd, 1000), 1000).voc
        > performance(*single_diode_curve(P, sd, 1000), 1000).voc
    )


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        short_circuit_current(DSSCParams(L=0.0))
    with pytest.raises(ValueError):
        single_diode_curve(P, SingleDiodeParams(rsh=-1.0))
    assert math.isfinite(short_circuit_current(DSSCParams(alpha=1 / P.L)))  # L*alpha = 1


def test_save_and_list_runs():
    engine = sa.create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assert runs_frame(session).empty
        v, i = equations_curve(P)
        perf = performance(v, i)
        save_run(
            session,
            operator="admin",
            label="first",
            model="equations",
            params=P,
            irradiance=1000.0,
            area_cm2=0.5,
            perf=perf,
            uvvis=[(450.0, 0.8), (520.0, 1.2)],
            uvvis_wavelength_nm=520.0,
        )
        frame = runs_frame(session)
    assert len(frame) == 1
    assert frame.loc[0, "Label"] == "first"
    assert frame.loc[0, "Isc (mA)"] == pytest.approx(perf.jsc * 0.5)
    assert frame.loc[0, "η (%)"] == pytest.approx(perf.efficiency)
