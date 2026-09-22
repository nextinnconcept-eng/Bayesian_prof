import pytest

from dssc_discovery.domain.units_registry import UnitConversionError, to_canonical


def test_temperature_kelvin_to_canonical_celsius():
    # 298.15 K == 25 degC; extraction_temperature_c's canonical unit is degC
    result = to_canonical(298.15, "kelvin", "extraction_temperature_c")
    assert result == pytest.approx(25.0, abs=1e-6)


def test_temperature_fahrenheit_to_canonical_celsius():
    # 77 degF == 25 degC
    result = to_canonical(77, "degF", "sensitization_temperature_c")
    assert result == pytest.approx(25.0, abs=1e-6)


def test_concentration_molar_to_canonical_millimolar():
    # 1 M == 1000 mM; pigment_concentration_mm's canonical unit is millimolar
    result = to_canonical(1, "molar", "pigment_concentration_mm")
    assert result == pytest.approx(1000.0)


def test_time_hour_to_canonical_minute():
    result = to_canonical(1, "hour", "extraction_time_min")
    assert result == pytest.approx(60.0)


def test_fraction_to_canonical_percent():
    # 0.9 dimensionless == 90 percent; retention_percent's canonical unit is percent
    result = to_canonical(0.9, "dimensionless", "retention_percent")
    assert result == pytest.approx(90.0)


def test_same_unit_is_a_no_op():
    result = to_canonical(42.0, "degC", "extraction_temperature_c")
    assert result == pytest.approx(42.0)


def test_unknown_field_raises_unit_conversion_error():
    with pytest.raises(UnitConversionError, match="Unknown field"):
        to_canonical(1.0, "degC", "not_a_registered_field")


def test_undefined_unit_raises_unit_conversion_error():
    with pytest.raises(UnitConversionError, match="Unrecognized unit"):
        to_canonical(1.0, "bananas", "extraction_temperature_c")


def test_incompatible_dimensionality_raises_unit_conversion_error():
    # minutes cannot convert to degC
    with pytest.raises(UnitConversionError, match="not compatible"):
        to_canonical(5.0, "minute", "extraction_temperature_c")
