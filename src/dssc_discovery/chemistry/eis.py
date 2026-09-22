"""EIS equivalent-circuit fitting: raw impedance spectrum -> Rs, Rct,
Rrec, electron lifetime via nonlinear least-squares fit to a two-arc
Randles-type circuit.

Circuit model (documented explicitly, per Chunk 2.4's requirement):

    Z(omega) = Rs + Rct/(1 + j*omega*Rct*Cce) + Rrec/(1 + j*omega*Rrec*Cmu)

Rs: series resistance (electrolyte + contacts + FTO sheet resistance).
Rct/Cce: charge-transfer resistance/capacitance at the counter
    electrode (typically the high-frequency arc).
Rrec/Cmu: recombination resistance/chemical capacitance at the
    TiO2/dye/electrolyte interface (typically the mid-frequency arc).
electron_lifetime = Rrec * Cmu -- the standard DSSC definition: the
    characteristic time constant of electron recombination.

This is the standard simplified two-time-constant DSSC equivalent
circuit (omitting Warburg diffusion, appropriate at the frequencies
where these two arcs dominate). EquivalentCircuitModel.RANDLES
(Chunk 0.6) refers to this model specifically.

Two DISTINCT failure modes, both explicit rather than silently
producing garbage (PLAN.txt Section 16, "incomplete EIS fitting"):
- Hard failure (fit_eis_spectrum returns converged=False, no parameter
  values): the optimizer cannot run at all -- non-finite input data, or
  the iteration budget is exhausted without convergence.
- A "successful" fit always reports fit_r_squared alongside whatever
  parameters it found, however good or bad that fit actually is --
  verified empirically that scipy.optimize.least_squares.success is
  near-useless as a quality signal on its own (it returns True even
  fitting these five parameters to pure random noise, R^2=-0.016).
  Judging whether a given R^2 is "good enough" is a downstream decision
  (Chunk 1.6-style validation, or a human review), not this module's.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from scipy.optimize import least_squares

from dssc_discovery.chemistry.raw_storage import resolve_raw_file
from dssc_discovery.domain.enums import EquivalentCircuitModel, MissingValueReason
from dssc_discovery.domain.measurements import ElectrochemicalMeasurement
from dssc_discovery.logging_setup import get_logger

logger = get_logger(__name__)

PROCESSING_VERSION = "eis-v1"
_N_PARAMS = 5  # Rs, Rct, Cce, Rrec, Cmu


class EISProcessingError(Exception):
    """Raised for malformed input data (parsing) -- NOT for fit
    non-convergence. A non-convergent fit is a normal, anticipated
    outcome represented by EISFitResult.converged=False, never an
    exception.
    """


@dataclass(frozen=True)
class EISFitResult:
    converged: bool
    series_resistance_ohm: float | None
    charge_transfer_resistance_ohm: float | None
    recombination_resistance_ohm: float | None
    electron_lifetime_ms: float | None
    fit_r_squared: float | None
    failure_reason: str | None


def parse_eis_csv(raw_bytes: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a three-column (frequency_hz, z_real_ohm, z_imag_ohm) CSV
    with a header row into frequency-sorted numpy arrays. z_imag uses
    the physics sign convention (negative for this circuit's capacitive
    arcs), not the "-Z''" Nyquist-plot display convention.
    """
    text = raw_bytes.decode("utf-8")
    frequencies: list[float] = []
    z_reals: list[float] = []
    z_imags: list[float] = []
    for row in csv.DictReader(text.splitlines()):
        try:
            frequencies.append(float(row["frequency_hz"]))
            z_reals.append(float(row["z_real_ohm"]))
            z_imags.append(float(row["z_imag_ohm"]))
        except (KeyError, ValueError) as exc:
            raise EISProcessingError(f"Malformed EIS CSV row: {row}") from exc

    if len(frequencies) < _N_PARAMS:
        raise EISProcessingError(
            f"EIS spectrum needs at least {_N_PARAMS} points to fit a "
            f"{_N_PARAMS}-parameter circuit, got {len(frequencies)}"
        )

    frequency = np.asarray(frequencies, dtype=float)
    z_real = np.asarray(z_reals, dtype=float)
    z_imag = np.asarray(z_imags, dtype=float)
    order = np.argsort(frequency)
    return frequency[order], z_real[order], z_imag[order]


def _randles_two_arc_impedance(params: np.ndarray, omega: np.ndarray) -> np.ndarray:
    rs, rct, cce, rrec, cmu = params
    z_ce = rct / (1 + 1j * omega * rct * cce)
    z_rec = rrec / (1 + 1j * omega * rrec * cmu)
    return rs + z_ce + z_rec


def _residuals(
    params: np.ndarray, omega: np.ndarray, z_real: np.ndarray, z_imag: np.ndarray
) -> np.ndarray:
    z_fit = _randles_two_arc_impedance(params, omega)
    return np.concatenate([z_fit.real - z_real, z_fit.imag - z_imag])


def _initial_guess(frequency: np.ndarray, z_real: np.ndarray) -> list[float]:
    """A standard EIS heuristic: Rs from the highest-frequency
    intercept, the remaining resistance range split unevenly between
    the two arcs, generic mid-range capacitance guesses -- verified
    empirically that this converges the fit exactly on clean synthetic
    data. All fit parameters are bounded to (0, inf) (physical
    resistances/capacitances can't be negative), so every guess is
    clamped to a small positive value rather than risk violating those
    bounds on pathological input (e.g. noise straddling zero) -- an
    out-of-bounds initial guess would otherwise make the optimizer
    itself raise before it ever gets to attempt a fit.
    """
    rs_guess = max(float(z_real[np.argmax(frequency)]), 1e-6)
    total_r_guess = float(z_real[np.argmin(frequency)]) - rs_guess
    total_r_guess = max(total_r_guess, 1e-6)
    return [rs_guess, total_r_guess * 0.3, 1e-5, total_r_guess * 0.7, 1e-5]


def fit_eis_spectrum(
    frequency_hz: np.ndarray,
    z_real_ohm: np.ndarray,
    z_imag_ohm: np.ndarray,
    max_nfev: int | None = None,
) -> EISFitResult:
    """Fit the two-arc Randles-type circuit via nonlinear least squares.
    Never raises for a non-convergent fit -- returns converged=False
    with a documented failure_reason instead (Section 16).
    """
    omega = 2 * np.pi * frequency_hz
    x0 = _initial_guess(frequency_hz, z_real_ohm)

    try:
        result = least_squares(
            _residuals,
            x0=x0,
            args=(omega, z_real_ohm, z_imag_ohm),
            bounds=(0, np.inf),
            method="trf",
            max_nfev=max_nfev,
        )
    except ValueError as exc:
        return EISFitResult(
            converged=False,
            series_resistance_ohm=None,
            charge_transfer_resistance_ohm=None,
            recombination_resistance_ohm=None,
            electron_lifetime_ms=None,
            fit_r_squared=None,
            failure_reason=f"Fit could not run: {exc}",
        )

    if not result.success:
        return EISFitResult(
            converged=False,
            series_resistance_ohm=None,
            charge_transfer_resistance_ohm=None,
            recombination_resistance_ohm=None,
            electron_lifetime_ms=None,
            fit_r_squared=None,
            failure_reason=f"Optimizer did not converge: {result.message}",
        )

    rs, rct, cce, rrec, cmu = result.x
    data = np.concatenate([z_real_ohm, z_imag_ohm])
    ss_res = float(np.sum(result.fun**2))
    ss_tot = float(np.sum((data - np.mean(data)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return EISFitResult(
        converged=True,
        series_resistance_ohm=float(rs),
        charge_transfer_resistance_ohm=float(rct),
        recombination_resistance_ohm=float(rrec),
        electron_lifetime_ms=float(rrec * cmu * 1000.0),
        fit_r_squared=r_squared,
        failure_reason=None,
    )


def derive_eis_measurement(
    device_id: UUID,
    source_spectrum_id: UUID,
    raw_data_reference: str,
    operator: str,
    instrument: str,
    max_nfev: int | None = None,
) -> ElectrochemicalMeasurement:
    """Full pipeline: resolve the raw file (Chunk 2.1), parse it, fit
    the circuit, and package the result as an ElectrochemicalMeasurement
    domain object -- ALWAYS produced, whether the fit converged or not
    (Section 16): a non-convergent fit still documents the attempt
    (missing_value_reason explains absence) rather than producing
    nothing at all.
    """
    raw_bytes = resolve_raw_file(raw_data_reference)
    frequency, z_real, z_imag = parse_eis_csv(raw_bytes)
    fit = fit_eis_spectrum(frequency, z_real, z_imag, max_nfev=max_nfev)

    if fit.converged:
        logger.info(
            "EIS fit converged: Rs=%.2f Rct=%.2f Rrec=%.2f tau=%.2fms R^2=%.4f",
            fit.series_resistance_ohm,
            fit.charge_transfer_resistance_ohm,
            fit.recombination_resistance_ohm,
            fit.electron_lifetime_ms,
            fit.fit_r_squared,
        )
        return ElectrochemicalMeasurement(
            device_id=device_id,
            raw_spectrum_id=source_spectrum_id,
            equivalent_circuit_model=EquivalentCircuitModel.RANDLES,
            series_resistance_ohm=fit.series_resistance_ohm,
            charge_transfer_resistance_ohm=fit.charge_transfer_resistance_ohm,
            recombination_resistance_ohm=fit.recombination_resistance_ohm,
            electron_lifetime_ms=fit.electron_lifetime_ms,
            fit_r_squared=fit.fit_r_squared,
            operator=operator,
            instrument=instrument,
        )

    logger.warning("EIS fit did not converge: %s", fit.failure_reason)
    return ElectrochemicalMeasurement(
        device_id=device_id,
        raw_spectrum_id=source_spectrum_id,
        equivalent_circuit_model=EquivalentCircuitModel.RANDLES,
        missing_value_reason=MissingValueReason.OTHER,
        notes=f"EIS fit did not converge: {fit.failure_reason}",
        operator=operator,
        instrument=instrument,
    )
