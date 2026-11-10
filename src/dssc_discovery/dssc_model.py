"""Simple DSSC device model: the document's four equations, a single-diode
model and Beer-Lambert absorption.

Pure calculation, no UI or database code. Units follow the source document:
lengths in cm, current as a current density in A/cm^2, voltage in V,
temperature in K, light as irradiance in W/m^2 (1000 W/m^2 = 100 mW/cm^2).

Temperature enters only through kT (Equations 2-4 and the diode term); the
single-diode saturation current I0 is held constant on purpose, to keep the
model simple.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

G_REF = 1000.0  # W/m^2, the irradiance at which phi and Iph are specified
BEER_LAMBERT_FACTOR = 2.303


@dataclass(frozen=True)
class DSSCParams:
    """Defaults are the values given in the source document (T is the working point
    inside its 273-375 K range)."""

    k: float = 1.381e-23  # Boltzmann constant, J/K
    q: float = 1.602e-19  # electron charge, C
    L: float = 2.2361e-3  # electron diffusion length, cm
    d: float = 10e-4  # film thickness / optical path length, cm
    alpha: float = 5000.0  # absorption coefficient, 1/cm
    m: float = 4.5  # ideality factor of the document's equations
    D: float = 5.0e-4  # diffusion coefficient, cm^2/s
    n0: float = 10e16  # equilibrium electron density, 1/cm^3
    phi: float = 1e17  # photon flux at 1000 W/m^2, 1/(cm^2 s)
    T: float = 300.0  # temperature, K


@dataclass(frozen=True)
class SingleDiodeParams:
    """Single-diode model parameters (per cm^2 of cell).

    The source document gives no values for these, so the defaults are typical
    assumptions and the user is expected to change them."""

    iph: float = 15.4e-3  # photocurrent at 1000 W/m^2, A/cm^2
    i0: float = 1e-7  # diode saturation current, A/cm^2
    n: float = 1.5  # diode ideality factor
    rs: float = 2.0  # series resistance, ohm cm^2
    rsh: float = 500.0  # shunt resistance, ohm cm^2


@dataclass(frozen=True)
class Performance:
    """Figures of merit read off an I-V curve (current in mA/cm^2, power in mW/cm^2)."""

    voc: float  # V
    jsc: float  # mA/cm^2
    vmp: float  # V
    imp: float  # mA/cm^2
    pmax: float  # mW/cm^2
    ff: float
    efficiency: float  # percent
    pin: float  # mW/cm^2


def _positive(**values: float) -> None:
    for name, value in values.items():
        if not value > 0:
            raise ValueError(f"{name} must be greater than 0 (got {value})")


# --- absorption coefficient ---------------------------------------------------


def absorption_coefficient(absorbance: float | np.ndarray, path_length_cm: float):
    """Beer-Lambert law: alpha = 2.303 * A / d, in 1/cm."""
    _positive(path_length_cm=path_length_cm)
    return BEER_LAMBERT_FACTOR * np.asarray(absorbance, dtype=float) / path_length_cm


# --- the document's four equations ---------------------------------------------


def short_circuit_current(p: DSSCParams, irradiance: float = G_REF) -> float:
    """Equation 1: short-circuit current density in A/cm^2.

    The photon flux scales linearly with irradiance."""
    _positive(L=p.L, d=p.d, alpha=p.alpha, phi=p.phi, irradiance=irradiance)
    phi = p.phi * irradiance / G_REF
    alpha = p.alpha
    if abs(p.L * alpha - 1.0) < 1e-9:  # Eq. 1 is 0/0 at L*alpha = 1; use the limit
        alpha *= 1 + 1e-7
    la = p.L * alpha
    bracket = -la + math.tanh(p.d / p.L) + la * math.exp(-p.d * alpha) / math.cosh(p.d / p.L)
    return p.q * phi * alpha * p.L / (1 - la**2) * bracket


def _dark_current_scale(p: DSSCParams) -> float:
    """q*D*n0/L * tanh(d/L), the prefactor shared by Equations 2-4 (A/cm^2)."""
    return p.q * p.D * p.n0 / p.L * math.tanh(p.d / p.L)


def open_circuit_voltage(p: DSSCParams, isc: float) -> float:
    """Equation 2: Voc = (k T m / q) ln[L Isc / (q D n0 tanh(d/L)) + 1]."""
    _positive(isc=isc, T=p.T, m=p.m)
    return p.k * p.T * p.m / p.q * math.log(isc / _dark_current_scale(p) + 1)


def voltage_at_current(p: DSSCParams, current: float | np.ndarray, isc: float):
    """Equation 3: V = (k T m / q) ln[L (Isc - I) / (q D n0 tanh(d/L)) + 1]."""
    return p.k * p.T * p.m / p.q * np.log((isc - np.asarray(current)) / _dark_current_scale(p) + 1)


def current_at_voltage(p: DSSCParams, voltage: float | np.ndarray, isc: float):
    """Equation 4: I = Isc - (q D n0 / L) tanh(d/L) [exp(qV / (k T m)) - 1]."""
    x = p.q * np.asarray(voltage, dtype=float) / (p.k * p.T * p.m)
    return isc - _dark_current_scale(p) * np.expm1(x)


def equations_curve(
    p: DSSCParams, irradiance: float = G_REF, points: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """I-V curve from Equations 1, 2 and 4, from 0 V to Voc (A/cm^2)."""
    isc = short_circuit_current(p, irradiance)
    voc = open_circuit_voltage(p, isc)
    v = np.linspace(0.0, voc, points)
    return v, current_at_voltage(p, v, isc)


# --- single-diode model --------------------------------------------------------


def single_diode_current(
    p: DSSCParams, sd: SingleDiodeParams, voltage: float, irradiance: float = G_REF
) -> float:
    """Solve I = Iph - I0 [exp(q (V + I Rs) / (n k T)) - 1] - (V + I Rs) / Rsh for I.

    The equation is implicit in I, so it is solved numerically for each voltage."""
    _positive(i0=sd.i0, n=sd.n, rsh=sd.rsh, T=p.T)
    iph = sd.iph * irradiance / G_REF
    vt = sd.n * p.k * p.T / p.q

    def f(i: float) -> float:
        x = voltage + i * sd.rs
        return iph - sd.i0 * math.expm1(min(x / vt, 700.0)) - x / sd.rsh - i

    low = -(abs(iph) + voltage / sd.rsh + 1.0)
    return brentq(f, low, iph, xtol=1e-14)


def single_diode_curve(
    p: DSSCParams, sd: SingleDiodeParams, irradiance: float = G_REF, points: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """I-V curve of the single-diode model from 0 V to Voc (A/cm^2)."""
    _positive(iph=sd.iph, i0=sd.i0, n=sd.n, rsh=sd.rsh, T=p.T, irradiance=irradiance)
    iph = sd.iph * irradiance / G_REF
    vt = sd.n * p.k * p.T / p.q
    # Voc is the voltage where I = 0 (Rs drops out); ignoring Rsh gives an upper bound.
    upper = vt * math.log(iph / sd.i0 + 1)
    voc = brentq(lambda v: iph - sd.i0 * math.expm1(v / vt) - v / sd.rsh, 0.0, upper, xtol=1e-12)
    v = np.linspace(0.0, voc, points)
    return v, np.array([single_diode_current(p, sd, float(x), irradiance) for x in v])


# --- figures of merit ----------------------------------------------------------


def performance(v: np.ndarray, i: np.ndarray, irradiance: float = G_REF) -> Performance:
    """Voc, Jsc, Vmp, Imp, Pmax, FF and efficiency from an I-V curve (A/cm^2 in).

    FF = Pmax / (Voc Jsc);  efficiency = Jsc Voc FF / Pin * 100, with
    Pin = irradiance / 10 mW/cm^2 (so 1000 W/m^2 -> 100 mW/cm^2)."""
    j = np.asarray(i) * 1e3  # mA/cm^2
    power = np.asarray(v) * j  # mW/cm^2
    k = int(np.argmax(power))
    voc, jsc, pmax = float(v[-1]), float(j[0]), float(power[k])
    ff = pmax / (voc * jsc) if voc * jsc > 0 else 0.0
    pin = irradiance / 10.0
    return Performance(
        voc=voc,
        jsc=jsc,
        vmp=float(v[k]),
        imp=float(j[k]),
        pmax=pmax,
        ff=ff,
        efficiency=jsc * voc * ff / pin * 100.0,
        pin=pin,
    )
