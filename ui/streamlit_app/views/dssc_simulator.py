"""DSSC Simulator: UV-Vis absorption, the document's DSSC equations, the single-diode
model, and I-V / P-V curves under different irradiation and temperature."""

from __future__ import annotations

import math
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from _shared import AQUA, BLUE, INK, MUTED, ORANGE, empty_state, page_header, session

from dssc_discovery.dssc_model import (
    DSSCParams,
    SingleDiodeParams,
    absorption_coefficient,
    equations_curve,
    performance,
    short_circuit_current,
    single_diode_curve,
)
from dssc_discovery.dssc_runs import runs_frame, save_run

EQUATIONS = "DSSC equations (Eq. 1–4)"
SINGLE_DIODE = "Single-diode model"
SERIES_COLORS = [BLUE, ORANGE, AQUA, "#8a5cd6", "#c9a227", "#d1457f"]
DEFAULTS = DSSCParams()

page_header(
    "DSSC Simulator",
    "Enter UV–Vis absorbance, then calculate Voc, Jsc, fill factor and efficiency, and draw "
    "I–V and P–V curves for different light intensities and temperatures.",
    eyebrow="Modelling",
)

bad_inputs: list[str] = []


def number(label: str, default: float, key: str, help: str = "") -> float:
    """A scientific-notation friendly number box (type 2.2361e-3 or 0.0022361)."""
    text = st.text_input(label, value=f"{default:.6g}", key=key, help=help or None)
    try:
        value = float(text)
    except ValueError:
        value = float("nan")
    if not (math.isfinite(value) and value > 0):
        st.error("Enter a positive number.")
        bad_inputs.append(label)
        return default
    return value


# --- model parameters (shared by every tab) -------------------------------------------

PARAM_KEYS = ["dssc_k", "dssc_q", "dssc_L", "dssc_d", "dssc_m", "dssc_D", "dssc_n0", "dssc_phi"]
with st.expander("Model parameters (defaults are the values from the source document)"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        k = number("k, Boltzmann constant (J/K)", DEFAULTS.k, "dssc_k")
        D = number("D, diffusion coefficient (cm²/s)", DEFAULTS.D, "dssc_D")
    with c2:
        q = number("q, electron charge (C)", DEFAULTS.q, "dssc_q")
        n0 = number("n₀, electron density (cm⁻³)", DEFAULTS.n0, "dssc_n0")
    with c3:
        L = number("L, diffusion length (cm)", DEFAULTS.L, "dssc_L")
        m = number("m, ideality factor", DEFAULTS.m, "dssc_m")
    with c4:
        d = number(
            "d, film thickness = optical path length (cm)",
            DEFAULTS.d,
            "dssc_d",
            help="Also used as the path length in the Beer–Lambert law.",
        )
        phi = number(
            "φ, photon flux at 1000 W/m² (cm⁻² s⁻¹)",
            DEFAULTS.phi,
            "dssc_phi",
            help="Scaled in proportion to the irradiance you choose.",
        )
    st.caption("The absorption coefficient α is set in the UV–Vis tab. Temperature is set per tab.")
    if st.button("Reset to document defaults"):
        for key in PARAM_KEYS:
            st.session_state.pop(key, None)
        st.rerun()

tab_uv, tab_eq, tab_sd, tab_vary = st.tabs(
    [
        "1 · UV–Vis → α",
        "2 · DSSC equations",
        "3 · Single-diode model",
        "4 · Irradiation & temperature",
    ]
)

# --- tab 1: UV-Vis and the absorption coefficient -------------------------------------

uvvis_pairs: list[tuple[float, float]] = []
uvvis_wavelength: float | None = None
with tab_uv:
    st.markdown(
        "Enter the wavelength and absorbance from your UV–Vis result (type into the table, or "
        "upload a CSV with the columns `wavelength_nm` and `absorbance`). The absorption "
        "coefficient comes from the Beer–Lambert law and feeds Equation 1."
    )
    st.latex(r"\alpha = \frac{2.303\,A}{d}")
    upload = st.file_uploader("Upload UV–Vis CSV (optional)", type="csv")
    table = pd.DataFrame(
        {"wavelength_nm": pd.Series(dtype=float), "absorbance": pd.Series(dtype=float)}
    )
    if upload is not None:
        try:
            loaded = pd.read_csv(upload)
            loaded.columns = [str(c).strip().lower() for c in loaded.columns]
            table = loaded[["wavelength_nm", "absorbance"]]
        except (KeyError, ValueError, pd.errors.ParserError):
            st.error("The CSV needs two columns named wavelength_nm and absorbance.")
    edited = st.data_editor(
        table,
        num_rows="dynamic",
        use_container_width=True,
        key=f"uvvis_{upload.name if upload else 'typed'}",
        column_config={
            "wavelength_nm": st.column_config.NumberColumn("Wavelength (nm)", min_value=0.0),
            "absorbance": st.column_config.NumberColumn("Absorbance A", min_value=0.0),
        },
    )
    spectrum = edited.apply(pd.to_numeric, errors="coerce").dropna()
    spectrum = spectrum[(spectrum["wavelength_nm"] > 0) & (spectrum["absorbance"] >= 0)]
    spectrum = spectrum.sort_values("wavelength_nm")

    use_uvvis = False
    if spectrum.empty:
        st.info("No UV–Vis data yet, so the typed α below is used.")
    else:
        spectrum["alpha_cm1"] = absorption_coefficient(spectrum["absorbance"].to_numpy(), d)
        uvvis_pairs = list(
            zip(spectrum["wavelength_nm"].tolist(), spectrum["absorbance"].tolist(), strict=True)
        )
        peak = int(np.argmax(spectrum["absorbance"].to_numpy()))
        left, right = st.columns([3, 2])
        with right:
            st.dataframe(
                spectrum.rename(
                    columns={
                        "wavelength_nm": "λ (nm)",
                        "absorbance": "A",
                        "alpha_cm1": "α (1/cm)",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        with left:
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.plot(spectrum["wavelength_nm"], spectrum["absorbance"], marker="o", color=BLUE)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Absorbance")
            ax.set_title("UV–Vis absorbance")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        chosen = st.selectbox(
            "Wavelength used by the model (nm)",
            spectrum["wavelength_nm"].tolist(),
            index=peak,
            help="Defaults to the absorption peak.",
        )
        uvvis_wavelength = float(chosen)
        alpha_uvvis = float(spectrum.loc[spectrum["wavelength_nm"] == chosen, "alpha_cm1"].iloc[0])
        use_uvvis = (
            st.radio(
                "α used by the model",
                ["From the UV–Vis data", "A typed value"],
                horizontal=True,
            )
            == "From the UV–Vis data"
        )

    if use_uvvis:
        alpha = alpha_uvvis
        st.metric("α used by the model (cm⁻¹)", f"{alpha:,.1f}")
    else:
        uvvis_wavelength = None
        alpha = number("α, absorption coefficient (cm⁻¹)", DEFAULTS.alpha, "dssc_alpha")

if bad_inputs:
    st.stop()

params = DSSCParams(k=k, q=q, L=L, d=d, alpha=alpha, m=m, D=D, n0=n0, phi=phi)


# --- shared helpers -------------------------------------------------------------------


def combined_plot(v: np.ndarray, i: np.ndarray, perf, title: str) -> None:
    """I–V (left axis) and P–V (right axis) with Isc, Voc, Imp, Vmp and Pmax marked."""
    current, power = i * 1e3, v * i * 1e3
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(v, current, color=BLUE)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current density (mA/cm²)", color=BLUE, labelpad=8)
    ax.set_xlim(0, perf.voc * 1.08)
    ax.set_ylim(0, perf.jsc * 1.25)
    ax2 = ax.twinx()
    ax2.plot(v, power, color=ORANGE)
    ax2.set_ylabel("Power density (mW/cm²)", color=ORANGE, labelpad=8)
    ax2.set_ylim(0, perf.pmax * 1.45)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.axvline(perf.vmp, color=MUTED, linestyle=":", linewidth=1)
    ax.scatter([0, perf.voc, perf.vmp], [perf.jsc, 0, perf.imp], color=BLUE, zorder=5, s=28)
    ax2.scatter([perf.vmp], [perf.pmax], color=ORANGE, zorder=5, s=40)
    ax.annotate("Isc", (0, perf.jsc), (8, 8), textcoords="offset points", color=BLUE)
    ax.annotate("Voc", (perf.voc, 0), (6, 8), textcoords="offset points", color=INK)
    ax.text(
        0.98, 0.97,
        f"Isc = {perf.jsc:.2f} mA/cm²\n"
        f"Voc = {perf.voc:.3f} V\n"
        f"Imp = {perf.imp:.2f} mA/cm²\n"
        f"Vmp = {perf.vmp:.3f} V\n"
        f"Pmax = {perf.pmax:.2f} mW/cm²",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": MUTED},
    )  # fmt: skip
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_results(perf, area: float, irradiance: float) -> None:  # type: ignore[no-untyped-def]
    cols = st.columns(6)
    cols[0].metric("Voc (V)", f"{perf.voc:.3f}")
    cols[1].metric("Jsc (mA/cm²)", f"{perf.jsc:.2f}")
    cols[2].metric("Isc (mA)", f"{perf.jsc * area:.2f}")
    cols[3].metric("Fill factor", f"{perf.ff:.3f}")
    cols[4].metric("Pmax (mW/cm²)", f"{perf.pmax:.2f}")
    cols[5].metric("Efficiency η (%)", f"{perf.efficiency:.2f}")
    st.caption(
        f"FF = Pmax / (Voc × Isc); η = Jsc × Voc × FF / Pin × 100 with Pin = {perf.pin:g} mW/cm² "
        f"({irradiance:g} W/m²). Isc in mA assumes a cell area of {area:g} cm²."
    )


def save_panel(  # type: ignore[no-untyped-def]
    key: str, model: str, p: DSSCParams, irradiance: float, area: float, perf, sd=None
) -> None:
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    label = left.text_input("Label for this result (optional)", key=f"{key}_label")
    if right.button("Save to database", key=f"{key}_save", use_container_width=True):
        save_run(
            session(),
            operator=str(st.session_state.get("user", "owner")),
            label=label,
            model=model,
            params=p,
            irradiance=irradiance,
            area_cm2=area,
            perf=perf,
            single_diode=sd,
            uvvis=uvvis_pairs,
            uvvis_wavelength_nm=uvvis_wavelength,
        )
        st.success("Saved. It appears in “Saved simulations” below.")


def operating_point(prefix: str) -> tuple[float, float, float]:
    """Temperature, irradiance and cell area for one tab."""
    a, b, c = st.columns(3)
    t = a.slider("Temperature (K)", 273, 375, 300, key=f"{prefix}_T")
    g = b.number_input("Irradiance Pin (W/m²)", 50.0, 1500.0, 1000.0, step=10.0, key=f"{prefix}_G")
    area = c.number_input(
        "Cell area (cm²)", 0.01, 100.0, 1.0, step=0.1, key=f"{prefix}_A",
        help="Only used to turn Jsc (mA/cm²) into Isc (mA).",
    )  # fmt: skip
    return float(t), float(g), float(area)


# --- tab 2: the document's four equations ---------------------------------------------

with tab_eq:
    with st.expander("Equations used"):
        st.latex(
            r"I_{sc}=\frac{q\phi\alpha L}{1-L^2\alpha^2}\left[-L\alpha+\tanh\frac{d}{L}"
            r"+\frac{L\alpha\,e^{-d\alpha}}{\cosh(d/L)}\right]"
        )
        st.latex(r"V_{oc}=\frac{kTm}{q}\ln\left[\frac{L I_{sc}}{qDn_0\tanh(d/L)}+1\right]")
        st.latex(r"V=\frac{kTm}{q}\ln\left[\frac{L(I_{sc}-I)}{qDn_0\tanh(d/L)}+1\right]")
        st.latex(r"I=I_{sc}-\frac{qDn_0}{L}\tanh\frac{d}{L}\left[e^{qV/(kTm)}-1\right]")
    temp, irr, area = operating_point("eq")
    p_eq = replace(params, T=temp)
    try:
        v, i = equations_curve(p_eq, irr)
        perf = performance(v, i, irr)
    except ValueError as error:
        st.error(str(error))
    else:
        st.markdown(f"**α = {params.alpha:,.1f} cm⁻¹**, T = {temp:g} K, {irr:g} W/m²")
        show_results(perf, area, irr)
        combined_plot(v, i, perf, f"I–V and P–V (DSSC equations), T = {temp:g} K, {irr:g} W/m²")
        save_panel("eq", "equations", p_eq, irr, area, perf)

# --- tab 3: single-diode model --------------------------------------------------------

SD_DEFAULTS = SingleDiodeParams()
with tab_sd:
    st.latex(
        r"I = I_{ph} - I_0\left[\exp\left(\frac{q(V+IR_s)}{nkT}\right)-1\right]"
        r"-\frac{V+IR_s}{R_{sh}}"
    )
    st.caption(
        "The document gives no values for I₀, n, Rs or Rsh, so the defaults below are typical "
        "assumptions. Change them to fit your cell. Iph starts from Equation 1's Isc."
    )
    iph_default = short_circuit_current(params) * 1e3  # mA/cm^2
    a, b, c, e, f = st.columns(5)
    with a:
        iph = number("Iph at 1000 W/m² (mA/cm²)", iph_default, f"sd_iph_{iph_default:.4g}")
    with b:
        i0 = number("I₀ (A/cm²)", SD_DEFAULTS.i0, "sd_i0")
    with c:
        n_ideal = number("n, ideality factor", SD_DEFAULTS.n, "sd_n")
    with e:
        rs = number("Rs (Ω·cm²)", SD_DEFAULTS.rs, "sd_rs")
    with f:
        rsh = number("Rsh (Ω·cm²)", SD_DEFAULTS.rsh, "sd_rsh")
    sd = SingleDiodeParams(iph=iph * 1e-3, i0=i0, n=n_ideal, rs=rs, rsh=rsh)
    temp_sd, irr_sd, area_sd = operating_point("sd")
    p_sd = replace(params, T=temp_sd)
    if bad_inputs:
        st.stop()
    try:
        v, i = single_diode_curve(p_sd, sd, irr_sd)
        perf = performance(v, i, irr_sd)
    except ValueError as error:
        st.error(str(error))
    else:
        show_results(perf, area_sd, irr_sd)
        combined_plot(v, i, perf, f"I–V and P–V (single diode), T = {temp_sd:g} K, {irr_sd:g} W/m²")
        save_panel("sd", "single_diode", p_sd, irr_sd, area_sd, perf, sd)

# --- tab 4: vary irradiation and temperature ------------------------------------------


def parse_list(text: str) -> list[float] | None:
    try:
        values = [float(x) for x in text.replace(";", ",").split(",") if x.strip()]
    except ValueError:
        return None
    return values if values and all(x > 0 for x in values) and len(values) <= 6 else None


def curves_for(model: str, vary: str, values: list[float], fixed: float):  # type: ignore[no-untyped-def]
    """One (label, v, i, performance) per value; `vary` is 'G' or 'T'."""
    out = []
    for value in values:
        g, t = (value, fixed) if vary == "G" else (fixed, value)
        p = replace(params, T=t)
        v, i = equations_curve(p, g) if model == EQUATIONS else single_diode_curve(p, sd, g)
        label = f"{value:g} W/m²" if vary == "G" else f"{value:g} K"
        out.append((label, v, i, performance(v, i, g)))
    return out


def family_plots(curves, title: str) -> None:  # type: ignore[no-untyped-def]
    fig, (ax_iv, ax_pv) = plt.subplots(1, 2, figsize=(11, 4.2))
    for index, (label, v, i, perf) in enumerate(curves):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        ax_iv.plot(v, i * 1e3, color=color, label=label)
        ax_pv.plot(v, v * i * 1e3, color=color, label=label)
        ax_pv.scatter([perf.vmp], [perf.pmax], color=color, zorder=5, s=30)
    ax_iv.set_xlabel("Voltage (V)")
    ax_iv.set_ylabel("Current density (mA/cm²)")
    ax_iv.set_title(f"I–V: {title}")
    ax_iv.legend()
    ax_pv.set_xlabel("Voltage (V)")
    ax_pv.set_ylabel("Power density (mW/cm²)")
    ax_pv.set_title("P–V (dots = maximum power point)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def family_table(curves, vary: str) -> None:  # type: ignore[no-untyped-def]
    st.dataframe(
        pd.DataFrame(
            {
                "Irradiance" if vary == "G" else "Temperature": label,
                "Voc (V)": perf.voc,
                "Jsc (mA/cm²)": perf.jsc,
                "Vmp (V)": perf.vmp,
                "Imp (mA/cm²)": perf.imp,
                "Pmax (mW/cm²)": perf.pmax,
                "FF": perf.ff,
                "η (%)": perf.efficiency,
            }
            for label, _v, _i, perf in curves
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            name: st.column_config.NumberColumn(format="%.3f")
            for name in ("Voc (V)", "Vmp (V)", "FF")
        }
        | {
            name: st.column_config.NumberColumn(format="%.2f")
            for name in ("Jsc (mA/cm²)", "Imp (mA/cm²)", "Pmax (mW/cm²)", "η (%)")
        },
    )


with tab_vary:
    model_choice = st.radio("Model", [EQUATIONS, SINGLE_DIODE], horizontal=True)

    st.subheader("Different irradiation")
    a, b = st.columns(2)
    t_fixed = a.slider("Temperature (K)", 273, 375, 300, key="vary_T")
    g_text = b.text_input("Irradiation levels (W/m²), comma separated", "320, 640, 1000")
    levels = parse_list(g_text)
    if levels is None:
        st.error("Enter up to 6 positive numbers, e.g. 320, 640, 1000.")
    else:
        try:
            curves = curves_for(model_choice, "G", levels, float(t_fixed))
        except ValueError as error:
            st.error(str(error))
        else:
            family_plots(curves, f"{t_fixed} K")
            family_table(curves, "G")

    st.subheader("Different temperature")
    a, b = st.columns(2)
    g_fixed = a.number_input("Irradiance (W/m²)", 50.0, 1500.0, 1000.0, step=10.0, key="vary_G")
    t_text = b.text_input("Temperatures (K), comma separated", "273, 300, 325, 350, 375")
    temps = parse_list(t_text)
    if temps is None:
        st.error("Enter up to 6 positive temperatures, e.g. 273, 300, 375.")
    else:
        if any(t < 273 or t > 375 for t in temps):
            st.warning("The source document covers 273–375 K; values outside are extrapolated.")
        try:
            curves = curves_for(model_choice, "T", temps, float(g_fixed))
        except ValueError as error:
            st.error(str(error))
        else:
            family_plots(curves, f"{g_fixed:g} W/m²")
            family_table(curves, "T")
            if model_choice == EQUATIONS:
                st.caption(
                    "In these equations temperature only enters through kTm/q, so Voc rises "
                    "slightly with T while Isc stays the same."
                )

# --- saved runs -----------------------------------------------------------------------

st.subheader("Saved simulations")
saved = runs_frame(session())
if saved.empty:
    empty_state("Nothing saved yet.", "Use “Save to database” under a result to keep it here.")
else:
    st.dataframe(saved, hide_index=True, use_container_width=True)
