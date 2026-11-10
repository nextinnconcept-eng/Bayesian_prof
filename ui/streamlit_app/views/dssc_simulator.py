"""DSSC Simulator: UV-Vis absorption, the document's DSSC equations, the single-diode
model, and I-V / P-V curves under different irradiation and temperature."""

from __future__ import annotations

import math
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from _shared import AQUA, BLUE, INK, MUTED, ORANGE, cards, empty_state, page_header, session

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
LIGHT_LEVELS = [100, 200, 320, 500, 640, 800, 1000, 1200]
TEMPERATURES = [273, 285, 300, 325, 350, 375]
DEFAULTS = DSSCParams()

page_header(
    "DSSC Simulator",
    "Predict how a dye-sensitized solar cell performs from its UV–Vis absorbance: open-circuit "
    "voltage, short-circuit current, fill factor and efficiency, with I–V and P–V curves. "
    "These are calculated (modelled) values, not lab measurements.",
    eyebrow="Modelling",
)
cards(
    [
        ("Enter UV–Vis", "Type or upload wavelength and absorbance. The app turns it into α."),
        ("Set conditions", "Choose temperature, light intensity and cell area."),
        ("Read the results", "Voc, Jsc, fill factor, efficiency and the I–V / P–V curves."),
        ("Compare and save", "See how light and temperature change the cell, and save a run."),
    ],
    numbered=True,
)
st.write("")

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


# --- operating conditions (shared by tabs 2-4) -----------------------------------------

with st.container(border=True):
    st.markdown("**Conditions** — used by tabs 2, 3 and 4")
    ca, cb, cc = st.columns(3)
    temp = float(
        ca.slider(
            "Temperature (K)",
            273,
            375,
            300,
            key="cond_T",
            help="The source document covers 273–375 K.",
        )
    )
    irr = float(
        cb.number_input(
            "Light intensity Pin (W/m²)",
            50.0,
            1500.0,
            1000.0,
            step=10.0,
            key="cond_G",
            help="1000 W/m² = 100 mW/cm² = one sun.",
        )
    )
    area = float(
        cc.number_input(
            "Cell area (cm²)",
            0.01,
            100.0,
            1.0,
            step=0.1,
            key="cond_A",
            help="Only used to turn Jsc (mA/cm²) into Isc (mA).",
        )
    )

# --- model parameters (shared by every tab) -------------------------------------------

PARAM_KEYS = ["dssc_k", "dssc_q", "dssc_L", "dssc_d", "dssc_m", "dssc_D", "dssc_n0", "dssc_phi"]
with st.expander(
    "⚙️ Advanced: model parameters (defaults are the values from the source document)"
):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Material and geometry**")
        L = number("L, electron diffusion length (cm)", DEFAULTS.L, "dssc_L")
        d = number(
            "d, TiO₂ film thickness = optical path length (cm)",
            DEFAULTS.d,
            "dssc_d",
            help="Also used as the path length in the Beer–Lambert law.",
        )
        D = number("D, diffusion coefficient (cm²/s)", DEFAULTS.D, "dssc_D")
        n0 = number("n₀, electron concentration (cm⁻³)", DEFAULTS.n0, "dssc_n0")
    with c2:
        st.markdown("**Physics**")
        k = number("k, Boltzmann constant (J/K)", DEFAULTS.k, "dssc_k")
        q = number("q, electron charge (C)", DEFAULTS.q, "dssc_q")
        m = number("m, ideality factor", DEFAULTS.m, "dssc_m")
    with c3:
        st.markdown("**Light**")
        phi = number(
            "φ, photon flux at 1000 W/m² (cm⁻² s⁻¹)",
            DEFAULTS.phi,
            "dssc_phi",
            help="Scaled in proportion to the light intensity you choose.",
        )
        st.caption(
            "Type numbers like 2.2361e-3 or 0.0022361. The absorption coefficient α is set in tab 1."
        )
        if st.button("Reset to document defaults"):
            for key in PARAM_KEYS:
                st.session_state.pop(key, None)
            st.rerun()

edited_params = any(
    not math.isclose(value, default, rel_tol=1e-9)
    for value, default in (
        (k, DEFAULTS.k),
        (q, DEFAULTS.q),
        (L, DEFAULTS.L),
        (d, DEFAULTS.d),
        (m, DEFAULTS.m),
        (D, DEFAULTS.D),
        (n0, DEFAULTS.n0),
        (phi, DEFAULTS.phi),
    )
)
status = st.empty()

tab_uv, tab_eq, tab_sd, tab_vary = st.tabs(
    [
        "1 · UV–Vis → α",
        "2 · Cell performance",
        "3 · Single-diode model",
        "4 · Light & temperature",
    ]
)

# --- tab 1: UV-Vis and the absorption coefficient -------------------------------------

uvvis_pairs: list[tuple[float, float]] = []
uvvis_wavelength: float | None = None
with tab_uv:
    st.markdown(
        "**What this does:** turns your UV–Vis absorbance into the absorption coefficient α with "
        "the Beer–Lambert law. α then feeds Equation 1 (the short-circuit current)."
    )
    st.latex(r"\alpha = \frac{2.303\,A}{d}")
    up_col, tpl_col = st.columns([4, 1], vertical_alignment="bottom")
    upload = up_col.file_uploader("Upload a UV–Vis CSV (optional)", type="csv")
    tpl_col.download_button(
        "Download CSV template",
        "wavelength_nm,absorbance\n",
        file_name="uvvis_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(
        "Or type directly into the table below: click the empty row at the bottom to add a row."
    )
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
        st.info(
            "No UV–Vis data yet, so the typed α below is used "
            f"(the document's value is {DEFAULTS.alpha:g} cm⁻¹)."
        )
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
        alpha_source = f"from UV–Vis at {uvvis_wavelength:g} nm"
        st.success(f"The model will use α = {alpha:,.1f} cm⁻¹ ({alpha_source}).")
    else:
        uvvis_wavelength = None
        alpha = number("α, absorption coefficient (cm⁻¹)", DEFAULTS.alpha, "dssc_alpha")
        alpha_source = "typed value"

if bad_inputs:
    st.stop()

params = DSSCParams(k=k, q=q, L=L, d=d, alpha=alpha, m=m, D=D, n0=n0, phi=phi)
status.markdown(
    f"**In use:** α = {alpha:,.0f} cm⁻¹ ({alpha_source}) · parameters: "
    f"{'edited by you' if edited_params else 'document defaults'} · "
    f"{temp:g} K · {irr:g} W/m² · {area:g} cm²"
)


# --- shared helpers -------------------------------------------------------------------


def combined_plot(v: np.ndarray, i: np.ndarray, perf, title: str) -> None:  # type: ignore[no-untyped-def]
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
    with st.expander("What do these numbers mean?"):
        st.markdown(
            "- **Voc** (open-circuit voltage): the voltage across the cell when no current flows.\n"
            "- **Jsc** (short-circuit current density): the current per cm² when the cell is "
            "shorted. **Isc** is the same current for the whole cell area.\n"
            "- **Pmax**: the most power the cell can deliver, at the *maximum power point* "
            "(**Vmp**, **Imp**).\n"
            "- **Fill factor (FF)**: how “square” the I–V curve is: Pmax / (Voc × Jsc). "
            "Closer to 1 is better.\n"
            "- **Efficiency η**: the share of the incoming light power turned into electricity."
        )


def save_panel(  # type: ignore[no-untyped-def]
    key: str, model: str, p: DSSCParams, irradiance: float, area: float, perf, sd=None
) -> None:
    with st.container(border=True):
        st.markdown("**Save this result to the database**")
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
            st.success("Saved. It appears in “Saved simulations” at the bottom of the page.")


# --- tab 2: the document's four equations ---------------------------------------------

with tab_eq:
    st.markdown(
        "**What this does:** uses the four DSSC equations to calculate the cell's short-circuit "
        "current, open-circuit voltage and full I–V curve at the conditions above."
    )
    with st.expander("Show the equations"):
        st.latex(
            r"I_{sc}=\frac{q\phi\alpha L}{1-L^2\alpha^2}\left[-L\alpha+\tanh\frac{d}{L}"
            r"+\frac{L\alpha\,e^{-d\alpha}}{\cosh(d/L)}\right]"
        )
        st.latex(r"V_{oc}=\frac{kTm}{q}\ln\left[\frac{L I_{sc}}{qDn_0\tanh(d/L)}+1\right]")
        st.latex(r"V=\frac{kTm}{q}\ln\left[\frac{L(I_{sc}-I)}{qDn_0\tanh(d/L)}+1\right]")
        st.latex(r"I=I_{sc}-\frac{qDn_0}{L}\tanh\frac{d}{L}\left[e^{qV/(kTm)}-1\right]")
    p_eq = replace(params, T=temp)
    try:
        v, i = equations_curve(p_eq, irr)
        perf = performance(v, i, irr)
    except ValueError as error:
        st.error(str(error))
    else:
        show_results(perf, area, irr)
        combined_plot(v, i, perf, f"I–V and P–V (DSSC equations), T = {temp:g} K, {irr:g} W/m²")
        save_panel("eq", "equations", p_eq, irr, area, perf)

# --- tab 3: single-diode model --------------------------------------------------------

SD_DEFAULTS = SingleDiodeParams()
with tab_sd:
    st.markdown(
        "**What this does:** models the cell as a current source with a diode, a series "
        "resistance Rs and a shunt resistance Rsh, and solves the equation below numerically."
    )
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
    if bad_inputs:
        st.stop()
    try:
        v, i = single_diode_curve(p_eq, sd, irr)
        perf = performance(v, i, irr)
    except ValueError as error:
        st.error(str(error))
    else:
        show_results(perf, area, irr)
        combined_plot(v, i, perf, f"I–V and P–V (single diode), T = {temp:g} K, {irr:g} W/m²")
        save_panel("sd", "single_diode", p_eq, irr, area, perf, sd)

# --- tab 4: vary irradiation and temperature ------------------------------------------


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
                "Light intensity" if vary == "G" else "Temperature": label,
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
    st.markdown(
        "**What this does:** draws several I–V and P–V curves on one graph so you can see how "
        "light intensity and temperature change the cell. The other condition comes from the "
        "**Conditions** box above."
    )
    model_choice = st.radio("Model", [EQUATIONS, SINGLE_DIODE], horizontal=True)

    st.subheader(f"Different light intensities (at {temp:g} K)")
    levels = st.multiselect(
        "Light intensities to compare (W/m²)",
        LIGHT_LEVELS,
        default=[320, 640, 1000],
        max_selections=6,
        key="vary_levels",
    )
    if not levels:
        st.info("Pick at least one light intensity.")
    else:
        try:
            curves = curves_for(model_choice, "G", sorted(levels), temp)
        except ValueError as error:
            st.error(str(error))
        else:
            family_plots(curves, f"{temp:g} K")
            family_table(curves, "G")

    st.subheader(f"Different temperatures (at {irr:g} W/m²)")
    temps = st.multiselect(
        "Temperatures to compare (K)",
        TEMPERATURES,
        default=[273, 300, 325, 350, 375],
        max_selections=6,
        key="vary_temps",
    )
    if not temps:
        st.info("Pick at least one temperature.")
    else:
        try:
            curves = curves_for(model_choice, "T", sorted(temps), irr)
        except ValueError as error:
            st.error(str(error))
        else:
            family_plots(curves, f"{irr:g} W/m²")
            family_table(curves, "T")
            st.caption(
                "In these equations temperature only enters through kTm/q (kT/q in the diode "
                "term), so Voc rises with T while the photocurrent stays the same."
            )

# --- saved runs -----------------------------------------------------------------------

saved = runs_frame(session())
st.subheader(f"Saved simulations ({len(saved)})")
if saved.empty:
    empty_state("Nothing saved yet.", "Use “Save to database” under a result to keep it here.")
else:
    st.dataframe(saved, hide_index=True, use_container_width=True)
