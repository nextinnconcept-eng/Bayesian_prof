"""Shared Streamlit plumbing: DB session, login state, page chrome, table
formatting and chart styling. Layout only -- all queries live in
dssc_discovery.reports.ui_queries."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import altair as alt
import matplotlib as mpl
import pandas as pd
import sqlalchemy as sa
import streamlit as st
from sqlalchemy.orm import Session

from dssc_discovery.database.session import get_session_factory

# --- demo login --------------------------------------------------------------
# School-project demo credentials. This is NOT real security: anyone with the
# source can read them. Replace with a proper auth provider before any real use.
DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin"

# --- palette (validated reference instance, light mode) ----------------------
BLUE = "#2a78d6"  # series 1
ORANGE = "#eb6834"  # series 2
AQUA = "#1baf7a"  # series 3
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"

SYNTHETIC_BADGE = "🧪 Synthetic"
REAL_BADGE = "Real"

STATE_LABELS = {
    "proposed": "🟡 Proposed",
    "approved": "🔵 Approved",
    "fabricated": "🔵 Fabricated",
    "measured": "🔵 Measured",
    "validated": "🔵 Validated",
    "ingested": "🟢 Ingested",
    "model_updated": "🟢 Model updated",
    "rejected": "⚪ Rejected",
    "failed_fabrication": "🔴 Fabrication failed",
    "failed_measurement": "🔴 Measurement failed",
    "superseded": "⚪ Superseded",
}

STAGE_LABELS = {
    "initial_doe": "Initial design",
    "bo_cycle_1": "BO cycle 1",
    "bo_cycle_2": "BO cycle 2",
    "bo_cycle_3": "BO cycle 3",
    "manual": "Manual",
}

COLUMN_LABELS = {
    "condition_code": "Condition",
    "device_code": "Device",
    "species_code": "Species",
    "design_source": "Stage",
    "state": "Status",
    "solvent": "Solvent",
    "solvent_ratio": "Solvent ratio",
    "ph": "pH",
    "extraction_temperature_c": "Extraction T (°C)",
    "extraction_time_min": "Extraction time (min)",
    "pigment_concentration_mm": "Pigment conc. (mM)",
    "adsorption_time_min": "Adsorption time (min)",
    "sensitization_concentration_mm": "Dye bath conc. (mM)",
    "sensitization_temperature_c": "Dye bath T (°C)",
    "tio2_thickness_um": "TiO₂ thickness (µm)",
    "sintering_temperature_c": "Sintering T (°C)",
    "sintering_time_min": "Sintering time (min)",
    "surface_treatment": "Surface treatment",
    "power_conversion_efficiency_percent": "PCE (%)",
    "open_circuit_voltage_v": "Voc (V)",
    "short_circuit_current_density_ma_cm2": "Jsc (mA/cm²)",
    "fill_factor": "Fill factor",
    "replicate_number": "Replicate",
    "active_area_cm2": "Area (cm²)",
    "measurement_date": "Measured on",
    "retention_percent": "Retention (%)",
    "ageing_duration_h": "Ageing (h)",
    "cycle_number": "Cycle",
    "objective": "Objective",
    "rank": "Rank",
    "predicted_objective": "Predicted PCE (%)",
    "predictive_uncertainty": "Uncertainty (±%)",
    "predicted_retention": "Predicted retention (%)",
    "acquisition_value": "Acquisition score",
    "rationale_text": "Why this experiment",
    "model_name": "Model",
    "model": "Model",
    "strategy": "Validation",
    "model_family": "Family",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "observation": "Observation",
    "candidate_mechanism": "Candidate mechanism",
    "confound_check": "Confound check",
    "proposed_confirmatory_experiment": "Confirmatory experiment",
    "status": "Status",
    "created_at": "Created",
    "operator": "By",
    "from_state": "From",
    "to_state": "To",
    "reason": "Note",
    "data": "Data",
    "n_devices": "Devices",
    "mean_pce": "Mean PCE (%)",
    "best_pce": "Best PCE (%)",
}

HIDDEN_SUFFIXES = ("_id", "_json")
HIDDEN_COLUMNS = {"id", "experiment_version", "notes", "is_synthetic", "interpretation_caveat"}


# --- session -----------------------------------------------------------------


def session() -> Session:
    if "db_session" not in st.session_state:
        st.session_state["db_session"] = get_session_factory()()
    return st.session_state["db_session"]


@st.cache_resource(show_spinner="Preparing the database…")
def _migrate(database_url: str) -> None:
    """Bring the database schema up to date once per database per server
    process, so a fresh install works without any manual setup step."""
    from alembic.config import Config

    from alembic import command
    from dssc_discovery.config import REPO_ROOT

    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)

    engine = sa.create_engine(database_url)
    try:
        tables = set(sa.inspect(engine).get_table_names())
    finally:
        engine.dispose()
    if tables and "alembic_version" not in tables:
        # Built directly from the ORM models (tests, demo scripts): the schema
        # already matches head, so record that instead of re-creating tables.
        command.stamp(config, "head")
    else:
        command.upgrade(config, "head")


def ensure_database() -> None:
    from dssc_discovery.config import get_config
    from dssc_discovery.database.session import resolve_database_url

    _migrate(resolve_database_url(get_config().database_url))


def is_logged_in() -> bool:
    return st.session_state.get("user") is not None


def check_credentials(username: str, password: str) -> bool:
    return username.strip() == DEMO_USERNAME and password == DEMO_PASSWORD


# --- page chrome --------------------------------------------------------------

_CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1200px; }}
h1 {{ font-weight: 700; letter-spacing: -0.02em; }}
.page-subtitle {{ color: {INK_2}; font-size: 1.02rem; margin: -0.6rem 0 1.4rem 0; max-width: 820px; }}
.eyebrow {{ color: {BLUE}; font-weight: 600; font-size: 0.78rem; letter-spacing: 0.08em;
           text-transform: uppercase; margin-bottom: 0.2rem; }}
div[data-testid="stMetric"] {{ background: {SURFACE}; border: 1px solid {GRID};
    border-radius: 10px; padding: 0.8rem 1rem; }}
div[data-testid="stMetricLabel"] p {{ color: {INK_2}; font-size: 0.85rem; }}
.card {{ background: {SURFACE}; border: 1px solid {GRID}; border-radius: 10px;
         padding: 1rem 1.1rem; height: 100%; }}
.card h4 {{ margin: 0 0 0.3rem 0; font-size: 1rem; }}
.card p {{ color: {INK_2}; margin: 0; font-size: 0.9rem; }}
.step-number {{ color: {BLUE}; font-weight: 700; font-size: 0.8rem; }}
</style>
"""


def apply_style() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE, AQUA]),
        }
    )


def page_header(title: str, subtitle: str, eyebrow: str = "") -> None:
    if eyebrow:
        st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def card(title: str, body: str, step: str = "") -> str:
    step_html = f'<div class="step-number">{step}</div>' if step else ""
    return f'<div class="card">{step_html}<h4>{title}</h4><p>{body}</p></div>'


def cards(items: Sequence[tuple[str, str]], numbered: bool = False) -> None:
    columns = st.columns(len(items))
    for i, (column, (title, body)) in enumerate(zip(columns, items, strict=True), start=1):
        column.markdown(card(title, body, f"STEP {i}" if numbered else ""), unsafe_allow_html=True)


def stage_chart(
    stages: Sequence[str], values: Sequence[float], y_title: str, height: int = 280
) -> None:
    """Best-so-far line across programme stages, in programme order (not
    alphabetical), with point markers and a hover tooltip."""
    frame = pd.DataFrame({"Stage": [STAGE_LABELS.get(s, s) for s in stages], y_title: values})
    order = list(frame["Stage"])
    chart = (
        alt.Chart(frame)
        .mark_line(color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(color=BLUE, size=70))
        .encode(
            x=alt.X("Stage:N", sort=order, axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y(f"{y_title}:Q", scale=alt.Scale(zero=False), axis=alt.Axis(title=y_title)),
            tooltip=["Stage", alt.Tooltip(f"{y_title}:Q", format=".2f")],
        )
        .properties(height=height)
        .configure_axis(gridColor=GRID, domainColor=AXIS, labelColor=MUTED, titleColor=INK_2)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)


def empty_state(message: str, hint: str = "") -> None:
    st.info(message + (f"\n\n{hint}" if hint else ""), icon="ℹ️")


# --- tables -------------------------------------------------------------------


def readable(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Friendly labels, status/stage names and a data badge; hides IDs and
    JSON blobs unless columns are listed explicitly."""
    if frame.empty:
        return frame
    frame = frame.copy()
    if "is_synthetic" in frame.columns:
        frame.insert(
            0, "data", frame["is_synthetic"].map({True: SYNTHETIC_BADGE, False: REAL_BADGE})
        )
    for column in ("state", "to_state", "from_state"):
        if column in frame.columns:
            frame[column] = frame[column].map(
                lambda v: "—" if v is None or pd.isna(v) else STATE_LABELS.get(v, v)
            )
    if "design_source" in frame.columns:
        frame["design_source"] = frame["design_source"].map(lambda v: STAGE_LABELS.get(v, v))
    if "created_at" in frame.columns:
        frame["created_at"] = pd.to_datetime(frame["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
    if columns is not None:
        keep = [c for c in columns if c in frame.columns]
    else:
        keep = [
            c for c in frame.columns if c not in HIDDEN_COLUMNS and not c.endswith(HIDDEN_SUFFIXES)
        ]
    keep = [c for c in keep if not frame[c].isna().all()]  # drop columns with no values at all
    return frame[keep].rename(columns=COLUMN_LABELS)


def show_table(
    frame: pd.DataFrame,
    columns: Iterable[str] | None = None,
    empty_message: str = "Nothing recorded yet.",
    height: int | None = None,
    key: str | None = None,
) -> None:
    if frame.empty:
        empty_state(empty_message)
        return
    st.dataframe(
        readable(frame, columns),
        use_container_width=True,
        hide_index=True,
        height=height if height is not None else "auto",
        column_config={
            label: st.column_config.NumberColumn(format="%.2f")
            for label in (
                "PCE (%)",
                "Predicted PCE (%)",
                "Uncertainty (±%)",
                "Mean PCE (%)",
                "Best PCE (%)",
                "Retention (%)",
                "Predicted retention (%)",
                "MAE",
                "RMSE",
                "R²",
            )
        }
        | {"Why this experiment": st.column_config.TextColumn(width="large")},
        key=key,
    )
    with st.expander("Show all columns (technical view)"):
        st.dataframe(frame, use_container_width=True, hide_index=True)
