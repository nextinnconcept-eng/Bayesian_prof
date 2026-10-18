"""'Start here' and 'Plan my experiments' against an empty database: the
owner can type a result in, gets plain-language feedback, never touches demo
data, and is offered a starting set when there is nothing to learn from yet."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

from dssc_discovery.config import clear_config_cache
from dssc_discovery.database.models import Base
from dssc_discovery.database.repositories import (
    ExperimentalConditionRepository,
    PhotovoltaicMeasurementRepository,
)
from dssc_discovery.database.session import reset_engine_cache
from dssc_discovery.domain.enums import ClosedLoopState

APP_DIR = Path(__file__).resolve().parents[2] / "ui" / "streamlit_app"
START = APP_DIR / "views" / "start.py"
PLAN = APP_DIR / "views" / "plan.py"

TYPED = {
    "solvent_ratio": "0.7",
    "ph": "5.5",
    "extraction_temperature_c": "25",
    "extraction_time_min": "60",
    "pigment_concentration_mm": "10",
    "adsorption_time_min": "120",
    "sensitization_concentration_mm": "0.5",
    "sensitization_temperature_c": "25",
    "tio2_thickness_um": "12",
    "sintering_temperature_c": "450",
    "sintering_time_min": "30",
    "active_area_cm2": "0.25",
    "open_circuit_voltage_v": "0.6",
    "short_circuit_current_density_ma_cm2": "8",
    "fill_factor": "0.6",
}
CHOSEN = {
    "solvent": "ethanol",
    "surface_treatment": "none",
    "electrolyte_type": "iodide_triiodide",
    "counter_electrode_type": "platinum",
    "illumination_protocol": "am1.5g_1_sun",
}


@pytest.fixture
def empty_database(tmp_path):
    url = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
    engine = sa.create_engine(url)
    Base.metadata.create_all(engine)
    engine.dispose()
    mp = pytest.MonkeyPatch()
    mp.setenv("DSSC_DATABASE_URL", url)
    mp.setenv("DSSC_DEMO_DATABASE_URL", f"sqlite:///{(tmp_path / 'demo.db').as_posix()}")
    mp.syspath_prepend(str(APP_DIR))
    clear_config_cache()
    reset_engine_cache()
    yield url
    mp.undo()
    clear_config_cache()
    reset_engine_cache()
    for name in ("_shared", "_user_data_ui"):
        sys.modules.pop(name, None)


def _open(path: Path) -> AppTest:
    app = AppTest.from_file(str(path), default_timeout=180)
    app.session_state["user"] = "admin"
    app.run()
    assert not app.exception, [e.message for e in app.exception]
    return app


def _fill_form(app: AppTest, version: int = 0, *, drop: str | None = None) -> None:
    for name, value in TYPED.items():
        if name != drop:
            app.text_input(key=f"f_{version}_{name}").input(value)
    for name, value in CHOSEN.items():
        app.selectbox(key=f"f_{version}_{name}").select(value)
    app.date_input(key=f"f_{version}_measurement_date").set_value("2026-10-01")


def _saved(url: str) -> tuple[list, int]:
    engine = sa.create_engine(url)
    with Session(engine) as session:
        conditions = ExperimentalConditionRepository(session).list()
        measurements = len(PhotovoltaicMeasurementRepository(session).list())
    engine.dispose()
    return conditions, measurements


def test_start_page_explains_itself_on_an_empty_database(empty_database):
    app = _open(START)
    assert any("Start here" == t.value for t in app.title)
    assert any("No real results" in i.value for i in app.info)


def test_typed_result_is_saved_as_real_ingested_data(empty_database):
    app = _open(START)
    app.radio[0].set_value("From my own earlier lab runs").run()
    _fill_form(app)
    next(b for b in app.button if b.label == "Save this result").click().run()
    assert not app.exception, [e.message for e in app.exception]
    conditions, measurements = _saved(empty_database)
    assert len(conditions) == 1 and measurements == 1
    assert conditions[0].state == ClosedLoopState.INGESTED and not conditions[0].is_synthetic
    assert conditions[0].operator == "admin"


def test_missing_value_is_explained_and_nothing_is_saved(empty_database):
    app = _open(START)
    _fill_form(app, drop="ph")
    next(b for b in app.button if b.label == "Save this result").click().run()
    assert any("pH of the dye extract" in m.value and "missing" in m.value for m in app.markdown)
    assert _saved(empty_database) == ([], 0)


def test_nothing_yet_points_to_planning(empty_database):
    app = _open(START)
    app.radio[0].set_value("Nothing yet").run()
    assert any("Plan my experiments" in i.value for i in app.info)


def test_plan_page_offers_a_starting_set_and_creates_it(empty_database):
    app = _open(PLAN)
    button = next(b for b in app.button if b.label == "Create my starting experiments")
    app.number_input[0].set_value(5)
    button.click().run()
    assert not app.exception, [e.message for e in app.exception]
    conditions, _ = _saved(empty_database)
    assert len(conditions) == 5
    assert all(c.state == ClosedLoopState.PROPOSED and not c.is_synthetic for c in conditions)
    assert any("Recipes waiting to be run (5)" in m.value for m in app.markdown)


def test_try_my_recipe_is_locked_until_there_is_data(empty_database):
    app = _open(PLAN)
    assert any("Available once you have" in i.value for i in app.info)


def test_demo_mode_blocks_entering_data(empty_database):
    app = AppTest.from_file(str(START), default_timeout=180)
    app.session_state["user"] = "admin"
    app.session_state["data_mode"] = "demo"
    app.run()
    assert not app.exception, [e.message for e in app.exception]
    assert any("simulated" in i.value.lower() for i in app.info)
    assert not any(b.label == "Save this result" for b in app.button)
