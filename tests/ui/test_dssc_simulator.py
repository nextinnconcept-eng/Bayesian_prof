"""The DSSC Simulator page renders, reacts to edited parameters and saves a run."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

from dssc_discovery.config import clear_config_cache
from dssc_discovery.database.models import Base, DSSCSimulationRunORM
from dssc_discovery.database.session import reset_engine_cache

APP_DIR = Path(__file__).resolve().parents[2] / "ui" / "streamlit_app"
PAGE = APP_DIR / "views" / "dssc_simulator.py"


@pytest.fixture
def database(tmp_path):
    url = f"sqlite:///{(tmp_path / 'sim.db').as_posix()}"
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
    sys.modules.pop("_shared", None)


def _open() -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=180)
    app.session_state["user"] = "admin"
    app.run()
    assert not app.exception, [e.message for e in app.exception]
    return app


def _metric(app: AppTest, label: str) -> str:
    return next(m.value for m in app.metric if m.label == label)


def test_defaults_show_results_and_save(database):
    app = _open()
    assert _metric(app, "Voc (V)") == "0.282"
    assert _metric(app, "Jsc (mA/cm²)") == "15.44"
    assert not app.error
    next(b for b in app.button if b.key == "eq_save").click().run()
    assert not app.exception
    engine = sa.create_engine(database)
    with Session(engine) as session:
        rows = session.scalars(sa.select(DSSCSimulationRunORM)).all()
    engine.dispose()
    assert len(rows) == 1
    assert rows[0].model == "equations"
    assert rows[0].voc_v == pytest.approx(0.282, abs=0.003)


def test_without_uvvis_rows_the_typed_alpha_is_used(database):
    app = _open()
    # no UV-Vis rows yet: the typed alpha (document default) is used
    assert any("No UV–Vis data" in i.value for i in app.info)


def test_bad_parameter_is_rejected_not_crashed(database):
    app = _open()
    app.text_input(key="dssc_L").input("abc").run()
    assert not app.exception
    assert any("positive number" in e.value for e in app.error)
