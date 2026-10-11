"""M11: every dashboard page renders without error against a populated
temporary database, synthetic rows are badged, the explainability page
shows the caveat, and an approval click goes through the state machine
with the operator's name."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

from dssc_discovery.closed_loop import state_machine as sm
from dssc_discovery.closed_loop.orchestrator import propose_conditions
from dssc_discovery.config import clear_config_cache
from dssc_discovery.database.models import Base
from dssc_discovery.database.repositories import ExperimentalConditionRepository
from dssc_discovery.database.session import reset_engine_cache
from dssc_discovery.domain.enums import ClosedLoopState, DesignSource, SpeciesCode
from dssc_discovery.features.synthetic import generate_synthetic_dataset
from dssc_discovery.optimization.initial_design import latin_hypercube_design
from dssc_discovery.optimization.search_space import search_space_for

APP_DIR = Path(__file__).resolve().parents[2] / "ui" / "streamlit_app"
PAGES = sorted((APP_DIR / "pages").glob("*.py"))


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("ui") / "ui.db"
    url = f"sqlite:///{db_path.as_posix()}"
    engine = sa.create_engine(url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        generate_synthetic_dataset(session, n=12, seed=1)
        space = search_space_for(SpeciesCode.BETA_VULGARIS)
        propose_conditions(
            session,
            SpeciesCode.BETA_VULGARIS,
            latin_hypercube_design(space, 2, seed=1),
            design_source=DesignSource.INITIAL_DOE,
            code_prefix="UI-DOE",
            operator="tester",
            is_synthetic=True,
        )
        session.commit()
    engine.dispose()

    mp = pytest.MonkeyPatch()
    mp.setenv("DSSC_DATABASE_URL", url)
    mp.syspath_prepend(str(APP_DIR))
    clear_config_cache()
    reset_engine_cache()
    yield url
    mp.undo()
    clear_config_cache()
    reset_engine_cache()
    sys.modules.pop("_shared", None)


def _run(path: Path) -> AppTest:
    app = AppTest.from_file(str(path), default_timeout=120)
    app.run()
    assert not app.exception, [e.message for e in app.exception]
    return app


def test_home_page_renders(database):
    app = _run(APP_DIR / "Home.py")
    assert app.title[0].value == "Natural-pigment DSSC discovery"


@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_every_page_renders(database, page):
    _run(page)


def test_experiments_page_badges_synthetic_rows(database):
    app = _run(APP_DIR / "pages" / "1_experiments.py")
    table = app.dataframe[0].value
    assert (table["data"] == "🧪 SYNTHETIC").all()


def test_explainability_page_shows_caveat_and_explains(database):
    app = _run(APP_DIR / "pages" / "5_explainability.py")
    assert "not causal" in app.warning[0].value
    app.button[0].click().run()
    assert not app.exception


def test_approval_click_goes_through_the_state_machine(database):
    app = _run(APP_DIR / "pages" / "7_closed_loop.py")
    app.text_input[0].input("Dr UI").run()
    app.text_input[1].input("looks good").run()
    app.button(key=next(b.key for b in app.button if b.key.startswith("approve-"))).click().run()
    assert not app.exception

    engine = sa.create_engine(database)
    with Session(engine) as session:
        approved = ExperimentalConditionRepository(session).filter(state=ClosedLoopState.APPROVED)
        assert len(approved) == 1
        assert sm.history(session, approved[0].id)[-1].operator == "Dr UI"
    engine.dispose()
