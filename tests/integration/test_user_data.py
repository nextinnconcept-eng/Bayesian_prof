"""Owner-entered data: field guide, upload/form import, readiness levels,
early-estimate planning and the 'try my recipe' prediction, on an in-memory
database with real (non-synthetic) rows."""

from __future__ import annotations

import pytest

from dssc_discovery.closed_loop.real_workflow import InsufficientRealDataError
from dssc_discovery.config import clear_config_cache
from dssc_discovery.database.repositories import (
    ExperimentalConditionRepository,
    PhotovoltaicMeasurementRepository,
)
from dssc_discovery.domain.enums import ClosedLoopState, SpeciesCode
from dssc_discovery.optimization.initial_design import latin_hypercube_design
from dssc_discovery.optimization.search_space import search_space_for
from dssc_discovery.user_data import field_guide as fg
from dssc_discovery.user_data import planner, readiness
from dssc_discovery.user_data.past_results import (
    UnreadableFileError,
    check_rows,
    import_past_results,
    prepare_rows,
    read_uploaded_table,
)

SPECIES = SpeciesCode.BETA_VULGARIS


@pytest.fixture(autouse=True)
def small_thresholds(monkeypatch, tmp_path):
    monkeypatch.setenv("DSSC_MIN_REAL_CONDITIONS_FOR_EARLY_ESTIMATE", "4")
    monkeypatch.setenv("DSSC_MIN_REAL_CONDITIONS_FOR_RETRAINING", "7")
    monkeypatch.setenv("DSSC_BO_BATCH_SIZE", "2")
    monkeypatch.setenv("DSSC_MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    clear_config_cache()
    yield
    clear_config_cache()


def _rows(n: int, *, seed: int | None = None, start: int = 0) -> list[dict[str, str]]:
    """n distinct real-looking rows: a spread of recipes, plausible J-V results."""
    seed = 3 + start if seed is None else seed
    vectors = latin_hypercube_design(search_space_for(SPECIES), n, seed=seed)
    rows = []
    for i, vector in enumerate(vectors, start=start):
        v = vector.model_dump(mode="json")
        row = {k: str(val) for k, val in v.items()}
        row.update(
            species_code=SPECIES.value,
            condition_code=f"OWN-{i:02d}",
            active_area_cm2="0.25",
            electrolyte_type="iodide_triiodide",
            counter_electrode_type="platinum",
            illumination_protocol="am1.5g_1_sun",
            measurement_date="2026-10-01",
            open_circuit_voltage_v="0.6",
            short_circuit_current_density_ma_cm2=str(6.0 + (i % 5)),
            fill_factor="0.6",
        )
        rows.append(row)
    return rows


# --- field guide -------------------------------------------------------------------


def test_template_columns_cover_every_required_field_and_parse():
    csv_text = fg.template_csv()
    header = csv_text.splitlines()[0].split(",")
    assert set(fg.REQUIRED_FIELD_NAMES) <= set(header)
    rows = read_uploaded_table(csv_text.encode(), "template.csv")
    assert len(rows) == 1
    assert check_rows(rows, "owner").ok


def test_every_field_has_plain_help_and_an_example():
    for spec in fg.ALL_FIELDS:
        assert spec.help and spec.example and spec.label


def test_instructions_mention_each_column():
    text = fg.instructions_text()
    for spec in fg.ALL_FIELDS:
        assert spec.name in text


def test_unreadable_file_type_is_rejected_in_plain_words():
    with pytest.raises(UnreadableFileError):
        read_uploaded_table(b"x", "notes.pdf")


# --- checking ------------------------------------------------------------------------


def test_missing_value_is_named_in_plain_language():
    rows = _rows(2)
    rows[1]["ph"] = ""
    result = check_rows(rows, "owner")
    assert not result.ok
    assert result.missing_by_field == {"ph": [2]}
    assert "pH of the dye extract" in result.problems[0].message


def test_pce_is_calculated_when_blank_and_checked_when_given():
    rows = _rows(1)
    prepared = prepare_rows(rows, "owner")
    assert float(prepared[0]["power_conversion_efficiency_percent"]) > 0
    rows[0]["power_conversion_efficiency_percent"] = "30"
    result = check_rows(rows, "owner")
    assert not result.ok
    assert "does not match" in result.problems[0].message


def test_forgiving_choice_spelling_and_forced_real_data():
    rows = _rows(1)
    rows[0]["solvent"] = "Ethanol"
    rows[0]["surface_treatment"] = "TiCl4"
    rows[0]["is_synthetic"] = "true"
    prepared = prepare_rows(rows, "owner")[0]
    assert prepared["solvent"] == "ethanol"
    assert prepared["surface_treatment"] == "ticl4"
    assert prepared["is_synthetic"] == "false"


def test_out_of_range_value_warns_but_does_not_block():
    rows = _rows(1)
    rows[0]["tio2_thickness_um"] = "60"
    result = check_rows(rows, "owner")
    assert result.ok
    assert any("tio2_thickness_um" in w.message for w in result.warnings)


def test_percentage_fill_factor_is_flagged():
    rows = _rows(1)
    rows[0]["fill_factor"] = "62"
    rows[0]["power_conversion_efficiency_percent"] = ""
    result = check_rows(rows, "owner")
    assert not result.ok or any("fraction" in w.message for w in result.warnings)


# --- importing -----------------------------------------------------------------------


def test_import_saves_real_ingested_data_and_names_the_owner(session):
    result = import_past_results(session, _rows(3), "Ada")
    assert result.saved
    assert sorted(result.conditions_ingested) == ["OWN-00", "OWN-01", "OWN-02"]
    conditions = ExperimentalConditionRepository(session).list()
    assert all(c.state == ClosedLoopState.INGESTED and not c.is_synthetic for c in conditions)
    assert all(c.operator == "Ada" for c in conditions)
    assert len(PhotovoltaicMeasurementRepository(session).list()) == 3


def test_failed_batch_saves_nothing(session):
    rows = _rows(3)
    rows[2]["open_circuit_voltage_v"] = "abc"
    result = import_past_results(session, rows, "Ada")
    assert not result.saved
    assert result.problems and "must be a number" in result.problems[0].message
    assert ExperimentalConditionRepository(session).list() == []


def test_repeat_cells_of_one_recipe_are_numbered_automatically(session):
    base = _rows(1)[0]
    rows = [dict(base), dict(base)]
    result = import_past_results(session, rows, "Ada")
    assert result.saved
    conditions = ExperimentalConditionRepository(session).list()
    assert len(conditions) == 1
    assert len(PhotovoltaicMeasurementRepository(session).list()) == 2
    # a later upload of another repeat continues the numbering instead of colliding
    again = import_past_results(session, [dict(base)], "Ada")
    assert again.saved


def test_blank_names_are_generated_and_identical_recipes_group(session):
    base = _rows(1)[0]
    base["condition_code"] = ""
    result = import_past_results(session, [dict(base), dict(base)], "Ada")
    assert result.saved
    assert len(ExperimentalConditionRepository(session).list()) == 1


def test_results_for_a_proposed_experiment_complete_it(session):
    plan = planner.plan_starting_experiments(session, SPECIES, "Ada", n=3)
    first = plan.experiments[0]
    row = {k: str(v) for k, v in first.parameters.items()}
    row.update(
        species_code=SPECIES.value,
        condition_code=first.condition_code,
        active_area_cm2="0.25",
        electrolyte_type="iodide_triiodide",
        counter_electrode_type="platinum",
        illumination_protocol="am1.5g_1_sun",
        measurement_date="2026-10-02",
        open_circuit_voltage_v="0.6",
        short_circuit_current_density_ma_cm2="8",
        fill_factor="0.6",
    )
    result = import_past_results(session, [row], "Ada")
    assert result.saved, result.problems
    state = (
        ExperimentalConditionRepository(session).get_one(condition_code=first.condition_code).state
    )
    assert state == ClosedLoopState.INGESTED


# --- readiness and planning -----------------------------------------------------------


def test_readiness_levels(session):
    assert readiness.assess_readiness(session, SPECIES).level == readiness.NONE
    import_past_results(session, _rows(2), "Ada")
    info = readiness.assess_readiness(session, SPECIES)
    assert info.level == readiness.TOO_FEW and info.needed_for_early == 2
    import_past_results(session, _rows(2, start=2), "Ada")
    assert readiness.assess_readiness(session, SPECIES).level == readiness.EARLY
    import_past_results(session, _rows(3, start=4), "Ada")
    assert readiness.assess_readiness(session, SPECIES).level == readiness.FULL


def test_unchanged_settings_are_reported(session):
    rows = _rows(4)
    for row in rows:
        row["ph"] = "5"
    import_past_results(session, rows, "Ada")
    info = readiness.assess_readiness(session, SPECIES)
    assert "pH of the dye extract" in info.unchanged_settings


def test_no_data_gets_a_spread_out_starting_set_and_lab_sheet(session):
    plan = planner.plan_starting_experiments(session, SPECIES, "Ada", n=6)
    assert plan.kind == "starting" and len(plan.experiments) == 6
    codes = {e.condition_code for e in plan.experiments}
    assert len(codes) == 6
    second = planner.plan_starting_experiments(session, SPECIES, "Ada", n=2)
    assert not codes & {e.condition_code for e in second.experiments}
    sheet = planner.lab_sheet_csv(SPECIES, plan.experiments)
    assert sheet.splitlines()[0].startswith("condition_code,species_code,solvent")
    assert len(sheet.splitlines()) == 7
    assert len(planner.pending_plan(session, SPECIES)) == 8


def test_too_few_experiments_refuse_to_predict_with_a_clear_message(session):
    import_past_results(session, _rows(2), "Ada")
    with pytest.raises(InsufficientRealDataError, match="Add 2 more"):
        planner.plan_next_experiments(session, SPECIES, "Ada")


def test_early_estimate_is_labelled_low_confidence_and_predicts(session):
    import_past_results(session, _rows(5), "Ada")
    plan = planner.plan_next_experiments(session, SPECIES, "Ada", batch_size=2)
    assert plan.kind == "optimizer"
    assert "low confidence" in plan.confidence
    assert plan.experiments and all(e.predicted_pce is not None for e in plan.experiments)
    values = search_space_for(SPECIES).decode_values([0.5] * 13)
    prediction = planner.predict_recipe(session, SPECIES, values)
    assert "low confidence" in prediction.confidence
    assert prediction.std_pce > 0


def test_full_mode_when_enough_data(session):
    import_past_results(session, _rows(7), "Ada")
    plan = planner.plan_next_experiments(session, SPECIES, "Ada", batch_size=2)
    assert plan.confidence == "full"
