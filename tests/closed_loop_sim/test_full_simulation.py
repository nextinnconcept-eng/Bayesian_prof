"""M7 gate (Chunk 7.5): three full synthetic cycles end to end."""

from __future__ import annotations

from dssc_discovery.closed_loop.orchestrator import run_simulated_closed_loop
from dssc_discovery.database.repositories import (
    BOIterationRepository,
    ExperimentalConditionRepository,
    ModelRunRepository,
)
from dssc_discovery.domain.enums import ClosedLoopState
from dssc_discovery.features.dataset_builder import build_ml_dataset
from tests.bo_helpers import FAST


def test_three_cycle_simulation(session, tmp_path):
    report = run_simulated_closed_loop(
        session,
        n_cycles=3,
        initial_doe_size=12,
        batch_size=2,
        seed=0,
        tracking_uri=str(tmp_path / "mlruns"),
        acquisition_options=FAST,
    )

    assert [c.cycle for c in report.cycles] == [1, 2, 3]
    best = [report.initial_best_true_value] + [c.best_true_value for c in report.cycles]
    assert best == sorted(best)  # best-so-far never regresses
    assert all(c.n_recommended == 2 for c in report.cycles)

    runs = ModelRunRepository(session).list()
    iterations = BOIterationRepository(session).list()
    assert len(runs) == 3 and len({i.model_run_id for i in iterations}) == 3
    assert all(r.is_synthetic for r in runs)
    assert build_ml_dataset(session)["is_synthetic"].all()

    states = {c.state for c in ExperimentalConditionRepository(session).list()}
    # last cycle's results are ingested but not yet retrained on
    assert states == {ClosedLoopState.MODEL_UPDATED, ClosedLoopState.INGESTED}
    assert "human decision" in report.convergence_summary
