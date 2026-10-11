"""Manually retrain the BO surrogate on real data (Chunk 8.3): new
dataset version, new real ModelRun, INGESTED conditions -> MODEL_UPDATED.
Refuses below min_real_conditions_for_retraining real conditions.

    python scripts/retrain_on_real_data.py --species BETA_VULGARIS --operator "A. Researcher"
"""

from __future__ import annotations

import argparse

from _common import species_arg

from dssc_discovery.closed_loop.real_workflow import (
    InsufficientRealDataError,
    retrain_on_real_data,
)
from dssc_discovery.database.session import get_session_factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", type=species_arg, required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    with get_session_factory()() as session:
        try:
            surrogate = retrain_on_real_data(session, args.species, args.operator, seed=args.seed)
        except InsufficientRealDataError as exc:
            raise SystemExit(f"refused: {exc}") from exc
        session.commit()
    print(
        f"retrained {args.species.value}: ModelRun {surrogate.model_run_id}, "
        f"{len(surrogate.train_Y)} training rows, best observed {surrogate.best_observed:.2f}%"
    )


if __name__ == "__main__":
    main()
