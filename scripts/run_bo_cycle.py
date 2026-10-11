"""Generate a real BO cycle's recommendations (Chunk 8.4) from the latest
real surrogate ModelRun. Output goes to PROPOSED and waits for human
approval (scripts/closed_loop_cli.py), exactly like the initial DoE.

Gate (PLAN.txt 8.4): only after the M7 simulation passed and the real
initial DoE (>= min_real_conditions_for_retraining conditions) is fully
ingested and retrained on.

    python scripts/run_bo_cycle.py --species BETA_VULGARIS --cycle 1 --operator "A. Researcher"
"""

from __future__ import annotations

import argparse
import json

from _common import species_arg

from dssc_discovery.closed_loop.real_workflow import (
    InsufficientRealDataError,
    recommend_real_cycle,
)
from dssc_discovery.database.session import get_session_factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", type=species_arg, required=True)
    parser.add_argument("--cycle", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--operator", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    with get_session_factory()() as session:
        try:
            batch = recommend_real_cycle(
                session, args.species, args.cycle, args.operator, args.batch_size
            )
        except InsufficientRealDataError as exc:
            raise SystemExit(f"refused: {exc}") from exc
        session.commit()
    rejected = json.loads(batch.bo_iteration.rejected_candidates_json)
    print(
        f"BOIteration {batch.bo_iteration.id}: {len(batch.recommendations)} PROPOSED, "
        f"{len(rejected)} infeasible candidates rejected"
    )
    for condition, rec in zip(batch.conditions, batch.recommendations, strict=True):
        print(f"{condition.condition_code}: {rec.rationale_text}")


if __name__ == "__main__":
    main()
