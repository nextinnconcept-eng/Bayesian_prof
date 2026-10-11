"""Manual closed-loop actions (Section K): the human side of the state
machine. Every action is recorded in closed_loop_state_log with the
operator you give.

    python scripts/closed_loop_cli.py list --species BETA_VULGARIS --state proposed
    python scripts/closed_loop_cli.py approve BETA_VULGARIS-DOE-001 \
        --operator "A. Researcher" --note "reviewed"
    python scripts/closed_loop_cli.py reject CODE --operator NAME --note "solvent unavailable"
    python scripts/closed_loop_cli.py fabricated CODE --operator NAME
    python scripts/closed_loop_cli.py fabrication-failed CODE --operator NAME --note "film cracked"
    python scripts/closed_loop_cli.py measurement-failed CODE --operator NAME --note "cell shorted"
    python scripts/closed_loop_cli.py ingest-results results.csv --operator NAME
    python scripts/closed_loop_cli.py history CODE
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import species_arg

from dssc_discovery.closed_loop import state_machine as sm
from dssc_discovery.closed_loop.real_ingestion import advance_measured_conditions
from dssc_discovery.database.repositories import (
    ExperimentalConditionRepository,
    RecommendationRepository,
)
from dssc_discovery.database.session import get_session_factory
from dssc_discovery.domain.enums import ClosedLoopState
from dssc_discovery.ingestion.csv_loader import load_condition_device_pv_csv

ACTIONS = {
    "approve": lambda s, cid, op, note: sm.approve(s, cid, op, note),
    "reject": lambda s, cid, op, note: sm.reject(s, cid, op, note),
    "fabricated": lambda s, cid, op, note: sm.mark_fabricated(s, cid, op, note),
    "fabrication-failed": lambda s, cid, op, note: sm.mark_fabrication_failed(s, cid, op, note),
    "measurement-failed": lambda s, cid, op, note: sm.mark_measurement_failed(s, cid, op, note),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list")
    listing.add_argument("--species", type=species_arg)
    listing.add_argument("--state", choices=[s.value for s in ClosedLoopState])

    for name in ACTIONS:
        action = sub.add_parser(name)
        action.add_argument("condition_code")
        action.add_argument("--operator", required=True)
        action.add_argument("--note", default="")

    ingest = sub.add_parser("ingest-results")
    ingest.add_argument("csv_path", type=Path)
    ingest.add_argument("--operator", required=True)

    show = sub.add_parser("history")
    show.add_argument("condition_code")

    args = parser.parse_args()
    with get_session_factory()() as session:
        repository = ExperimentalConditionRepository(session)
        if args.command == "list":
            filters: dict[str, object] = {}
            if args.species:
                filters["species_code"] = args.species
            if args.state:
                filters["state"] = ClosedLoopState(args.state)
            rows = repository.filter(**filters) if filters else repository.list()
            for c in sorted(rows, key=lambda c: c.condition_code):
                rec = RecommendationRepository(session).get_one(experimental_condition_id=c.id)
                extra = f"  {rec.rationale_text}" if rec else ""
                badge = " [SYNTHETIC]" if c.is_synthetic else ""
                print(f"{c.condition_code}{badge}  {c.state.value}{extra}")
            return
        if args.command == "ingest-results":
            report = load_condition_device_pv_csv(args.csv_path, session)
            if not report.all_succeeded:
                for failure in report.failures:
                    print(f"row failed: {failure}")
                print("ingestion had failures; nothing advanced")
                return
            codes = {
                c.condition_code
                for c in repository.list()
                if c.state in (ClosedLoopState.FABRICATED, ClosedLoopState.MEASURED)
            }
            advance = advance_measured_conditions(session, sorted(codes), args.operator)
            session.commit()
            print(f"ingested: {advance.ingested}")
            for code, reason in advance.blocked.items():
                print(f"BLOCKED {code}: {reason}")
            return

        condition = repository.get_one(condition_code=args.condition_code)
        if condition is None:
            raise SystemExit(f"unknown condition_code {args.condition_code!r}")
        if args.command == "history":
            for row in sm.history(session, condition.id):
                print(
                    f"{row.created_at:%Y-%m-%d %H:%M} {row.operator}: "
                    f"{row.from_state.value if row.from_state else '-'} -> "
                    f"{row.to_state.value} {row.reason}"
                )
            return
        updated = ACTIONS[args.command](session, condition.id, args.operator, args.note)
        session.commit()
        print(f"{updated.condition_code}: now {updated.state.value}")


if __name__ == "__main__":
    main()
