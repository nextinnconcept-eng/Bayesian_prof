"""Generate the real initial DoE (Chunk 8.1): a constrained Latin
Hypercube design (6.4) persisted as REAL (non-synthetic) PROPOSED
conditions. Nothing is approved -- a person reviews and approves each
condition with scripts/closed_loop_cli.py.

    python scripts/generate_initial_doe.py --species BETA_VULGARIS --operator "A. Researcher"
"""

from __future__ import annotations

import argparse

from _common import species_arg

from dssc_discovery.closed_loop.orchestrator import propose_conditions
from dssc_discovery.config import get_config
from dssc_discovery.database.session import get_session_factory
from dssc_discovery.domain.enums import DesignSource
from dssc_discovery.optimization.initial_design import (
    latin_hypercube_design,
    min_pairwise_distance,
)
from dssc_discovery.optimization.search_space import search_space_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", type=species_arg, required=True)
    parser.add_argument("--operator", required=True, help="person generating the design")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the design, write nothing")
    args = parser.parse_args()

    space = search_space_for(args.species)
    seed = get_config().random_seed if args.seed is None else args.seed
    vectors = latin_hypercube_design(space, args.n, seed)
    print(
        f"{len(vectors)}-point design for {args.species.value} (search space {space.version}); "
        f"min pairwise distance {min_pairwise_distance(space, vectors):.3f}"
    )
    for i, vector in enumerate(vectors, start=1):
        print(f"{i:3d} {vector.model_dump(mode='json')}")
    if args.dry_run:
        return

    with get_session_factory()() as session:
        conditions = propose_conditions(
            session,
            args.species,
            vectors,
            design_source=DesignSource.INITIAL_DOE,
            code_prefix=f"{args.species.name}-DOE",
            operator=args.operator,
            is_synthetic=False,
        )
        session.commit()
    print(f"wrote {len(conditions)} PROPOSED conditions; none are approved.")


if __name__ == "__main__":
    main()
