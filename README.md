# dssc-discovery

Closed-loop, ML-assisted discovery and experimental optimization of natural
plant pigments for dye-sensitized solar cells (DSSCs).

PhD research platform. See `PLAN.txt` for the full architecture and
chunked development roadmap (milestones M0–M11). Implementation proceeds one
chunk at a time; do not skip ahead of the roadmap.

Status: all roadmap milestones (M0-M11) are implemented and tested on
synthetic data (498 tests). Results become scientifically meaningful only
with real data (see each milestone's gate in PLAN.txt).

Dashboard: `streamlit run ui/streamlit_app/Home.py` from the repository root
(needs the `ui` extra). Demo login: **admin / admin** -- hard-coded for this
school project only, not real security.
Real closed loop: `scripts/generate_initial_doe.py` -> `scripts/closed_loop_cli.py`
(approve / fabricated / ingest-results) -> `scripts/retrain_on_real_data.py` ->
`scripts/run_bo_cycle.py`.
