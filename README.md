# dssc-discovery

Closed-loop, ML-assisted discovery and experimental optimization of natural
plant pigments for dye-sensitized solar cells (DSSCs).

PhD research platform. See `PLAN.txt` for the full architecture and
chunked development roadmap (milestones M0–M11). Implementation proceeds one
chunk at a time; do not skip ahead of the roadmap.

Status: all roadmap milestones (M0-M11) are implemented. M5-M11 have not
been tested yet -- run and test them before relying on any of it.

Dashboard: `streamlit run ui/streamlit_app/Home.py` (needs the `ui` extra).
Real closed loop: `scripts/generate_initial_doe.py` -> `scripts/closed_loop_cli.py`
(approve / fabricated / ingest-results) -> `scripts/retrain_on_real_data.py` ->
`scripts/run_bo_cycle.py`.
