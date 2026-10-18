# dssc-discovery

Closed-loop, ML-assisted discovery and experimental optimization of natural
plant pigments for dye-sensitized solar cells (DSSCs).

PhD research platform. See `PLAN.txt` for the full architecture and
chunked development roadmap (milestones M0–M11). Implementation proceeds one
chunk at a time; do not skip ahead of the roadmap.

Status: all roadmap milestones (M0-M11) are implemented and tested on
synthetic data (498 tests). Results become scientifically meaningful only
with real data (see each milestone's gate in PLAN.txt).

Dashboard:

    pip install -e ".[ui]"
    streamlit run streamlit_app.py

**Using your own data (no coding needed).** Open the dashboard and use the first two
pages, *Start here* and *Plan my experiments*:

1. *Start here* -- type in results you already have (one form, every box explained) or
   upload a filled-in CSV/Excel template. Missing or odd values are explained in plain
   words and nothing is saved until everything is valid. Rows are always saved as real data.
2. *Plan my experiments* -- with no data yet you get a spread-out starting set of recipes
   to run in the lab (plus a lab sheet to take to the bench). With `min_real_conditions_for_early_estimate`
   (default 8) real experiments you get a clearly-labelled low-confidence **early estimate**;
   at `min_real_conditions_for_retraining` (default 40) the full mode. A "try my own recipe"
   box predicts the efficiency and its uncertainty before you spend any materials.

The sidebar switches between **My real data** and **Demo (simulated)**; the demo lives in a
separate database (`demo_database_url`) so simulated rows never mix with real ones.

Demo login: **admin / admin** -- hard-coded for this school project only, not
real security. The database tables are created automatically on first run; on
an empty database the dashboard offers a **Load demo data** button (simulated
data, clearly marked as synthetic).
Real closed loop: `scripts/generate_initial_doe.py` -> `scripts/closed_loop_cli.py`
(approve / fabricated / ingest-results) -> `scripts/retrain_on_real_data.py` ->
`scripts/run_bo_cycle.py`.
