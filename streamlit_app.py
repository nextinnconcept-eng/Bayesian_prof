"""DSSC Discovery dashboard -- main entry point.

    pip install -e ".[ui]"
    streamlit run streamlit_app.py

Sign in with the demo login admin / admin. The app lives in ui/streamlit_app/.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "ui" / "streamlit_app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app import main  # noqa: E402

main()
