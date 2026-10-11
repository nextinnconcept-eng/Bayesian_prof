"""Shared Streamlit plumbing: one DB session per rerun and the synthetic
badge. Layout only -- all queries live in dssc_discovery.reports.ui_queries."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy.orm import Session

from dssc_discovery.database.session import get_session_factory

SYNTHETIC_BADGE = "🧪 SYNTHETIC"


def session() -> Session:
    if "db_session" not in st.session_state:
        st.session_state["db_session"] = get_session_factory()()
    return st.session_state["db_session"]


def badge_synthetic(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "is_synthetic" not in frame.columns:
        return frame
    frame = frame.copy()
    frame.insert(0, "data", frame["is_synthetic"].map({True: SYNTHETIC_BADGE, False: "real"}))
    return frame


def show_table(frame: pd.DataFrame, empty_message: str = "Nothing recorded yet.") -> None:
    if frame.empty:
        st.info(empty_message)
    else:
        st.dataframe(badge_synthetic(frame), use_container_width=True, hide_index=True)
