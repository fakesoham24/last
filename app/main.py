"""Streamlit home page: manual single-client prediction."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

try:
    from app.manual_prediction import render_manual_prediction_page
    from app.main_sidebar import render_shared_sidebar
except ImportError:
    from manual_prediction import render_manual_prediction_page
    from main_sidebar import render_shared_sidebar


def main() -> None:
    st.set_page_config(
        page_title="Bank Term Deposit Prediction",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_shared_sidebar()
    render_manual_prediction_page()


if __name__ == "__main__":
    main()
