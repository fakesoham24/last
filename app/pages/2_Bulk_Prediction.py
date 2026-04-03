"""Second page: bulk file upload & batch inference (bulk_prediction module unchanged)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

try:
    from app.bulk_prediction import render_bulk_prediction_scanner
    from app.main_sidebar import render_shared_sidebar
except ImportError:
    from bulk_prediction import render_bulk_prediction_scanner
    from main_sidebar import render_shared_sidebar

st.set_page_config(
    page_title="Bulk Prediction Scanner",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_shared_sidebar()
render_bulk_prediction_scanner()
