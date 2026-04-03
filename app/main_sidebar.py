"""Shared sidebar: deployment / model status (used by home and optional pages)."""

from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PATH, MODEL_PATH


def render_shared_sidebar() -> None:
    with st.sidebar:
        st.header("Deployment")
        st.caption(f"Data: `{DATA_PATH.name}`")
        st.caption(f"Model artifact: `{MODEL_PATH}`")
        if MODEL_PATH.is_file():
            st.success("Model file found — inference ready.")
        else:
            st.error("Model missing. Run `python train_model.py` from the project root.")
        st.caption("Switch pages via the sidebar **app pages** menu (Manual = home, Bulk = second page).")
