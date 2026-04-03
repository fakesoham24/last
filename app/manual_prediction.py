"""Single-client manual prediction form (hands-free UI)."""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_COLUMNS,
    MODEL_PATH,
    NUMERIC_FEATURES,
)
from src.predict import load_pipeline, predict_batch, labels_to_yes_no


@st.cache_data(show_spinner=False)
def _load_training_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def _categorical_choices() -> dict[str, list[str]]:
    df = _load_training_data()
    out: dict[str, list[str]] = {}
    for col in CATEGORICAL_FEATURES:
        vals = sorted(df[col].astype(str).str.strip().unique().tolist())
        out[col] = vals
    return out


def _default_index(options: list[str], preferred: str) -> int:
    p = preferred.lower().strip()
    for i, o in enumerate(options):
        if o.lower().strip() == p:
            return i
    return 0


def render_manual_prediction_page() -> None:
    """Main landing experience: form + predict button."""
    st.title("🏦 Bank Term Deposit Prediction System")
    st.caption(
        "Predict whether a client will subscribe to a term deposit based on their profile and campaign history."
    )

    choices = _categorical_choices()

    st.subheader("📝 Client Information")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=18, max_value=120, value=30, step=1)
        job_opts = choices["job"]
        job = st.selectbox(
            "Job",
            options=job_opts,
            index=_default_index(job_opts, "admin."),
        )
        marital_opts = choices["marital"]
        marital = st.selectbox(
            "Marital Status",
            options=marital_opts,
            index=_default_index(marital_opts, "divorced"),
        )
        education_opts = choices["education"]
        education = st.selectbox(
            "Education",
            options=education_opts,
            index=_default_index(education_opts, "primary"),
        )
        default_opts = choices["default"]
        default_v = st.selectbox(
            "Has Credit in Default?",
            options=default_opts,
            index=_default_index(default_opts, "no"),
        )
        balance = st.number_input(
            "Yearly Average Balance (in EUR)",
            min_value=-10_000,
            max_value=200_000,
            value=1000,
            step=100,
        )

    with c2:
        housing_opts = choices["housing"]
        housing = st.selectbox(
            "Has Housing Loan?",
            options=housing_opts,
            index=_default_index(housing_opts, "no"),
        )
        loan_opts = choices["loan"]
        loan = st.selectbox(
            "Has Personal Loan?",
            options=loan_opts,
            index=_default_index(loan_opts, "no"),
        )
        contact_opts = choices["contact"]
        contact = st.selectbox(
            "Contact Communication Type",
            options=contact_opts,
            index=_default_index(contact_opts, "cellular"),
        )
        day = st.number_input(
            "Last Contact Day of the Month",
            min_value=1,
            max_value=31,
            value=15,
            step=1,
        )
        month_opts = choices["month"]
        month = st.selectbox(
            "Last Contact Month",
            options=month_opts,
            index=_default_index(month_opts, "jan"),
        )

    with c3:
        duration = st.number_input(
            "Last Contact Duration (seconds)",
            min_value=0,
            max_value=10_000,
            value=200,
            step=10,
        )
        campaign = st.number_input(
            "Number of Contacts during this campaign",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
        )
        pdays = st.number_input(
            "Days since last contact (from previous campaign)",
            min_value=-1,
            max_value=1000,
            value=-1,
            step=1,
        )
        previous = st.number_input(
            "Number of Contacts performed before this campaign",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )
        poutcome_opts = choices["poutcome"]
        poutcome = st.selectbox(
            "Outcome of previous campaign",
            options=poutcome_opts,
            index=_default_index(poutcome_opts, "failure"),
        )

    submitted = st.button("🚀 Predict Conversion", use_container_width=True, type="primary")

    row = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default_v,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
    }
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    if not submitted:
        return

    if not MODEL_PATH.is_file():
        st.error(f"Model not found at `{MODEL_PATH}`. Run `python train_model.py` first.")
        return

    try:
        model = load_pipeline(MODEL_PATH)
        labels, proba_yes = predict_batch(model, X)
    except Exception as exc:
        st.error(f"Prediction failed: {exc!s}")
        return

    pred_label = labels_to_yes_no(labels)[0]
    p_yes = float(proba_yes[0])

    st.success("Prediction complete")
    a, b, c = st.columns(3)
    a.metric("Predicted subscription", pred_label)
    b.metric("P(subscribe = Yes)", f"{100 * p_yes:.1f}%")
    c.metric("P(subscribe = No)", f"{100 * (1 - p_yes):.1f}%")

    st.caption(
        "Same fitted sklearn **Pipeline** as bulk mode: scaling, one-hot encoding, then classifier."
    )
