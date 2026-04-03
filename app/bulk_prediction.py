"""Bulk Prediction Scanner UI: templates, upload, validation, batch inference, download."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.config import FEATURE_COLUMNS, MODEL_PATH
from src.predict import attach_predictions, labels_to_yes_no, load_pipeline, predict_batch
from src.utils import (
    ValidationError,
    dataframe_to_download_bytes,
    load_dataframe_from_bytes,
    sample_dataframe,
    sample_json_bytes,
    validate_features,
)


def _sample_csv_bytes() -> bytes:
    return sample_dataframe().to_csv(index=False).encode("utf-8")


def _sample_xlsx_bytes() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        sample_dataframe().to_excel(writer, index=False, sheet_name="data")
    return buf.getvalue()


def render_bulk_prediction_scanner() -> None:
    """Streamlit section: Bulk Prediction Scanner."""
    st.markdown("### 🔍 Bulk Prediction Scanner")
    st.caption(
        "Upload a file with the same columns as training data (without `y`). "
        "Predictions use the saved sklearn **Pipeline** (preprocessing + model)."
    )

    st.markdown("#### Sample templates")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            label="Download CSV sample",
            data=_sample_csv_bytes(),
            file_name="sample_bulk_input.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            label="Download Excel sample",
            data=_sample_xlsx_bytes(),
            file_name="sample_bulk_input.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            label="Download JSON sample",
            data=sample_json_bytes(),
            file_name="sample_bulk_input.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload input file (CSV, XLSX, JSON)",
        type=["csv", "xlsx", "xls", "json"],
        help="Drag and drop or browse. Columns must match training features (target column `y` optional).",
    )

    output_fmt = st.radio(
        "Result file format",
        options=["CSV (default)", "Excel"],
        horizontal=True,
    )
    fmt_key = "csv" if output_fmt.startswith("CSV") else "xlsx"

    run = st.button("Run Bulk Prediction", type="primary", use_container_width=False)

    if not run:
        return

    if uploaded is None:
        st.warning("Please upload a file before running bulk prediction.")
        return

    progress = st.progress(0, text="Loading model…")
    try:
        model = load_pipeline(MODEL_PATH)
    except FileNotFoundError as e:
        progress.empty()
        st.error(str(e))
        return

    progress.progress(20, text="Reading file…")
    try:
        raw_bytes = uploaded.getvalue()
        df_raw = load_dataframe_from_bytes(raw_bytes, uploaded.name)
    except ValidationError as e:
        progress.empty()
        st.error(e.message)
        return

    st.subheader("Preview (first rows)")
    st.dataframe(df_raw.head(15), use_container_width=True)

    progress.progress(45, text="Validating columns and types…")
    try:
        df_valid, val_warnings = validate_features(df_raw, drop_target_if_present=True)
    except ValidationError as e:
        progress.empty()
        st.error(e.message)
        if e.details:
            with st.expander("Details"):
                st.json(e.details)
        return

    for w in val_warnings:
        st.info(w)

    n_rows = len(df_valid)
    progress.progress(65, text=f"Running batch prediction ({n_rows} rows)…")

    try:
        labels, proba_yes = predict_batch(model, df_valid)
    except Exception as exc:
        progress.empty()
        st.error(f"Prediction failed: {exc!s}")
        return

    pred_yes_no = labels_to_yes_no(labels)
    result = attach_predictions(df_valid, pred_yes_no, proba_yes)

    progress.progress(90, text="Preparing download…")

    yes_count = int((pred_yes_no == "Yes").sum())
    no_count = int((pred_yes_no == "No").sum())
    total = yes_count + no_count or 1

    st.subheader("Prediction summary")
    c_a, c_b, c_c = st.columns(3)
    c_a.metric("Rows", n_rows)
    c_b.metric("Predicted Yes", f"{yes_count} ({100 * yes_count / total:.1f}%)")
    c_c.metric("Predicted No", f"{no_count} ({100 * no_count / total:.1f}%)")

    try:
        data_bytes, fname, mime = dataframe_to_download_bytes(result, fmt_key)
    except Exception as exc:
        progress.empty()
        st.error(f"Could not build output file: {exc!s}")
        return

    progress.progress(100, text="Done.")
    progress.empty()

    st.download_button(
        label="Download Predictions",
        data=data_bytes,
        file_name=fname,
        mime=mime,
        type="primary",
        use_container_width=True,
    )

    with st.expander("Result preview"):
        st.dataframe(result.head(20), use_container_width=True)
