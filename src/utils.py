"""File loading and dataframe validation for bulk uploads."""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd

from src.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


class ValidationError(Exception):
    """Raised when uploaded data fails schema or type checks."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


def load_dataframe_from_bytes(
    raw: bytes,
    filename: str,
) -> pd.DataFrame:
    """
    Load CSV, XLSX, or JSON from bytes. Raises ValidationError on empty/corrupt input.
    """
    if not raw or not raw.strip():
        raise ValidationError("The uploaded file is empty.")

    name = (filename or "").lower()

    try:
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(io.BytesIO(raw))
        if name.endswith(".json"):
            return _read_json_flexible(io.BytesIO(raw))
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(
            f"Could not parse file as {'CSV' if name.endswith('.csv') else 'Excel' if 'xls' in name else 'JSON'}. "
            f"Check format and encoding. Detail: {exc!s}"
        ) from exc

    raise ValidationError(
        "Unsupported file type. Please upload a .csv, .xlsx, or .json file."
    )


def _read_json_flexible(buffer: io.BytesIO) -> pd.DataFrame:
    """Try common JSON layouts for tabular data."""
    raw = buffer.getvalue()
    try:
        return pd.read_json(io.BytesIO(raw), orient="records")
    except Exception:
        pass
    try:
        return pd.read_json(io.BytesIO(raw), lines=True)
    except Exception as exc:
        raise ValidationError(
            f"Could not read JSON as 'records' or line-delimited JSON. Detail: {exc!s}"
        ) from exc


def validate_features(
    df: pd.DataFrame,
    drop_target_if_present: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Check columns and coarse types. Returns cleaned df (only FEATURE_COLUMNS, ordered) and warnings.
    """
    warnings: list[str] = []
    work = df.copy()

    if work.empty:
        raise ValidationError("The file contains no rows.")

    cols = [str(c).strip() for c in work.columns]
    work.columns = cols

    if drop_target_if_present and "y" in work.columns:
        work = work.drop(columns=["y"])
        warnings.append("Column 'y' was present and ignored for prediction.")

    expected = set(FEATURE_COLUMNS)
    actual = set(work.columns)

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing:
        raise ValidationError(
            f"Missing required column(s): {', '.join(missing)}.",
            details={"missing": missing, "extra": extra},
        )
    if extra:
        raise ValidationError(
            f"Unexpected extra column(s): {', '.join(extra)}. Remove them or use only training features.",
            details={"missing": missing, "extra": extra},
        )

    work = work[FEATURE_COLUMNS].copy()

    for col in NUMERIC_FEATURES:
        coerced = pd.to_numeric(work[col], errors="coerce")
        if coerced.isna().all():
            raise ValidationError(
                f"Column '{col}' must be numeric; all values failed conversion."
            )
        if coerced.isna().any():
            raise ValidationError(
                f"Column '{col}' contains non-numeric values at rows: "
                f"{list(work.index[coerced.isna()].tolist()[:20])}"
                + (" ..." if coerced.isna().sum() > 20 else "")
            )
        work[col] = coerced

    for col in CATEGORICAL_FEATURES:
        work[col] = work[col].astype(str).str.strip()

    return work, warnings


def dataframe_to_download_bytes(
    df: pd.DataFrame,
    fmt: str,
) -> tuple[bytes, str, str]:
    """Serialize dataframe to bytes for st.download_button. fmt: 'csv' or 'xlsx'."""
    if fmt == "csv":
        data = df.to_csv(index=False).encode("utf-8")
        return data, "predictions.csv", "text/csv"
    if fmt == "xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="predictions")
        return buf.getvalue(), "predictions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    raise ValueError("fmt must be 'csv' or 'xlsx'")


def sample_dataframe(n: int = 5, seed: int = 42) -> pd.DataFrame:
    """Small deterministic sample for template downloads (no target)."""
    _ = seed  # reserved for future stratified sampling
    return pd.DataFrame(
        {
            "age": [32, 41, 28, 55, 37],
            "job": ["management", "technician", "student", "retired", "services"],
            "marital": ["married", "single", "single", "married", "divorced"],
            "education": ["tertiary", "secondary", "tertiary", "primary", "secondary"],
            "default": ["no", "no", "no", "no", "yes"],
            "balance": [1200, -200, 0, 450, 89],
            "housing": ["yes", "yes", "no", "yes", "no"],
            "loan": ["no", "no", "yes", "no", "no"],
            "contact": ["cellular", "unknown", "cellular", "telephone", "cellular"],
            "day": [12, 5, 18, 3, 22],
            "month": ["may", "jul", "aug", "feb", "apr"],
            "duration": [120, 300, 45, 200, 90],
            "campaign": [1, 2, 1, 4, 1],
            "pdays": [-1, -1, 180, -1, -1],
            "previous": [0, 0, 1, 0, 0],
            "poutcome": ["unknown", "failure", "success", "unknown", "other"],
        }
    ).head(n)


def sample_json_bytes() -> bytes:
    df = sample_dataframe()
    return json.dumps(df.to_dict(orient="records"), indent=2).encode("utf-8")
