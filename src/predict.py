"""Load persisted pipeline and run vectorized batch predictions."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH


def load_pipeline(model_path: Path | str | None = None) -> Pipeline:
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found at {path}. Train the model first (see train_model.py)."
        )
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError("Loaded artifact is not a sklearn Pipeline.")
    return model


def predict_batch(
    model: Pipeline,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized predictions. Returns (labels_int, proba_yes).
    Assumes positive class index 1 corresponds to subscription (yes).
    """
    labels = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # Binary: column 1 = positive class
        proba_yes = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    else:
        proba_yes = np.asarray(labels, dtype=float)

    return np.asarray(labels).ravel(), np.asarray(proba_yes).ravel()


def labels_to_yes_no(labels: np.ndarray) -> np.ndarray:
    """Map 0/1 to No/Yes for display."""
    return np.where(labels.astype(int) == 1, "Yes", "No")


def attach_predictions(
    original_with_features: pd.DataFrame,
    prediction_yes_no: np.ndarray,
    proba_yes: np.ndarray | None = None,
) -> pd.DataFrame:
    out = original_with_features.copy()
    out["prediction"] = prediction_yes_no
    if proba_yes is not None:
        out["probability_yes"] = np.round(proba_yes.astype(float), 4)
    return out
