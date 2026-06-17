from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_binary(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (scores >= threshold).astype(int)
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = float("nan")
    return {
        "threshold": float(threshold),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

