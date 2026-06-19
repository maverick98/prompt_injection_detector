from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def threshold_sweep(y_true: np.ndarray, scores: np.ndarray, steps: int = 101) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.0, 1.0, steps):
        y_pred = (scores >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "false_positives": int(((y_true == 0) & (y_pred == 1)).sum()),
                "false_negatives": int(((y_true == 1) & (y_pred == 0)).sum()),
            }
        )
    return pd.DataFrame(rows)


def operating_points(y_true: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, float]]:
    sweep = threshold_sweep(y_true, scores)
    points: dict[str, dict[str, float]] = {}
    policies = {
        "max_recall": ("recall", False),
        "balanced_f1": ("f1", False),
        "high_precision": ("precision", False),
    }
    for name, (metric, _) in policies.items():
        candidates = sweep.copy()
        if name == "max_recall":
            candidates = candidates[candidates["precision"] >= 0.90]
            sort_cols = ["recall", "f1", "threshold"]
        elif name == "balanced_f1":
            sort_cols = ["f1", "recall", "precision"]
        else:
            candidates = candidates[candidates["recall"] >= 0.95]
            sort_cols = ["precision", "f1", "threshold"]
        if candidates.empty:
            candidates = sweep
        row = candidates.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]
        points[name] = {key: float(row[key]) for key in ["threshold", "precision", "recall", "f1"]}
        points[name]["false_positives"] = int(row["false_positives"])
        points[name]["false_negatives"] = int(row["false_negatives"])
    return points

