from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from prompt_injection_detector.evaluation.metrics import evaluate_binary
from prompt_injection_detector.evaluation.thresholds import operating_points
from prompt_injection_detector.schema import InjectionCategory, Prediction


MODEL_FACTORIES = {
    "logistic_regression": lambda: LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    ),
    "svm_rbf": lambda: SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=42,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}


@dataclass
class TrainedDetector:
    name: str
    pipeline: Pipeline
    category_pipeline: Pipeline
    threshold: float
    metrics: dict[str, Any]

    def predict_one(self, text: str) -> Prediction:
        score = score_text(self.pipeline, [text])[0]
        is_injection = score >= self.threshold
        category = self.category_pipeline.predict([text])[0] if is_injection else InjectionCategory.CLEAN.value
        return Prediction(
            text=text,
            is_injection=bool(is_injection),
            score=float(score),
            category=InjectionCategory(category),
            top_features=explain_text(self.pipeline, text),
        )


def make_pipeline(model_name: str) -> Pipeline:
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {sorted(MODEL_FACTORIES)}")
    model = MODEL_FACTORIES[model_name]()
    if model_name == "svm_rbf":
        model = CalibratedClassifierCV(model, method="sigmoid", cv=3)
    return Pipeline(
        steps=[
            (
                "tfidf",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=(1, 3),
                                min_df=1,
                                max_df=0.95,
                                sublinear_tf=True,
                                strip_accents="unicode",
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=1,
                                max_df=0.98,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            ("model", model),
        ]
    )


def score_text(pipeline: Pipeline, texts: list[str] | pd.Series) -> np.ndarray:
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        return pipeline.predict_proba(texts)[:, 1]
    decision = pipeline.decision_function(texts)
    return 1.0 / (1.0 + np.exp(-decision))


def find_recall_threshold(y_true: np.ndarray, scores: np.ndarray, min_precision: float = 0.55) -> float:
    rows: list[tuple[float, float, float, float]] = []
    for threshold in np.linspace(0.1, 0.9, 81):
        y_pred = (scores >= threshold).astype(int)
        predicted_positive = max(y_pred.sum(), 1)
        precision = ((y_pred == 1) & (y_true == 1)).sum() / predicted_positive
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if precision >= min_precision:
            rows.append((float(threshold), float(precision), float(recall), float(f1)))
    if not rows:
        return 0.5

    best_recall = max(row[2] for row in rows)
    recall_candidates = [row for row in rows if row[2] == best_recall]
    best_f1 = max(row[3] for row in recall_candidates)
    plateau = [row[0] for row in recall_candidates if row[3] == best_f1]
    return float(np.median(plateau))


def train_classical_models(
    train: pd.DataFrame,
    val: pd.DataFrame,
    model_names: list[str] | None = None,
    decision_threshold: float | None = None,
) -> TrainedDetector:
    model_names = model_names or list(MODEL_FACTORIES)
    trained: list[TrainedDetector] = []

    for name in model_names:
        pipeline = make_pipeline(name)
        pipeline.fit(train["text"], train["label"].astype(int))
        val_scores = score_text(pipeline, val["text"])
        threshold = decision_threshold or find_recall_threshold(val["label"].to_numpy(dtype=int), val_scores)
        metrics = evaluate_binary(val["label"].to_numpy(dtype=int), val_scores, threshold)
        metrics["validation_operating_points"] = operating_points(
            val["label"].to_numpy(dtype=int),
            val_scores,
        )

        category_pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, strip_accents="unicode")),
                ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
            ]
        )
        injection_train = train[train["label"].astype(int) == 1]
        category_pipeline.fit(injection_train["text"], injection_train["category"])
        trained.append(TrainedDetector(name, pipeline, category_pipeline, threshold, metrics))

    return sorted(
        trained,
        key=lambda item: (
            item.metrics["classification_report"]["1"]["recall"],
            item.metrics["classification_report"]["1"]["f1-score"],
        ),
        reverse=True,
    )[0]


def evaluate_detector(detector: TrainedDetector, frame: pd.DataFrame) -> dict[str, Any]:
    scores = score_text(detector.pipeline, frame["text"])
    metrics = evaluate_binary(frame["label"].to_numpy(dtype=int), scores, detector.threshold)
    y_pred = (scores >= detector.threshold).astype(int)
    injection_frame = frame[frame["label"].astype(int) == 1].copy()
    if not injection_frame.empty:
        injection_scores = scores[frame["label"].astype(int).to_numpy() == 1]
        injection_frame["detected"] = injection_scores >= detector.threshold
        metrics["per_category_detection_rate"] = (
            injection_frame.groupby("category")["detected"].mean().sort_index().to_dict()
        )
    else:
        metrics["per_category_detection_rate"] = {}
    metrics["predictions"] = y_pred.tolist()
    metrics["scores"] = scores.tolist()
    return metrics


def save_detector(detector: TrainedDetector, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, path)
    return path


def load_detector(path: str | Path) -> TrainedDetector:
    return joblib.load(path)


def explain_text(pipeline: Pipeline, text: str, top_k: int = 8) -> list[tuple[str, float]]:
    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["model"]
    vector = vectorizer.transform([text])
    feature_names = np.asarray(vectorizer.get_feature_names_out())

    if hasattr(model, "coef_"):
        weights = vector.multiply(model.coef_[0]).toarray()[0]
    elif hasattr(model, "feature_importances_"):
        weights = vector.toarray()[0] * model.feature_importances_
    else:
        values = vector.toarray()[0]
        indices = np.argsort(values)[::-1][:top_k]
        return [(feature_names[i], float(values[i])) for i in indices if values[i] > 0]

    indices = np.argsort(np.abs(weights))[::-1][:top_k]
    return [(feature_names[i], float(weights[i])) for i in indices if weights[i] != 0]
