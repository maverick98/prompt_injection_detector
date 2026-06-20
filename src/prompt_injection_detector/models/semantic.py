from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prompt_injection_detector.evaluation.metrics import evaluate_binary
from prompt_injection_detector.evaluation.thresholds import operating_points, threshold_sweep


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def top_k_mean_similarity(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    top_k: int = 5,
) -> np.ndarray:
    if len(reference_embeddings) == 0:
        return np.zeros(len(query_embeddings), dtype=float)
    similarities = l2_normalize(query_embeddings) @ l2_normalize(reference_embeddings).T
    k = min(top_k, similarities.shape[1])
    top_values = np.partition(similarities, -k, axis=1)[:, -k:]
    return top_values.mean(axis=1)


def contrastive_similarity_score(
    query_embeddings: np.ndarray,
    attack_embeddings: np.ndarray,
    clean_embeddings: np.ndarray,
    top_k: int = 5,
) -> np.ndarray:
    attack_affinity = top_k_mean_similarity(query_embeddings, attack_embeddings, top_k=top_k)
    clean_affinity = top_k_mean_similarity(query_embeddings, clean_embeddings, top_k=top_k)
    return np.clip((attack_affinity - clean_affinity + 1.0) / 2.0, 0.0, 1.0)


def select_semantic_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    sweep = threshold_sweep(y_true, scores)
    candidates = sweep[sweep["precision"] >= 0.80]
    if candidates.empty:
        candidates = sweep
    row = candidates.sort_values(["recall", "f1", "precision"], ascending=False).iloc[0]
    return float(row["threshold"])


def _encode_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=float)


def train_evaluate_semantic_similarity(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_path: str | Path | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 5,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Evaluate MiniLM-style semantic similarity as an embedding detector.

    The detector is intentionally training-light: it embeds train/val/test text,
    uses train split embeddings as clean and attack references, calibrates a
    recall-first threshold on validation data, then reports test metrics.
    """

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install optional dependencies with `pip install -e .[hf]` to run MiniLM embeddings."
        ) from exc

    model = SentenceTransformer(model_name)
    train_texts = train["text"].astype(str).tolist()
    val_texts = val["text"].astype(str).tolist()
    test_texts = test["text"].astype(str).tolist()

    train_embeddings = _encode_texts(model, train_texts, batch_size)
    val_embeddings = _encode_texts(model, val_texts, batch_size)
    test_embeddings = _encode_texts(model, test_texts, batch_size)

    train_labels = train["label"].to_numpy(dtype=int)
    attack_embeddings = train_embeddings[train_labels == 1]
    clean_embeddings = train_embeddings[train_labels == 0]

    val_scores = contrastive_similarity_score(
        val_embeddings,
        attack_embeddings,
        clean_embeddings,
        top_k=top_k,
    )
    threshold = select_semantic_threshold(val["label"].to_numpy(dtype=int), val_scores)
    test_scores = contrastive_similarity_score(
        test_embeddings,
        attack_embeddings,
        clean_embeddings,
        top_k=top_k,
    )
    metrics = evaluate_binary(test["label"].to_numpy(dtype=int), test_scores, threshold)
    y_pred = (test_scores >= threshold).astype(int)
    injection_frame = test[test["label"].astype(int) == 1].copy()
    if not injection_frame.empty:
        injection_frame["detected"] = y_pred[test["label"].astype(int).to_numpy() == 1].astype(bool)
        metrics["per_category_detection_rate"] = (
            injection_frame.groupby("category")["detected"].mean().sort_index().to_dict()
        )
    else:
        metrics["per_category_detection_rate"] = {}
    metrics["model_name"] = model_name
    metrics["method"] = "embedding_semantic_similarity"
    metrics["top_k"] = int(top_k)
    metrics["reference_counts"] = {
        "attack": int(len(attack_embeddings)),
        "clean": int(len(clean_embeddings)),
    }
    metrics["validation_operating_points"] = operating_points(
        val["label"].to_numpy(dtype=int),
        val_scores,
    )
    metrics["scores"] = test_scores.tolist()
    metrics["predictions"] = y_pred.tolist()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
