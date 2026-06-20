import numpy as np

from prompt_injection_detector.data.hard_cases import build_hard_case_suite
from prompt_injection_detector.evaluation.benchmark import write_research_summary
from prompt_injection_detector.evaluation.thresholds import operating_points, threshold_sweep


def test_hard_case_suite_has_clean_and_injection_cases():
    frame = [sample.model_dump() for sample in build_hard_case_suite()]
    labels = {row["label"] for row in frame}
    sources = {row["source"] for row in frame}
    assert labels == {0, 1}
    assert "curated_hard_clean" in sources
    assert "curated_hard_injection" in sources
    assert len(frame) >= 40


def test_threshold_sweep_and_operating_points_are_well_formed():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    sweep = threshold_sweep(y_true, scores, steps=5)
    points = operating_points(y_true, scores)
    assert {"threshold", "precision", "recall", "f1"}.issubset(sweep.columns)
    assert {"max_recall", "balanced_f1", "high_precision"} == set(points)


def test_research_summary_includes_step2_model_comparison(tmp_path):
    test_metrics = {
        "threshold": 0.4,
        "roc_auc": 0.99,
        "confusion_matrix": [[9, 1], [0, 10]],
        "selected_model": "logistic_regression",
        "classification_report": {
            "1": {"precision": 0.91, "recall": 1.0, "f1-score": 0.95},
        },
        "model_comparison": [
            {
                "model": "logistic_regression",
                "threshold": 0.4,
                "precision": 0.91,
                "recall": 1.0,
                "f1": 0.95,
                "roc_auc": 0.99,
                "false_positives": 1,
                "false_negatives": 0,
            }
        ],
        "validation_operating_points": {
            "balanced_f1": {
                "threshold": 0.4,
                "precision": 0.91,
                "recall": 1.0,
                "f1": 0.95,
                "false_positives": 1,
                "false_negatives": 0,
            }
        },
    }
    hard_metrics = {
        "hard_case_count": 2,
        "roc_auc": 1.0,
        "confusion_matrix": [[1, 0], [0, 1]],
        "classification_report": {
            "1": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0},
        },
        "best_hard_suite_threshold": {"threshold": 0.5},
        "recall_locked_hard_suite_threshold": {"threshold": 0.4},
        "errors": [],
    }

    output = write_research_summary(test_metrics, hard_metrics, tmp_path / "summary.md")
    content = output.read_text(encoding="utf-8")

    assert "Validation Baseline Comparison" in content
    assert "held-out test" in content
    assert "logistic_regression" in content
    assert "Validation Operating Points" in content
