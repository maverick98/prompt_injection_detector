import numpy as np

from prompt_injection_detector.data.hard_cases import build_hard_case_suite
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

