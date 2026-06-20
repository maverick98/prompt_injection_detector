import json

import pandas as pd

from prompt_injection_detector.evaluation.html_report import write_html_report


def test_write_html_report_creates_self_contained_artifact(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    metrics = {
        "threshold": 0.2,
        "selected_model": "logistic_regression",
        "roc_auc": 0.99,
        "confusion_matrix": [[8, 2], [0, 10]],
        "classification_report": {
            "0": {"precision": 1.0, "recall": 0.8, "f1-score": 0.89},
            "1": {"precision": 0.83, "recall": 1.0, "f1-score": 0.91},
        },
        "per_category_detection_rate": {"role_override": 1.0},
    }
    (reports / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (reports / "hard_case_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    pd.DataFrame(
        [
            {"iteration": 1, "attack_success_rate": 0.2},
            {"iteration": 2, "attack_success_rate": 0.1},
        ]
    ).to_csv(reports / "adversarial_history.csv", index=False)
    pd.DataFrame(
        [
            {"strategy": "paraphrase", "bypassed": False},
            {"strategy": "encoding", "bypassed": True},
        ]
    ).to_csv(reports / "evasion_variants.csv", index=False)

    output = write_html_report(reports, reports / "report.html")
    content = output.read_text(encoding="utf-8")

    assert "Adaptive LLM Security Research Report" in content
    assert "architecture-svg" in content
    assert "@keyframes" in content
    assert "Model Scoreboard" in content
    assert "Game-Theoretic Defender Policy" in content
