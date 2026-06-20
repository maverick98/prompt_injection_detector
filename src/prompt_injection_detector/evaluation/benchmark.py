from __future__ import annotations

from pathlib import Path
from typing import Any

from prompt_injection_detector.data.hard_cases import build_hard_case_suite
from prompt_injection_detector.data.io import samples_to_frame
from prompt_injection_detector.evaluation.metrics import evaluate_binary
from prompt_injection_detector.evaluation.thresholds import threshold_sweep
from prompt_injection_detector.models.classical import score_text


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_Not available in this run. Regenerate with the latest training code._"
    header = "|" + "|".join(label for label, _ in columns) + "|"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value))
        body.append("|" + "|".join(values) + "|")
    return "\n".join([header, separator, *body])


def _format_validation_baseline_comparison(test_metrics: dict[str, Any]) -> str:
    rows = test_metrics.get("model_comparison", [])
    return _markdown_table(
        rows,
        [
            ("Model", "model"),
            ("Threshold", "threshold"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1", "f1"),
            ("ROC-AUC", "roc_auc"),
            ("FP", "false_positives"),
            ("FN", "false_negatives"),
        ],
    )


def _format_operating_points(test_metrics: dict[str, Any]) -> str:
    points = test_metrics.get("validation_operating_points", {})
    rows = [{"policy": name, **values} for name, values in points.items()]
    return _markdown_table(
        rows,
        [
            ("Policy", "policy"),
            ("Threshold", "threshold"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1", "f1"),
            ("FP", "false_positives"),
            ("FN", "false_negatives"),
        ],
    )


def evaluate_hard_cases(detector, output_dir: str | Path | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir or "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = samples_to_frame(build_hard_case_suite())
    scores = score_text(detector.pipeline, frame["text"])
    metrics = evaluate_binary(frame["label"].to_numpy(dtype=int), scores, detector.threshold)
    frame = frame.copy()
    frame["score"] = scores
    frame["predicted_label"] = (scores >= detector.threshold).astype(int)
    frame["correct"] = frame["predicted_label"] == frame["label"].astype(int)
    frame["error_type"] = ""
    frame.loc[(frame["label"] == 0) & (frame["predicted_label"] == 1), "error_type"] = "false_positive"
    frame.loc[(frame["label"] == 1) & (frame["predicted_label"] == 0), "error_type"] = "false_negative"

    sweep = threshold_sweep(frame["label"].to_numpy(dtype=int), scores)
    best_f1_row = sweep.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0].to_dict()
    recall_locked = sweep[sweep["recall"] >= 1.0]
    if recall_locked.empty:
        recall_locked = sweep[sweep["recall"] >= 0.95]
    recall_locked_row = recall_locked.sort_values(
        ["precision", "f1", "threshold"],
        ascending=False,
    ).iloc[0].to_dict()
    metrics["hard_case_count"] = len(frame)
    metrics["errors"] = frame[~frame["correct"]][
        ["text", "label", "category", "score", "predicted_label", "error_type"]
    ].to_dict(orient="records")
    metrics["best_hard_suite_threshold"] = {
        key: float(best_f1_row[key])
        for key in ["threshold", "precision", "recall", "f1", "false_positives", "false_negatives"]
    }
    metrics["recall_locked_hard_suite_threshold"] = {
        key: float(recall_locked_row[key])
        for key in ["threshold", "precision", "recall", "f1", "false_positives", "false_negatives"]
    }

    frame.to_csv(output_dir / "hard_case_predictions.csv", index=False)
    sweep.to_csv(output_dir / "hard_case_threshold_sweep.csv", index=False)
    return metrics


def write_research_summary(
    test_metrics: dict[str, Any],
    hard_metrics: dict[str, Any],
    output: str | Path,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = test_metrics["classification_report"]
    hard_report = hard_metrics["classification_report"]
    errors = hard_metrics.get("errors", [])
    error_lines = "\n".join(
        f"- `{item['error_type']}` score={item['score']:.3f}: {item['text']}" for item in errors[:10]
    )
    if not error_lines:
        error_lines = "- No hard-suite errors at the selected threshold."
    content = f"""# Prompt Injection Detector Local Evaluation Summary

## Starter Split

| Metric | Value |
|---|---:|
| Threshold | {test_metrics['threshold']:.3f} |
| ROC-AUC | {test_metrics['roc_auc']:.3f} |
| Injection precision | {report['1']['precision']:.3f} |
| Injection recall | {report['1']['recall']:.3f} |
| Injection F1 | {report['1']['f1-score']:.3f} |

Confusion matrix: `{test_metrics['confusion_matrix']}`

Selected classical model: `{test_metrics.get('selected_model', 'unknown')}`

## Validation Baseline Comparison

These rows are computed on the validation split during model selection. The
starter split section above reports the selected detector on the held-out test
split.

{_format_validation_baseline_comparison(test_metrics)}

## Validation Operating Points

{_format_operating_points(test_metrics)}

## Curated Hard Suite

| Metric | Value |
|---|---:|
| Cases | {hard_metrics['hard_case_count']} |
| ROC-AUC | {hard_metrics['roc_auc']:.3f} |
| Injection precision | {hard_report['1']['precision']:.3f} |
| Injection recall | {hard_report['1']['recall']:.3f} |
| Injection F1 | {hard_report['1']['f1-score']:.3f} |

Confusion matrix: `{hard_metrics['confusion_matrix']}`

Best threshold on the hard suite by F1:
`{hard_metrics['best_hard_suite_threshold']}`

Best threshold on the hard suite while preserving full available recall:
`{hard_metrics['recall_locked_hard_suite_threshold']}`

## Hard-Suite Errors At Selected Threshold

{error_lines}

## Interpretation

The starter split validates the end-to-end pipeline. The curated hard suite is
more important for research storytelling because it includes false-positive-like
clean prompts and subtler attacks. Use this report to discuss threshold tradeoffs,
not just headline accuracy.
"""
    output.write_text(content, encoding="utf-8")
    return output
