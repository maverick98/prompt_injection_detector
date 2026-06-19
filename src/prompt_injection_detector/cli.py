from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from rich import print

from prompt_injection_detector.adversarial import run_adversarial_loop
from prompt_injection_detector.evaluation.benchmark import evaluate_hard_cases, write_research_summary
from prompt_injection_detector.data.io import load_frame, save_samples, stratified_split
from prompt_injection_detector.data.synthetic import generate_synthetic_dataset
from prompt_injection_detector.models.classical import (
    evaluate_detector,
    load_detector,
    save_detector,
    train_classical_models,
)
from prompt_injection_detector.redteam.strategies import RuleBasedRedTeamGenerator, score_variants
from prompt_injection_detector.robustness import run_robustness_suite

app = typer.Typer(help="Prompt injection detector research toolkit.")


@app.command()
def build_dataset(
    output: Path = typer.Option(Path("data/processed/dataset.csv"), help="Output CSV or JSONL path."),
    injection_samples: int = 750,
    clean_samples: int = 750,
    seed: int = 42,
) -> None:
    samples = generate_synthetic_dataset(injection_samples, clean_samples, seed)
    path = save_samples(samples, output)
    frame = stratified_split(load_frame(path), random_state=seed)
    frame.to_csv(path, index=False)
    print(f"[green]Wrote {len(frame)} samples to {path}[/green]")


@app.command()
def train(
    dataset: Path = typer.Option(Path("data/processed/dataset.csv")),
    model_out: Path = typer.Option(Path("artifacts/detector.joblib")),
    metrics_out: Path = typer.Option(Path("reports/test_metrics.json")),
    decision_threshold: float | None = typer.Option(
        None,
        help="Optional fixed decision threshold. If omitted, validation calibration is used.",
    ),
) -> None:
    frame = load_frame(dataset)
    if "split" not in frame.columns or frame["split"].isna().any():
        frame = stratified_split(frame)
    train_frame = frame[frame["split"] == "train"]
    val_frame = frame[frame["split"] == "val"]
    test_frame = frame[frame["split"] == "test"]
    detector = train_classical_models(train_frame, val_frame, decision_threshold=decision_threshold)
    metrics = evaluate_detector(detector, test_frame)
    metrics["validation_operating_points"] = detector.metrics.get("validation_operating_points", {})
    save_detector(detector, model_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[green]Saved {detector.name} detector to {model_out}[/green]")
    print(f"Recall: {metrics['classification_report']['1']['recall']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")


@app.command()
def predict(
    text: str,
    model_path: Path = typer.Option(Path("artifacts/detector.joblib")),
) -> None:
    detector = load_detector(model_path)
    print(detector.predict_one(text).model_dump())


@app.command()
def redteam(
    text: str,
    model_path: Path = typer.Option(Path("artifacts/detector.joblib")),
) -> None:
    detector = load_detector(model_path)
    variants = score_variants(detector, RuleBasedRedTeamGenerator().generate(text))
    for variant in variants:
        print(variant.model_dump())


@app.command()
def loop(
    dataset: Path = typer.Option(Path("data/processed/dataset.csv")),
    iterations: int = 3,
    output_dir: Path = typer.Option(Path("reports")),
) -> None:
    frame = load_frame(dataset)
    if "split" not in frame.columns or frame["split"].isna().any():
        frame = stratified_split(frame)
    result = run_adversarial_loop(
        frame[frame["split"] == "train"],
        frame[frame["split"] == "val"],
        frame[frame["split"] == "test"],
        iterations=iterations,
        output_dir=output_dir,
    )
    print(json.dumps(result, indent=2))


@app.command()
def robust(
    dataset: Path = typer.Option(Path("data/processed/dataset.csv")),
    model_path: Path = typer.Option(Path("artifacts/detector.joblib")),
    output: Path = typer.Option(Path("reports/robustness_report.json")),
) -> None:
    detector = load_detector(model_path)
    frame = load_frame(dataset)
    test_frame = frame[frame["split"] == "test"] if "split" in frame.columns else frame
    result = run_robustness_suite(detector, test_frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


@app.command()
def benchmark(
    dataset: Path = typer.Option(Path("data/processed/dataset.csv")),
    model_path: Path = typer.Option(Path("artifacts/detector.joblib")),
    output_dir: Path = typer.Option(Path("reports")),
) -> None:
    """Run the curated hard-suite evaluation and write a summary report."""

    detector = load_detector(model_path)
    frame = load_frame(dataset)
    test_frame = frame[frame["split"] == "test"] if "split" in frame.columns else frame
    test_metrics = evaluate_detector(detector, test_frame)
    hard_metrics = evaluate_hard_cases(detector, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hard_case_metrics.json").write_text(
        json.dumps(hard_metrics, indent=2),
        encoding="utf-8",
    )
    report_path = write_research_summary(
        test_metrics,
        hard_metrics,
        output_dir / "local_evaluation_summary.md",
    )
    print(f"[green]Wrote hard-suite metrics to {output_dir / 'hard_case_metrics.json'}[/green]")
    print(f"[green]Wrote hard-suite predictions to {output_dir / 'hard_case_predictions.csv'}[/green]")
    print(f"[green]Wrote threshold sweep to {output_dir / 'hard_case_threshold_sweep.csv'}[/green]")
    print(f"[green]Wrote research summary to {report_path}[/green]")
    print(
        "Hard-suite injection recall: "
        f"{hard_metrics['classification_report']['1']['recall']:.3f}; "
        f"precision: {hard_metrics['classification_report']['1']['precision']:.3f}"
    )


@app.command()
def export_hf_data_card(output: Path = typer.Option(Path("reports/hf_data_card.md"))) -> None:
    content = """# Prompt Injection Detector Dataset

This dataset contains labeled prompt-injection and clean user-input examples for
training and evaluating prompt-injection detectors.

## Labels

- `label=1`: prompt injection attempt
- `label=0`: clean user input

## Injection Categories

- role_override
- instruction_smuggling
- data_extraction
- jailbreak
- indirect_injection

## Intended Use

Research, benchmarking, and defensive model development for LLM application security.
The starter data generated by this repository is synthetic and should be augmented
with public datasets, responsibly gathered examples, and manual review before any
production use.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"[green]Wrote {output}[/green]")


if __name__ == "__main__":
    app()
