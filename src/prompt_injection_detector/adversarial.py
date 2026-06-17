from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from prompt_injection_detector.models.classical import (
    evaluate_detector,
    train_classical_models,
)
from prompt_injection_detector.redteam.strategies import RuleBasedRedTeamGenerator, score_variants


def run_adversarial_loop(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    iterations: int = 3,
    max_false_negatives_per_round: int = 100,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir or "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = RuleBasedRedTeamGenerator()
    working_train = train.copy()
    history: list[dict[str, Any]] = []
    all_evasions: list[dict[str, Any]] = []

    for iteration in range(1, iterations + 1):
        detector = train_classical_models(working_train, val)
        metrics = evaluate_detector(detector, test)
        scores = pd.Series(metrics["scores"], index=test.index)
        injection_mask = test["label"].astype(int) == 1
        false_negative_mask = injection_mask & (scores < detector.threshold)
        false_negatives = test[false_negative_mask].head(max_false_negatives_per_round)
        if false_negatives.empty:
            detected_injections = test[injection_mask & (scores >= detector.threshold)]
            red_team_seeds = detected_injections.head(max_false_negatives_per_round)
            seed_source = "detected_injections"
        else:
            red_team_seeds = false_negatives
            seed_source = "false_negatives"

        successful_rows = []
        strategy_counts: dict[str, int] = {}
        for _, row in red_team_seeds.iterrows():
            variants = score_variants(detector, generator.generate(row["text"]))
            for variant in variants:
                record = variant.model_dump()
                record["iteration"] = iteration
                all_evasions.append(record)
                if variant.bypassed:
                    strategy_counts[variant.strategy] = strategy_counts.get(variant.strategy, 0) + 1
                    successful_rows.append(
                        {
                            "text": variant.variant_text,
                            "label": 1,
                            "category": row["category"],
                            "source": f"adversarial_round_{iteration}",
                            "split": "train",
                        }
                    )

        attack_attempts = max(len(red_team_seeds) * 5, 1)
        attack_success_rate = len(successful_rows) / attack_attempts
        history.append(
            {
                "iteration": iteration,
                "model": detector.name,
                "threshold": detector.threshold,
                "attack_success_rate": attack_success_rate,
                "false_negatives": int(false_negative_mask.sum()),
                "red_team_seed_source": seed_source,
                "red_team_seed_count": int(len(red_team_seeds)),
                "detector_recall": metrics["classification_report"]["1"]["recall"],
                "detector_f1": metrics["classification_report"]["1"]["f1-score"],
                "roc_auc": metrics["roc_auc"],
                "strategy_counts": strategy_counts,
            }
        )

        if successful_rows:
            working_train = pd.concat([working_train, pd.DataFrame(successful_rows)], ignore_index=True)

    pd.DataFrame(history).to_csv(output_dir / "adversarial_history.csv", index=False)
    if all_evasions:
        pd.DataFrame(all_evasions).to_csv(output_dir / "evasion_variants.csv", index=False)
    return {"history": history, "train_size": len(working_train)}
