from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prompt_injection_detector.models.classical import score_text
from prompt_injection_detector.redteam.strategies import RuleBasedRedTeamGenerator, STRATEGIES


def _normalize_distribution(values: np.ndarray) -> list[float]:
    values = np.where(np.abs(values) < 1e-9, 0.0, values)
    total = values.sum()
    if total <= 0:
        return (np.ones_like(values) / len(values)).round(6).tolist()
    return (values / total).round(6).tolist()


def solve_zero_sum_game(payoff_matrix: np.ndarray) -> dict[str, Any]:
    """Solve a finite zero-sum game.

    Rows are attacker strategies, columns are defender threshold policies, and
    values are attacker payoffs. The attacker maximizes bypass rate; the defender
    minimizes it.
    """

    try:
        from scipy.optimize import linprog
    except ImportError:
        row_security = payoff_matrix.min(axis=1)
        col_risk = payoff_matrix.max(axis=0)
        return {
            "solver": "pure_strategy_fallback",
            "value": float(col_risk.min()),
            "attacker_mixed_strategy": _normalize_distribution(
                np.eye(payoff_matrix.shape[0])[int(row_security.argmax())]
            ),
            "defender_mixed_strategy": _normalize_distribution(
                np.eye(payoff_matrix.shape[1])[int(col_risk.argmin())]
            ),
        }

    rows, cols = payoff_matrix.shape

    # Defender LP: minimize v subject to A q <= v, sum(q)=1, q>=0.
    defender_c = np.r_[np.zeros(cols), 1.0]
    defender_a_ub = np.c_[payoff_matrix, -np.ones(rows)]
    defender_a_eq = np.r_[np.ones(cols), 0.0].reshape(1, -1)
    defender_bounds = [(0.0, 1.0)] * cols + [(None, None)]
    defender_result = linprog(
        defender_c,
        A_ub=defender_a_ub,
        b_ub=np.zeros(rows),
        A_eq=defender_a_eq,
        b_eq=np.array([1.0]),
        bounds=defender_bounds,
        method="highs",
    )

    # Attacker LP: maximize v subject to A^T p >= v, sum(p)=1, p>=0.
    attacker_c = np.r_[np.zeros(rows), -1.0]
    attacker_a_ub = np.c_[-payoff_matrix.T, np.ones(cols)]
    attacker_a_eq = np.r_[np.ones(rows), 0.0].reshape(1, -1)
    attacker_bounds = [(0.0, 1.0)] * rows + [(None, None)]
    attacker_result = linprog(
        attacker_c,
        A_ub=attacker_a_ub,
        b_ub=np.zeros(cols),
        A_eq=attacker_a_eq,
        b_eq=np.array([1.0]),
        bounds=attacker_bounds,
        method="highs",
    )

    if not defender_result.success or not attacker_result.success:
        raise RuntimeError(
            "Failed to solve zero-sum game: "
            f"defender={defender_result.message}; attacker={attacker_result.message}"
        )

    value = (float(defender_result.x[-1]) + float(attacker_result.x[-1])) / 2
    return {
        "solver": "scipy_linprog_highs",
        "value": value,
        "attacker_mixed_strategy": _normalize_distribution(attacker_result.x[:-1]),
        "defender_mixed_strategy": _normalize_distribution(defender_result.x[:-1]),
    }


def evaluate_strategy_game(
    detector,
    seed_texts: list[str],
    benign_texts: list[str],
    thresholds: list[float],
    strategies: list[str] | None = None,
    seed: int = 42,
    false_positive_weight: float = 0.25,
) -> dict[str, Any]:
    strategies = strategies or list(STRATEGIES)
    thresholds = sorted({float(threshold) for threshold in thresholds})
    generator = RuleBasedRedTeamGenerator(strategies=strategies, seed=seed)
    records: list[dict[str, Any]] = []

    for original_text in seed_texts:
        for variant in generator.generate(original_text):
            score = float(score_text(detector.pipeline, [variant.variant_text])[0])
            for threshold in thresholds:
                records.append(
                    {
                        "strategy": variant.strategy,
                        "threshold": threshold,
                        "score": score,
                        "bypassed": score < threshold,
                        "original_text": original_text,
                        "variant_text": variant.variant_text,
                    }
                )

    frame = pd.DataFrame(records)
    bypass_payoff = (
        frame.groupby(["strategy", "threshold"])["bypassed"]
        .mean()
        .unstack(fill_value=0.0)
        .reindex(index=strategies, columns=thresholds, fill_value=0.0)
    )
    benign_scores = score_text(detector.pipeline, benign_texts) if benign_texts else np.array([])
    false_positive_rates = {
        threshold: float((benign_scores >= threshold).mean()) if len(benign_scores) else 0.0
        for threshold in thresholds
    }
    payoff = bypass_payoff.copy()
    for threshold in thresholds:
        payoff[threshold] = payoff[threshold] + false_positive_weight * false_positive_rates[threshold]
    equilibrium = solve_zero_sum_game(payoff.to_numpy(dtype=float))
    result = {
        "attacker_strategies": strategies,
        "defender_thresholds": thresholds,
        "false_positive_weight": false_positive_weight,
        "false_positive_rates": false_positive_rates,
        "bypass_matrix": bypass_payoff.values.round(6).tolist(),
        "payoff_matrix": payoff.values.round(6).tolist(),
        "payoff_matrix_records": payoff.reset_index().to_dict(orient="records"),
        "equilibrium": equilibrium,
        "variant_count": int(len(frame)),
        "benign_count": int(len(benign_texts)),
    }
    return result


def write_game_outputs(result: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payoff_path = output_dir / "game_payoff_matrix.csv"
    equilibrium_path = output_dir / "game_equilibrium.json"
    report_path = output_dir / "game_theory_report.md"

    payoff = pd.DataFrame(
        result["payoff_matrix"],
        index=result["attacker_strategies"],
        columns=[str(threshold) for threshold in result["defender_thresholds"]],
    )
    payoff.index.name = "attacker_strategy"
    payoff.to_csv(payoff_path)
    equilibrium_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def markdown_table(frame: pd.DataFrame) -> str:
        table = frame.reset_index()
        headers = [str(column) for column in table.columns]
        rows = [[str(value) for value in row] for row in table.to_numpy()]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    attacker_mix = "\n".join(
        f"- `{name}`: {weight:.3f}"
        for name, weight in zip(
            result["attacker_strategies"],
            result["equilibrium"]["attacker_mixed_strategy"],
            strict=True,
        )
    )
    defender_mix = "\n".join(
        f"- threshold `{threshold:.3f}`: {weight:.3f}"
        for threshold, weight in zip(
            result["defender_thresholds"],
            result["equilibrium"]["defender_mixed_strategy"],
            strict=True,
        )
    )
    fp_lines = "\n".join(
        f"- threshold `{threshold:.3f}`: {rate:.3f}"
        for threshold, rate in result["false_positive_rates"].items()
    )
    report_path.write_text(
        f"""# Game-Theoretic Attacker/Defender Analysis

This analysis treats prompt-injection defense as a finite zero-sum game.

- Attacker actions: red-team evasion strategies.
- Defender actions: detector threshold policies.
- Defender loss: bypass rate plus `{result['false_positive_weight']:.3f}` times false-positive rate.
- Defender objective: minimize worst-case security and usability loss.

## Equilibrium

Solver: `{result['equilibrium']['solver']}`

Game value, interpreted as expected defender loss under equilibrium play:
`{result['equilibrium']['value']:.3f}`

## Attacker Mixed Strategy

{attacker_mix}

## Defender Mixed Strategy

{defender_mix}

## False-Positive Rates By Threshold

{fp_lines}

## Loss Matrix

Rows are attacker strategies. Columns are defender thresholds. Values are bypass
rate plus weighted false-positive burden, so lower is better for the defender.

{markdown_table(payoff)}
""",
        encoding="utf-8",
    )
    return {
        "payoff_matrix": payoff_path,
        "equilibrium": equilibrium_path,
        "report": report_path,
    }
