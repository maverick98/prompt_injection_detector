from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from prompt_injection_detector.models.classical import score_text
from prompt_injection_detector.research.risk_physics import (
    ATTACK_PROTOTYPE,
    CLEAN_PROTOTYPE,
    analyze_prompt,
    energy_risk,
    feature_signals,
    feature_vector,
)


ACTION_LOSSES = {
    "allow": {"attack": 10.0, "clean": 0.0},
    "review": {"attack": 1.2, "clean": 1.0},
    "quarantine": {"attack": 0.25, "clean": 2.8},
    "block": {"attack": 0.05, "clean": 5.0},
}


def _clip(value: float) -> float:
    return float(np.clip(value, 1e-6, 1.0 - 1e-6))


def _logit(value: float) -> float:
    value = _clip(value)
    return float(math.log(value / (1.0 - value)))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-value)))


def _normalize(vector: np.ndarray) -> np.ndarray:
    total = float(np.sum(vector))
    if total <= 0:
        return np.ones_like(vector, dtype=float) / len(vector)
    return vector / total


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize(np.asarray(p, dtype=float) + 1e-9)
    q = _normalize(np.asarray(q, dtype=float) + 1e-9)
    midpoint = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2(p / midpoint)))
    kl_qm = float(np.sum(q * np.log2(q / midpoint)))
    return float(0.5 * (kl_pm + kl_qm))


def bayesian_decision(
    detector_score: float | None,
    risk_energy: float,
    graph_risk: float,
    prior_attack_rate: float = 0.20,
) -> dict[str, Any]:
    detector_evidence = 0.0 if detector_score is None else 1.25 * _logit(detector_score)
    energy_evidence = 1.65 * _logit(max(risk_energy, 0.01))
    graph_evidence = 1.10 * _logit(max(graph_risk, 0.01))
    posterior = _sigmoid(_logit(prior_attack_rate) + detector_evidence + energy_evidence + graph_evidence)
    expected_losses = {
        action: posterior * losses["attack"] + (1.0 - posterior) * losses["clean"]
        for action, losses in ACTION_LOSSES.items()
    }
    action = min(expected_losses, key=expected_losses.get)
    return {
        "prior_attack_rate": prior_attack_rate,
        "posterior_attack_probability": posterior,
        "evidence": {
            "detector_log_odds": detector_evidence,
            "energy_log_odds": energy_evidence,
            "graph_log_odds": graph_evidence,
        },
        "expected_losses": expected_losses,
        "bayes_optimal_action": action,
    }


def pac_bayes_bound(
    empirical_error: float,
    sample_count: int,
    kl_complexity: float,
    delta: float = 0.05,
) -> dict[str, Any]:
    if sample_count <= 1:
        return {
            "empirical_error": empirical_error,
            "sample_count": sample_count,
            "kl_complexity": kl_complexity,
            "delta": delta,
            "bound": 1.0,
            "interpretation": "insufficient sample count for a meaningful bound",
        }
    penalty = math.sqrt((kl_complexity + math.log(2.0 * math.sqrt(sample_count) / delta)) / (2.0 * sample_count))
    bound = float(min(1.0, empirical_error + penalty))
    return {
        "empirical_error": float(empirical_error),
        "sample_count": int(sample_count),
        "kl_complexity": float(kl_complexity),
        "delta": float(delta),
        "bound": bound,
        "interpretation": "upper confidence bound on shifted-distribution error",
    }


def robust_threshold_optimization(
    y_true: np.ndarray,
    scores: np.ndarray,
    radius: float = 0.06,
    false_positive_weight: float = 0.35,
) -> dict[str, Any]:
    rows = []
    for threshold in np.linspace(0.10, 0.90, 81):
        injection = y_true == 1
        clean = y_true == 0
        robust_fn = float(np.mean((scores[injection] - radius) < threshold)) if np.any(injection) else 0.0
        robust_fp = float(np.mean((scores[clean] + radius) >= threshold)) if np.any(clean) else 0.0
        loss = robust_fn + false_positive_weight * robust_fp
        rows.append(
            {
                "threshold": float(threshold),
                "worst_case_false_negative_rate": robust_fn,
                "worst_case_false_positive_rate": robust_fp,
                "distributionally_robust_loss": float(loss),
            }
        )
    best = min(rows, key=lambda row: row["distributionally_robust_loss"])
    return {
        "radius": radius,
        "false_positive_weight": false_positive_weight,
        "best_threshold": best["threshold"],
        "best_loss": best["distributionally_robust_loss"],
        "sweep": rows,
    }


def information_bottleneck_proxy(signals: dict[str, float]) -> dict[str, Any]:
    relevant = max(
        signals["override_pressure"],
        signals["exfiltration_pressure"],
        signals["authority_hijack_pressure"],
        signals["information_leakage_pressure"],
    )
    nuisance = 0.55 * signals["obfuscation_pressure"] + 0.20 * signals["tool_abuse_pressure"]
    efficiency = float(relevant / max(relevant + nuisance, 1e-9))
    return {
        "attack_relevant_information": float(relevant),
        "nuisance_information": float(nuisance),
        "bottleneck_efficiency": efficiency,
        "interpretation": "intent-dominated" if efficiency >= 0.62 else "surface-form dominated",
    }


def minimum_description_length(signals: dict[str, float], risk_energy: float) -> dict[str, Any]:
    attack_signal = max(signals["override_pressure"], signals["exfiltration_pressure"], signals["information_leakage_pressure"])
    clean_signal = 1.0 - risk_energy
    attack_description_length = -math.log2(_clip(0.15 + 0.80 * attack_signal))
    clean_description_length = -math.log2(_clip(0.15 + 0.80 * clean_signal))
    explanation = "attack_model" if attack_description_length < clean_description_length else "clean_model"
    return {
        "attack_description_length": float(attack_description_length),
        "clean_description_length": float(clean_description_length),
        "shorter_explanation": explanation,
    }


def information_geometry(vector: np.ndarray) -> dict[str, Any]:
    attack_js = jensen_shannon(vector, ATTACK_PROTOTYPE)
    clean_js = jensen_shannon(vector, CLEAN_PROTOTYPE)
    fisher_proxy_attack = float(np.linalg.norm(np.sqrt(_normalize(vector + 1e-9)) - np.sqrt(_normalize(ATTACK_PROTOTYPE))))
    fisher_proxy_clean = float(np.linalg.norm(np.sqrt(_normalize(vector + 1e-9)) - np.sqrt(_normalize(CLEAN_PROTOTYPE))))
    return {
        "jensen_shannon_to_attack": attack_js,
        "jensen_shannon_to_clean": clean_js,
        "fisher_proxy_to_attack": fisher_proxy_attack,
        "fisher_proxy_to_clean": fisher_proxy_clean,
        "geometric_neighborhood": "attack" if attack_js < clean_js else "clean",
    }


def spectral_graph_and_percolation(signals: dict[str, float]) -> dict[str, Any]:
    nodes = ["untrusted", "instruction", "policy", "hidden_context", "tool_sink"]
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=float)
    edges = [
        (0, 1, signals["indirect_injection_pressure"]),
        (1, 2, signals["override_pressure"]),
        (2, 3, signals["exfiltration_pressure"]),
        (3, 4, signals["tool_abuse_pressure"]),
        (1, 3, signals["authority_hijack_pressure"]),
    ]
    for source, target, weight in edges:
        adjacency[source, target] = weight
        adjacency[target, source] = 0.35 * weight
    eigenvalues = np.linalg.eigvals(adjacency)
    spectral_radius = float(max(abs(value) for value in eigenvalues))
    undirected = np.maximum(adjacency, adjacency.T)
    laplacian = np.diag(undirected.sum(axis=1)) - undirected
    lap_eigs = np.sort(np.linalg.eigvalsh(laplacian))
    algebraic_connectivity = float(lap_eigs[1]) if len(lap_eigs) > 1 else 0.0
    path_probabilities = [weight for _, _, weight in edges if weight > 0.0]
    percolation_probability = float(1.0 - np.prod([1.0 - min(value, 0.99) for value in path_probabilities]))
    return {
        "nodes": nodes,
        "adjacency": adjacency.tolist(),
        "spectral_radius": spectral_radius,
        "algebraic_connectivity": algebraic_connectivity,
        "percolation_probability": percolation_probability,
        "critical": bool(spectral_radius >= 0.85 or percolation_probability >= 0.75),
    }


def mdp_control_policy(posterior_attack: float, current_action: str | None = None) -> dict[str, Any]:
    transition_risk = {
        "allow": min(1.0, posterior_attack + 0.20),
        "review": max(0.0, posterior_attack - 0.18),
        "quarantine": max(0.0, posterior_attack - 0.35),
        "block": max(0.0, posterior_attack - 0.52),
    }
    rewards = {
        action: -(ACTION_LOSSES[action]["attack"] * posterior_attack + ACTION_LOSSES[action]["clean"] * (1.0 - posterior_attack))
        - 1.5 * next_risk
        for action, next_risk in transition_risk.items()
    }
    optimal = max(rewards, key=rewards.get)
    return {
        "state": "suspected_attack" if posterior_attack >= 0.5 else "normal_or_ambiguous",
        "current_action": current_action,
        "transition_risk": transition_risk,
        "action_values": rewards,
        "optimal_policy_action": optimal,
    }


def martingale_sequential_test(turns: list[str] | None, current_energy: float) -> dict[str, Any]:
    energies = [energy_risk(feature_signals(turn))["risk_energy"] for turn in turns] if turns else [current_energy]
    baseline = 0.28
    lambda_value = 1.8
    e_value = 1.0
    trace = []
    for idx, value in enumerate(energies, start=1):
        increment = math.exp(lambda_value * (value - baseline) - 0.5 * (lambda_value**2) * 0.08)
        e_value *= increment
        trace.append({"turn": idx, "risk_energy": float(value), "e_value": float(e_value)})
    return {
        "e_value": float(e_value),
        "alarm": bool(e_value >= 20.0),
        "anytime_valid_threshold": 20.0,
        "trace": trace,
    }


def lyapunov_stability(risk_energy: float, graph_risk: float, posterior_attack: float, action: str) -> dict[str, Any]:
    current = risk_energy**2 + graph_risk**2 + posterior_attack**2
    reductions = {"allow": 0.00, "review": 0.18, "quarantine": 0.42, "block": 0.68}
    next_value = current * (1.0 - reductions.get(action, 0.0))
    return {
        "current_safety_potential": float(current),
        "post_action_safety_potential": float(next_value),
        "decreases": bool(next_value < current),
        "stable_under_action": bool(action != "allow" and next_value < current),
    }


def hidden_intent_filter(turns: list[str] | None, current_energy: float) -> dict[str, Any]:
    energies = [energy_risk(feature_signals(turn))["risk_energy"] for turn in turns] if turns else [current_energy]
    belief = 0.20
    trace = []
    for idx, value in enumerate(energies, start=1):
        prediction = 0.85 * belief + 0.05
        likelihood_attack = 0.15 + 0.80 * value
        likelihood_clean = 0.90 - 0.70 * value
        numerator = prediction * likelihood_attack
        denominator = numerator + (1.0 - prediction) * likelihood_clean
        belief = numerator / max(denominator, 1e-9)
        trace.append({"turn": idx, "risk_energy": float(value), "hidden_attack_belief": float(belief)})
    return {"hidden_attack_belief": float(belief), "trace": trace}


def causal_and_privacy_audit(text: str, signals: dict[str, float], leakage_score: float) -> dict[str, Any]:
    benign_security_context = any(
        phrase in text.lower()
        for phrase in ["explain why", "mitigation", "defense", "prevent", "security training", "best practices"]
    )
    causal_path_score = (
        signals["override_pressure"]
        * max(signals["exfiltration_pressure"], signals["authority_hijack_pressure"])
        * max(signals["tool_abuse_pressure"], 0.25)
    )
    epsilon_leakage_proxy = float(-math.log(max(1.0 - leakage_score, 1e-6)))
    return {
        "benign_security_context": benign_security_context,
        "causal_policy_violation_score": float(causal_path_score),
        "dp_epsilon_leakage_proxy": epsilon_leakage_proxy,
        "privacy_risk": "high" if epsilon_leakage_proxy >= 1.0 and not benign_security_context else "controlled",
    }


def formal_invariant_check(signals: dict[str, float], graph_risk: float) -> dict[str, Any]:
    checks = [
        {
            "invariant": "untrusted content must not override system/developer policy",
            "passed": signals["override_pressure"] < 0.45 and signals["indirect_injection_pressure"] < 0.45,
        },
        {
            "invariant": "hidden context must not flow to external sinks",
            "passed": not (signals["exfiltration_pressure"] >= 0.45 and signals["tool_abuse_pressure"] >= 0.25),
        },
        {
            "invariant": "ambiguous prompts require reduced privilege",
            "passed": graph_risk < 0.62,
        },
    ]
    return {
        "checks": checks,
        "violations": [item["invariant"] for item in checks if not item["passed"]],
        "all_passed": all(item["passed"] for item in checks),
    }


def analyze_frontier_prompt(
    text: str,
    detector_score: float | None = None,
    detector_threshold: float = 0.5,
    turns: list[str] | None = None,
) -> dict[str, Any]:
    base = analyze_prompt(text, detector_score=detector_score, detector_threshold=detector_threshold, turns=turns).to_dict()
    signals = base["feature_signals"]
    vector = feature_vector(text)
    risk_energy = base["energy_risk"]["risk_energy"]
    graph_path_risk = base["graph_risk"]["path_risk"]
    leakage = base["information_leakage"]["score"]
    bayes = bayesian_decision(detector_score, risk_energy, graph_path_risk)
    mdp = mdp_control_policy(bayes["posterior_attack_probability"], base["control_policy"]["action"])
    return {
        "risk_physics": base,
        "bayesian_decision": bayes,
        "information_bottleneck": information_bottleneck_proxy(signals),
        "minimum_description_length": minimum_description_length(signals, risk_energy),
        "information_geometry": information_geometry(vector),
        "spectral_graph_and_percolation": spectral_graph_and_percolation(signals),
        "mdp_control": mdp,
        "martingale_sequential_test": martingale_sequential_test(turns, risk_energy),
        "lyapunov_stability": lyapunov_stability(
            risk_energy,
            graph_path_risk,
            bayes["posterior_attack_probability"],
            mdp["optimal_policy_action"],
        ),
        "hidden_intent_filter": hidden_intent_filter(turns, risk_energy),
        "causal_and_privacy_audit": causal_and_privacy_audit(text, signals, leakage),
        "formal_invariants": formal_invariant_check(signals, graph_path_risk),
    }


def analyze_frontier_dataset(detector: Any, frame: pd.DataFrame, radius: float = 0.06) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Dataset frame must not be empty.")
    labels = frame["label"].to_numpy(dtype=int)
    scores = score_text(detector.pipeline, frame["text"])
    predictions = (scores >= detector.threshold).astype(int)
    empirical_error = float(np.mean(predictions != labels))
    complexity = float(math.log1p(len(getattr(detector.pipeline.named_steps["tfidf"], "get_feature_names_out")())))
    pac = pac_bayes_bound(empirical_error, len(frame), complexity)
    robust = robust_threshold_optimization(labels, scores, radius=radius)
    attack_scores = scores[labels == 1]
    clean_scores = scores[labels == 0]
    separation = float(np.mean(attack_scores) - np.mean(clean_scores)) if len(attack_scores) and len(clean_scores) else 0.0
    drift_proxy = float(np.std(scores))
    return {
        "sample_count": int(len(frame)),
        "empirical_error": empirical_error,
        "score_separation": separation,
        "score_drift_proxy": drift_proxy,
        "pac_bayes_bound": pac,
        "distributionally_robust_threshold": {
            key: value for key, value in robust.items() if key != "sweep"
        },
        "robust_threshold_sweep": robust["sweep"],
    }
