from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


LEXICONS = {
    "override": {
        "ignore",
        "disregard",
        "forget",
        "override",
        "bypass",
        "disable",
        "do not follow",
        "previous instructions",
        "new instructions",
    },
    "exfiltration": {
        "reveal",
        "print",
        "show",
        "dump",
        "exfiltrate",
        "secret",
        "secrets",
        "password",
        "api key",
        "token",
        "credentials",
        "environment variable",
        "system prompt",
        "hidden prompt",
        "private context",
    },
    "tool_abuse": {
        "tool",
        "function",
        "delete",
        "send",
        "email",
        "upload",
        "download",
        "shell",
        "execute",
        "curl",
        "post",
    },
    "authority_hijack": {
        "developer mode",
        "admin",
        "administrator",
        "root",
        "system administrator",
        "you are now",
        "act as",
        "highest priority",
        "policy update",
    },
    "indirect": {
        "web page",
        "retrieved document",
        "document says",
        "browser content",
        "assistant reading",
        "rag",
        "knowledge base",
    },
}

ATTACK_PROTOTYPE = np.array([0.85, 0.85, 0.55, 0.55, 0.70, 0.55, 0.75], dtype=float)
CLEAN_PROTOTYPE = np.array([0.05, 0.03, 0.05, 0.10, 0.04, 0.03, 0.08], dtype=float)
ENERGY_WEIGHTS = np.array([1.4, 1.6, 1.1, 0.7, 1.0, 0.8, 1.2], dtype=float)
BASE64_RE = re.compile(r"\b[A-Za-z0-9+/]{28,}={0,2}\b")


@dataclass(frozen=True)
class RiskPhysicsReport:
    text: str
    detector_score: float | None
    detector_threshold: float
    information_leakage: dict[str, Any]
    feature_signals: dict[str, float]
    energy_risk: dict[str, Any]
    phase_transition: dict[str, Any]
    conformal_uncertainty: dict[str, Any]
    sequential_detection: dict[str, Any]
    graph_risk: dict[str, Any]
    optimal_transport: dict[str, Any]
    control_policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "detector_score": self.detector_score,
            "detector_threshold": self.detector_threshold,
            "information_leakage": self.information_leakage,
            "feature_signals": self.feature_signals,
            "energy_risk": self.energy_risk,
            "phase_transition": self.phase_transition,
            "conformal_uncertainty": self.conformal_uncertainty,
            "sequential_detection": self.sequential_detection,
            "graph_risk": self.graph_risk,
            "optimal_transport": self.optimal_transport,
            "control_policy": self.control_policy,
        }


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return float(-sum((count / length) * math.log2(count / length) for count in counts.values()))


def normalized_entropy(text: str) -> float:
    if len(text) <= 1:
        return 0.0
    alphabet = max(len(set(text)), 2)
    return float(min(shannon_entropy(text) / math.log2(alphabet), 1.0))


def compression_pressure(text: str) -> float:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 4:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    punctuation_ratio = sum(1 for char in text if not char.isalnum() and not char.isspace()) / max(len(text), 1)
    return float(np.clip(0.55 * unique_ratio + 0.45 * min(punctuation_ratio * 5.0, 1.0), 0.0, 1.0))


def keyword_score(text: str, terms: set[str]) -> float:
    lowered = text.lower()
    hits = sum(1 for term in terms if term in lowered)
    return float(1.0 - math.exp(-hits / 2.0))


def obfuscation_score(text: str) -> float:
    if not text:
        return 0.0
    spaced_letters = len(re.findall(r"(?:\b[a-zA-Z]\s+){3,}[a-zA-Z]\b", text))
    encoded = len(BASE64_RE.findall(text))
    non_ascii_ratio = sum(1 for char in text if ord(char) > 127) / max(len(text), 1)
    entropy = normalized_entropy(text)
    score = 0.35 * min(spaced_letters, 1) + 0.35 * min(encoded, 1) + 0.2 * min(non_ascii_ratio * 8, 1) + 0.1 * entropy
    return float(np.clip(score, 0.0, 1.0))


def feature_vector(text: str) -> np.ndarray:
    signals = feature_signals(text)
    return np.array(
        [
            signals["override_pressure"],
            signals["exfiltration_pressure"],
            signals["tool_abuse_pressure"],
            signals["obfuscation_pressure"],
            signals["authority_hijack_pressure"],
            signals["indirect_injection_pressure"],
            signals["information_leakage_pressure"],
        ],
        dtype=float,
    )


def feature_signals(text: str) -> dict[str, float]:
    exfiltration = keyword_score(text, LEXICONS["exfiltration"])
    signals = {
        "override_pressure": keyword_score(text, LEXICONS["override"]),
        "exfiltration_pressure": exfiltration,
        "tool_abuse_pressure": keyword_score(text, LEXICONS["tool_abuse"]),
        "obfuscation_pressure": obfuscation_score(text),
        "authority_hijack_pressure": keyword_score(text, LEXICONS["authority_hijack"]),
        "indirect_injection_pressure": keyword_score(text, LEXICONS["indirect"]),
        "information_leakage_pressure": information_leakage_score(text, exfiltration),
    }
    return signals


def information_leakage_score(text: str, exfiltration: float | None = None) -> float:
    exfiltration = keyword_score(text, LEXICONS["exfiltration"]) if exfiltration is None else exfiltration
    hidden_state_terms = keyword_score(
        text,
        {
            "hidden",
            "private",
            "confidential",
            "internal",
            "memory",
            "context",
            "policy",
            "system",
            "developer",
        },
    )
    entropy_term = 0.3 * normalized_entropy(text)
    return float(np.clip(0.55 * exfiltration + 0.35 * hidden_state_terms + 0.10 * entropy_term, 0.0, 1.0))


def information_leakage(text: str) -> dict[str, Any]:
    score = information_leakage_score(text)
    return {
        "score": score,
        "entropy_bits_per_char": shannon_entropy(text),
        "normalized_entropy": normalized_entropy(text),
        "compression_pressure": compression_pressure(text),
        "interpretation": "high leakage intent" if score >= 0.62 else "low/moderate leakage intent",
    }


def energy_risk(signals: dict[str, float]) -> dict[str, Any]:
    values = np.array(list(signals.values()), dtype=float)
    energy = float(np.dot(values, ENERGY_WEIGHTS) / ENERGY_WEIGHTS.sum())
    free_energy = float(-math.log(max(1.0 - energy, 1e-6)))
    return {
        "risk_energy": energy,
        "free_energy": free_energy,
        "dominant_signal": max(signals, key=signals.get),
    }


def phase_transition(energy: float) -> dict[str, Any]:
    if energy >= 0.62:
        phase = "high-risk phase"
        action = "block or quarantine"
    elif energy >= 0.35:
        phase = "critical transition band"
        action = "review or require stronger guardrails"
    else:
        phase = "low-risk phase"
        action = "allow with normal monitoring"
    distance = min(abs(energy - 0.35), abs(energy - 0.62))
    return {
        "phase": phase,
        "critical_thresholds": [0.35, 0.62],
        "distance_to_nearest_boundary": float(distance),
        "recommended_action": action,
    }


def conformal_uncertainty(
    energy: float,
    detector_score: float | None,
    detector_threshold: float,
) -> dict[str, Any]:
    energy_margin = abs(energy - 0.5)
    detector_margin = 0.0 if detector_score is None else abs(detector_score - detector_threshold)
    margin = 0.45 * energy_margin + 0.55 * detector_margin
    uncertainty = float(np.clip(1.0 - margin / 0.35, 0.0, 1.0))
    near_classifier_boundary = detector_score is not None and detector_margin <= 0.05
    near_energy_boundary = min(abs(energy - 0.35), abs(energy - 0.62)) <= 0.06
    ambiguous = uncertainty >= 0.55 or near_classifier_boundary or near_energy_boundary
    prediction_set = ["clean", "injection"] if ambiguous else (
        ["injection"] if energy >= 0.5 or (detector_score is not None and detector_score >= detector_threshold) else ["clean"]
    )
    return {
        "uncertainty": uncertainty,
        "prediction_set": prediction_set,
        "decision_region": "ambiguous" if ambiguous else "confident",
        "near_classifier_boundary": near_classifier_boundary,
        "near_energy_boundary": near_energy_boundary,
    }


def sequential_detection(turns: list[str] | None, current_energy: float) -> dict[str, Any]:
    sequence = turns or []
    energies = [energy_risk(feature_signals(turn))["risk_energy"] for turn in sequence]
    if turns is None or not energies:
        energies.append(current_energy)
    baseline = 0.28
    cusum = 0.0
    states = []
    for idx, value in enumerate(energies, start=1):
        cusum = max(0.0, 0.72 * cusum + value - baseline)
        states.append({"turn": idx, "risk_energy": float(value), "cusum": float(cusum)})
    return {
        "turn_count": len(states),
        "cusum": float(cusum),
        "alert": bool(cusum >= 0.75),
        "states": states,
    }


def graph_risk(text: str, signals: dict[str, float]) -> dict[str, Any]:
    edges = []
    if signals["indirect_injection_pressure"] > 0.1:
        edges.append(("untrusted_content", "instruction_override", signals["indirect_injection_pressure"]))
    if signals["override_pressure"] > 0.1:
        edges.append(("instruction_override", "policy_boundary", signals["override_pressure"]))
    if signals["exfiltration_pressure"] > 0.1:
        edges.append(("policy_boundary", "hidden_context", signals["exfiltration_pressure"]))
    if signals["tool_abuse_pressure"] > 0.1:
        edges.append(("hidden_context", "external_tool_or_sink", signals["tool_abuse_pressure"]))
    if not edges:
        return {"path_risk": 0.0, "dangerous_paths": [], "edge_count": 0}
    path_risk = float(np.clip(sum(edge[2] for edge in edges) / max(len(edges), 1), 0.0, 1.0))
    dangerous_paths = [" -> ".join([edge[0], edge[1]]) for edge in edges]
    if "email" in text.lower() or "send" in text.lower() or "upload" in text.lower():
        dangerous_paths.append("hidden_context -> external_sink")
        path_risk = float(np.clip(path_risk + 0.15, 0.0, 1.0))
    return {"path_risk": path_risk, "dangerous_paths": dangerous_paths, "edge_count": len(edges)}


def optimal_transport_distance(vector: np.ndarray) -> dict[str, Any]:
    attack_distance = float(np.mean(np.abs(vector - ATTACK_PROTOTYPE)))
    clean_distance = float(np.mean(np.abs(vector - CLEAN_PROTOTYPE)))
    attack_affinity = float(clean_distance / max(clean_distance + attack_distance, 1e-9))
    return {
        "clean_distance": clean_distance,
        "attack_distance": attack_distance,
        "attack_affinity": attack_affinity,
        "nearest_distribution": "attack" if attack_distance < clean_distance else "clean",
    }


def control_policy(energy: float, uncertainty: float, graph_path_risk: float) -> dict[str, Any]:
    aggregate = max(energy, 0.85 * graph_path_risk, 0.55 * uncertainty)
    if aggregate >= 0.72:
        action = "block"
        guardrail = "deny execution, log full trace, and route to security review"
    elif aggregate >= 0.55:
        action = "quarantine"
        guardrail = "strip tool access and require human approval"
    elif aggregate >= 0.38:
        action = "review"
        guardrail = "allow only with reduced privileges and stronger output filtering"
    else:
        action = "allow"
        guardrail = "normal execution with telemetry"
    return {
        "aggregate_control_signal": float(aggregate),
        "action": action,
        "guardrail": guardrail,
    }


def analyze_prompt(
    text: str,
    detector_score: float | None = None,
    detector_threshold: float = 0.5,
    turns: list[str] | None = None,
) -> RiskPhysicsReport:
    signals = feature_signals(text)
    leakage = information_leakage(text)
    energy = energy_risk(signals)
    phase = phase_transition(energy["risk_energy"])
    uncertainty = conformal_uncertainty(energy["risk_energy"], detector_score, detector_threshold)
    sequence = sequential_detection(turns, energy["risk_energy"])
    graph = graph_risk(text, signals)
    transport = optimal_transport_distance(feature_vector(text))
    policy = control_policy(
        energy["risk_energy"],
        uncertainty["uncertainty"],
        graph["path_risk"],
    )
    return RiskPhysicsReport(
        text=text,
        detector_score=detector_score,
        detector_threshold=detector_threshold,
        information_leakage=leakage,
        feature_signals=signals,
        energy_risk=energy,
        phase_transition=phase,
        conformal_uncertainty=uncertainty,
        sequential_detection=sequence,
        graph_risk=graph,
        optimal_transport=transport,
        control_policy=policy,
    )
