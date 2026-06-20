from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt(value: Any, digits: int = 3, default: str = "n/a") -> str:
    if value is None:
        return default
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _class_metrics(metrics: dict[str, Any], label: str = "1") -> dict[str, float]:
    report = metrics.get("classification_report", {})
    values = report.get(label, {})
    return {
        "precision": float(values.get("precision", 0.0)),
        "recall": float(values.get("recall", 0.0)),
        "f1": float(values.get("f1-score", 0.0)),
        "roc_auc": float(metrics.get("roc_auc", 0.0) or 0.0),
    }


def _metric_card(title: str, metrics: dict[str, Any], note: str) -> str:
    values = _class_metrics(metrics)
    return f"""
    <article class="metric-card">
      <div class="card-topline">{escape(title)}</div>
      <div class="metric-grid">
        <div><span>{_fmt(values["recall"])}</span><small>Injection recall</small></div>
        <div><span>{_fmt(values["precision"])}</span><small>Injection precision</small></div>
        <div><span>{_fmt(values["f1"])}</span><small>Injection F1</small></div>
        <div><span>{_fmt(values["roc_auc"])}</span><small>ROC-AUC</small></div>
      </div>
      <p>{escape(note)}</p>
    </article>
    """


def _confusion_matrix(matrix: list[list[int]] | None) -> str:
    matrix = matrix or [[0, 0], [0, 0]]
    return f"""
    <div class="confusion">
      <div></div><b>Pred clean</b><b>Pred injection</b>
      <b>Actual clean</b><span>{matrix[0][0]}</span><span>{matrix[0][1]}</span>
      <b>Actual injection</b><span>{matrix[1][0]}</span><span>{matrix[1][1]}</span>
    </div>
    """


def _category_bars(rates: dict[str, Any]) -> str:
    if not rates:
        return '<p class="muted">No per-category rates were available in this run.</p>'
    rows = []
    for name, value in sorted(rates.items()):
        pct = max(0.0, min(100.0, float(value) * 100.0))
        rows.append(
            f"""
            <div class="bar-row">
              <span>{escape(name.replace("_", " "))}</span>
              <div class="bar"><i style="width:{pct:.1f}%"></i></div>
              <strong>{pct:.1f}%</strong>
            </div>
            """
        )
    return "\n".join(rows)


def _line_chart(history: pd.DataFrame) -> str:
    if history.empty:
        return '<p class="muted">Adversarial loop history was not available.</p>'

    points = []
    labels = []
    width = 640
    height = 240
    pad = 36
    values = history["attack_success_rate"].astype(float).tolist()
    max_value = max([1.0, *values])
    for idx, value in enumerate(values):
        x = pad + idx * ((width - pad * 2) / max(1, len(values) - 1))
        y = height - pad - (value / max_value) * (height - pad * 2)
        points.append((x, y))
        labels.append(
            f'<text x="{x:.1f}" y="{height - 10}" text-anchor="middle">I{int(history.iloc[idx]["iteration"])}</text>'
        )
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dots = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7"><title>{values[i]:.3f}</title></circle>'
        for i, (x, y) in enumerate(points)
    )
    return f"""
    <svg class="line-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Attack success over adversarial iterations">
      <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" />
      <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" />
      <polyline points="{polyline}" />
      {dots}
      {''.join(labels)}
      <text x="{pad}" y="22">attack success rate</text>
    </svg>
    """


def _strategy_table(variants: pd.DataFrame) -> str:
    if variants.empty or "strategy" not in variants.columns:
        return '<p class="muted">No evasion variants were available.</p>'
    grouped = (
        variants.groupby("strategy")["bypassed"]
        .agg(total="count", bypasses="sum", attack_success_rate="mean")
        .reset_index()
        .sort_values("strategy")
    )
    rows = []
    for _, row in grouped.iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(str(row['strategy']).replace('_', ' '))}</td>"
            f"<td>{int(row['total'])}</td>"
            f"<td>{int(row['bypasses'])}</td>"
            f"<td>{float(row['attack_success_rate']) * 100:.1f}%</td>"
            "</tr>"
        )
    return """
    <table>
      <thead><tr><th>Strategy</th><th>Variants</th><th>Bypasses</th><th>Attack success</th></tr></thead>
      <tbody>
    """ + "\n".join(rows) + """
      </tbody>
    </table>
    """


def _verdict(test_metrics: dict[str, Any], hard_metrics: dict[str, Any], variants: pd.DataFrame) -> str:
    test_recall = _class_metrics(test_metrics)["recall"]
    hard_recall = _class_metrics(hard_metrics)["recall"]
    bypass_count = int(variants["bypassed"].sum()) if "bypassed" in variants.columns else 0
    if test_recall >= 0.99 and hard_recall >= 0.99 and bypass_count == 0:
        return (
            "Strong recall-first security prototype. The detector catches all evaluated attacks "
            "in the starter and hard-suite runs, and the current red-team generator found no bypasses."
        )
    if test_recall >= 0.95:
        return (
            "Promising detector with strong attack recall. Review false negatives, false positives, "
            "and successful evasion strategies before treating it as production-ready."
        )
    return (
        "Research pipeline is working, but detector recall is not yet strong enough for a security gate. "
        "Expand hard examples and retrain before submission."
    )


def _game_summary(game: dict[str, Any]) -> str:
    if not game:
        return '<p class="muted">Game-theoretic analysis was not available.</p>'
    value = game.get("equilibrium", {}).get("value")
    attackers = game.get("attacker_strategies", [])
    attacker_mix = game.get("equilibrium", {}).get("attacker_mixed_strategy", [])
    defenders = game.get("defender_thresholds", [])
    defender_mix = game.get("equilibrium", {}).get("defender_mixed_strategy", [])

    def rows(names: list[Any], weights: list[Any]) -> str:
        html_rows = []
        for name, weight in zip(names, weights):
            pct = float(weight) * 100.0
            html_rows.append(
                f"<div class='mix-row'><span>{escape(str(name))}</span><b>{pct:.1f}%</b><i style='width:{pct:.1f}%'></i></div>"
            )
        return "\n".join(html_rows)

    return f"""
    <div class="game-grid">
      <div><h3>Equilibrium Loss</h3><strong class="big-number">{_fmt(value)}</strong></div>
      <div><h3>Attacker Mix</h3>{rows(attackers, attacker_mix)}</div>
      <div><h3>Defender Mix</h3>{rows(defenders, defender_mix)}</div>
    </div>
    """


def _frontier_summary(frontier: dict[str, Any]) -> str:
    prompt = frontier.get("prompt_frontier_analysis", {})
    dataset = frontier.get("dataset_frontier_analysis", {})
    if not prompt and not dataset:
        return '<p class="muted">Frontier diagnostics were not available.</p>'
    bayes = prompt.get("bayesian_decision", {})
    mdp = prompt.get("mdp_control", {})
    pac = dataset.get("pac_bayes_bound", {})
    robust = dataset.get("distributionally_robust_threshold", {})
    items = [
        ("Bayesian posterior", _fmt(bayes.get("posterior_attack_probability"))),
        ("Bayes action", str(bayes.get("bayes_optimal_action", "n/a"))),
        ("MDP policy", str(mdp.get("optimal_policy_action", "n/a"))),
        ("PAC-Bayes-style bound", _fmt(pac.get("bound"))),
        ("Robust threshold", _fmt(robust.get("best_threshold"))),
    ]
    return "\n".join(
        f"<article class='theory-chip'><span>{escape(label)}</span><strong>{escape(value)}</strong></article>"
        for label, value in items
    )


def write_html_report(reports_dir: str | Path = "reports", output: str | Path | None = None) -> Path:
    reports_dir = Path(reports_dir)
    output = Path(output) if output else reports_dir / "prompt_injection_research_report.html"
    output.parent.mkdir(parents=True, exist_ok=True)

    test_metrics = _read_json(reports_dir / "test_metrics.json")
    hard_metrics = _read_json(reports_dir / "hard_case_metrics.json")
    minilm_metrics = _read_json(reports_dir / "minilm_semantic_metrics.json")
    transformer_metrics = _read_json(reports_dir / "transformer_metrics.json")
    game = _read_json(reports_dir / "game_equilibrium.json")
    frontier = _read_json(reports_dir / "frontier_analysis.json")
    robustness = _read_json(reports_dir / "robustness_report.json")
    history = _read_csv(reports_dir / "adversarial_history.csv")
    variants = _read_csv(reports_dir / "evasion_variants.csv")

    confusion = test_metrics.get("confusion_matrix")
    hard_confusion = hard_metrics.get("confusion_matrix")
    bypass_count = int(variants["bypassed"].sum()) if "bypassed" in variants.columns else 0
    variant_count = int(len(variants))
    false_negatives = confusion[1][0] if confusion else "n/a"
    hard_false_positives = hard_confusion[0][1] if hard_confusion else "n/a"
    model_name = escape(str(test_metrics.get("selected_model", "classical detector")))
    verdict = _verdict(test_metrics, hard_metrics, variants)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prompt Injection Detector Research Report</title>
<style>
:root {{
  --ink: #111827;
  --muted: #5b6475;
  --paper: #fffaf4;
  --panel: rgba(255,255,255,.86);
  --red: #e0242d;
  --blue: #2563eb;
  --green: #059669;
  --amber: #d97706;
  --violet: #7c3aed;
  --line: rgba(17,24,39,.12);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at 15% 10%, rgba(37,99,235,.18), transparent 28%),
    radial-gradient(circle at 80% 5%, rgba(224,36,45,.16), transparent 26%),
    linear-gradient(135deg, #fffaf4, #f8fbff 52%, #fff7ed);
}}
.shell {{ width: min(1180px, calc(100% - 36px)); margin: 0 auto; }}
.hero {{
  min-height: 88vh;
  display: grid;
  grid-template-columns: 1.05fr .95fr;
  gap: 44px;
  align-items: center;
  padding: 46px 0 28px;
}}
.eyebrow {{ color: var(--red); font-weight: 800; text-transform: uppercase; letter-spacing: .12em; font-size: 12px; }}
h1 {{ font-size: clamp(42px, 8vw, 92px); line-height: .9; margin: 14px 0 20px; letter-spacing: 0; }}
h2 {{ font-size: 28px; margin: 0 0 18px; }}
h3 {{ margin: 0 0 10px; font-size: 16px; }}
p {{ line-height: 1.62; color: var(--muted); }}
.hero p {{ font-size: 19px; max-width: 700px; }}
.verdict {{
  border-left: 6px solid var(--red);
  padding: 18px 22px;
  background: rgba(255,255,255,.72);
  box-shadow: 0 20px 60px rgba(17,24,39,.08);
  border-radius: 8px;
  margin-top: 26px;
}}
.orbital {{
  min-height: 500px;
  perspective: 1000px;
  display: grid;
  place-items: center;
}}
.architecture-svg {{
  width: 100%;
  max-width: 560px;
  filter: drop-shadow(0 28px 42px rgba(17,24,39,.18));
  transform: rotateX(10deg) rotateY(-14deg);
  animation: float3d 7s ease-in-out infinite alternate;
}}
@keyframes float3d {{
  from {{ transform: rotateX(8deg) rotateY(-15deg) translateY(0); }}
  to {{ transform: rotateX(14deg) rotateY(-7deg) translateY(-16px); }}
}}
.flow-path {{ stroke-dasharray: 12 9; animation: dash 10s linear infinite; }}
@keyframes dash {{ to {{ stroke-dashoffset: -220; }} }}
.section {{ padding: 46px 0; }}
.grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }}
.grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
.metric-card, .panel, .theory-chip {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 18px 42px rgba(17,24,39,.07);
  backdrop-filter: blur(10px);
}}
.card-topline {{ font-weight: 900; color: var(--red); margin-bottom: 14px; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
.metric-grid div {{ border-top: 1px solid var(--line); padding-top: 12px; }}
.metric-grid span {{ display: block; font-size: 32px; font-weight: 900; }}
.metric-grid small {{ color: var(--muted); }}
.stat-strip {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 24px; }}
.stat-strip article {{ padding: 18px; border-radius: 8px; background: #111827; color: white; }}
.stat-strip span {{ display: block; color: #d1d5db; font-size: 12px; text-transform: uppercase; font-weight: 800; }}
.stat-strip strong {{ display: block; font-size: 34px; margin-top: 8px; }}
.confusion {{ display: grid; grid-template-columns: 1.1fr repeat(2, 1fr); gap: 8px; }}
.confusion b, .confusion span {{ padding: 14px; border-radius: 7px; background: white; border: 1px solid var(--line); text-align: center; }}
.confusion span {{ font-size: 26px; font-weight: 900; }}
.bar-row {{ display: grid; grid-template-columns: 190px 1fr 70px; gap: 12px; align-items: center; margin: 12px 0; }}
.bar {{ height: 14px; background: rgba(17,24,39,.1); border-radius: 99px; overflow: hidden; }}
.bar i {{ display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--red), var(--amber), var(--green)); }}
.line-chart {{ width: 100%; min-height: 220px; }}
.line-chart line {{ stroke: rgba(17,24,39,.25); }}
.line-chart polyline {{ fill: none; stroke: var(--blue); stroke-width: 6; stroke-linecap: round; stroke-linejoin: round; }}
.line-chart circle {{ fill: var(--red); stroke: white; stroke-width: 4; }}
.line-chart text {{ fill: var(--muted); font-size: 13px; font-weight: 800; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; }}
th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; }}
th {{ background: #111827; color: white; }}
.game-grid {{ display: grid; grid-template-columns: .7fr 1fr 1fr; gap: 16px; }}
.big-number {{ font-size: 50px; color: var(--red); }}
.mix-row {{ position: relative; display: grid; grid-template-columns: 1fr 70px; gap: 10px; padding: 10px 0; border-bottom: 1px solid var(--line); }}
.mix-row i {{ position: absolute; left: 0; bottom: 0; height: 3px; background: var(--blue); }}
.theory-list {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }}
.theory-chip span {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; font-weight: 900; }}
.theory-chip strong {{ display: block; margin-top: 9px; font-size: 20px; }}
.muted {{ color: var(--muted); }}
.footer {{ padding: 34px 0 52px; color: var(--muted); }}
@media (max-width: 900px) {{
  .hero, .grid-2, .grid-3, .game-grid, .stat-strip, .theory-list {{ grid-template-columns: 1fr; }}
  .orbital {{ min-height: 320px; }}
  .bar-row {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main class="shell">
  <section class="hero">
    <div>
      <div class="eyebrow">Prompt Injection Detector</div>
      <h1>Adaptive LLM Security Research Report</h1>
      <p>A recall-first detector, red-team generator, adversarial loop, semantic baseline, transformer comparison, hard-suite benchmark, game-theoretic threshold policy, and frontier mathematical diagnostics in one reproducible Colab artifact.</p>
      <div class="verdict"><strong>Verdict:</strong> {escape(verdict)}</div>
      <div class="stat-strip">
        <article><span>Selected model</span><strong>{model_name}</strong></article>
        <article><span>False negatives</span><strong>{false_negatives}</strong></article>
        <article><span>Red-team bypasses</span><strong>{bypass_count}</strong></article>
        <article><span>Hard-suite FP</span><strong>{hard_false_positives}</strong></article>
      </div>
    </div>
    <div class="orbital">
      <svg class="architecture-svg" viewBox="0 0 620 520" role="img" aria-label="Animated prompt injection detector architecture">
        <defs>
          <linearGradient id="hot" x1="0" x2="1"><stop stop-color="#e0242d"/><stop offset=".55" stop-color="#f59e0b"/><stop offset="1" stop-color="#059669"/></linearGradient>
          <filter id="glow"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        <rect x="18" y="24" width="584" height="470" rx="28" fill="rgba(255,255,255,.78)" stroke="rgba(17,24,39,.16)"/>
        <path class="flow-path" d="M130 260 C130 110 490 110 490 260 C490 410 130 410 130 260" fill="none" stroke="url(#hot)" stroke-width="8" filter="url(#glow)"/>
        <g font-family="Inter,Arial" font-weight="800" text-anchor="middle">
          <g transform="translate(130 260)"><circle r="72" fill="#111827"/><text y="-8" fill="white">Dataset</text><text y="18" fill="#d1d5db" font-size="14">public + synthetic</text></g>
          <g transform="translate(310 112)"><circle r="74" fill="#2563eb"/><text y="-8" fill="white">Detector</text><text y="18" fill="#dbeafe" font-size="14">TF-IDF + MiniLM</text></g>
          <g transform="translate(490 260)"><circle r="72" fill="#e0242d"/><text y="-8" fill="white">Red Team</text><text y="18" fill="#fee2e2" font-size="14">evasions</text></g>
          <g transform="translate(310 408)"><circle r="74" fill="#059669"/><text y="-8" fill="white">Retrain</text><text y="18" fill="#d1fae5" font-size="14">hard examples</text></g>
          <g transform="translate(310 260)"><circle r="58" fill="#fffaf4" stroke="#111827" stroke-width="4"/><text y="-8" fill="#111827">Metrics</text><text y="18" fill="#5b6475" font-size="13">recall first</text></g>
        </g>
      </svg>
    </div>
  </section>

  <section class="section">
    <h2>Model Scoreboard</h2>
    <div class="grid-3">
      {_metric_card("Classical TF-IDF", test_metrics, "Primary recall-first guardrail selected by validation recall and F1.")}
      {_metric_card("MiniLM Semantic Similarity", minilm_metrics, "Dense embedding baseline using all-MiniLM-L6-v2 nearest-reference affinity.")}
      {_metric_card("Transformer", transformer_metrics, "Optional DistilBERT or RoBERTa supervised classifier when Colab GPU is available.")}
    </div>
  </section>

  <section class="section grid-2">
    <div class="panel">
      <h2>Held-Out Confusion Matrix</h2>
      {_confusion_matrix(confusion)}
    </div>
    <div class="panel">
      <h2>Per-Category Detection</h2>
      {_category_bars(test_metrics.get("per_category_detection_rate", {}))}
    </div>
  </section>

  <section class="section grid-2">
    <div class="panel">
      <h2>Adversarial Loop</h2>
      <p>The attacker generates variants and the detector is evaluated across iterations. Lower attack success is better.</p>
      {_line_chart(history)}
    </div>
    <div class="panel">
      <h2>Evasion Diversity</h2>
      <p>Total variants generated: <strong>{variant_count}</strong>. Successful bypasses: <strong>{bypass_count}</strong>.</p>
      {_strategy_table(variants)}
    </div>
  </section>

  <section class="section grid-2">
    <div class="panel">
      <h2>Curated Hard Suite</h2>
      <p>This suite stresses benign security-adjacent prompts and subtle attacks. It is the most honest local diagnostic for threshold tradeoffs.</p>
      {_confusion_matrix(hard_confusion)}
      <div class="bar-row"><span>Hard-suite injection recall</span><div class="bar"><i style="width:{_class_metrics(hard_metrics)["recall"] * 100:.1f}%"></i></div><strong>{_class_metrics(hard_metrics)["recall"] * 100:.1f}%</strong></div>
      <div class="bar-row"><span>Hard-suite injection precision</span><div class="bar"><i style="width:{_class_metrics(hard_metrics)["precision"] * 100:.1f}%"></i></div><strong>{_class_metrics(hard_metrics)["precision"] * 100:.1f}%</strong></div>
    </div>
    <div class="panel">
      <h2>Robustness Snapshot</h2>
      {_category_bars(robustness.get("edge_case_detection_rate", {}))}
    </div>
  </section>

  <section class="section">
    <h2>Game-Theoretic Defender Policy</h2>
    <div class="panel">{_game_summary(game)}</div>
  </section>

  <section class="section">
    <h2>Frontier Mathematical Diagnostics</h2>
    <div class="theory-list">{_frontier_summary(frontier)}</div>
  </section>

  <section class="footer">
    Generated from artifacts in <code>{escape(str(reports_dir))}</code>. This HTML file is self-contained and suitable for upload, review, or submission appendix material.
  </section>
</main>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    return output
