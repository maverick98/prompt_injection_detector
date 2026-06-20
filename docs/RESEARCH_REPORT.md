# Prompt Injection Detector Research Report

## Executive Summary

This project implements an end-to-end defensive research pipeline for prompt
injection: a detector, a red-team generator, an adversarial loop, robustness
tests, and a Streamlit demo.

The local implementation is complete and runnable. It uses a synthetic 1,500-row
starter dataset so the full system can be exercised without external services.
The next research step is to augment the dataset with public, manual, and
LLM-reviewed examples, then rerun the same pipeline.

## Problem

Prompt injection attempts try to override system instructions, extract private
context, jailbreak policies, or hide malicious instructions inside external
content. The project treats this as a security classification problem where
recall matters more than precision: a missed attack is more costly than a false
alarm.

## System Design

The system has five layers:

1. Dataset builder for clean and injected prompts.
2. Recall-optimized detector training and evaluation, including lexical,
   semantic-similarity, and transformer paths.
3. Red-team generator that mutates attacks with five strategies.
4. Adversarial loop that feeds successful evasions back into training.
5. Robustness and demo layer for category-level inspection.

## Dataset

The local starter dataset contains:

- 1,500 total samples
- 750 prompt-injection attempts
- 750 clean prompts
- balanced coverage across five injection categories
- 70/15/15 train/validation/test split

Categories:

- role override
- instruction smuggling
- data extraction
- jailbreak
- indirect injection

## Detector

The baseline compares:

- Logistic Regression
- RBF SVM with calibrated probabilities
- Random Forest

The project now also includes a MiniLM semantic-similarity detector:

- embeds prompts with `sentence-transformers/all-MiniLM-L6-v2`
- compares each prompt to known clean and injection reference examples
- calibrates a recall-first threshold on validation data
- writes `reports/minilm_semantic_metrics.json`

This adds a semantic retrieval-style signal beside the lexical TF-IDF detector
and the optional fine-tuned DistilBERT/RoBERTa classifier.

The selected local model is Logistic Regression with a decision threshold of
`0.13`, chosen to prioritize injection recall.

## Evaluation

The detector is evaluated as a security classifier, not as a generic text
classifier. The operating policy is recall-first: false negatives are treated as
the highest-risk error because they allow malicious prompts to reach the LLM.
False positives are still measured because they create user friction, but they
are less costly than missed attacks.

Required metric reporting:

| Metric | Why it matters |
|---|---|
| Precision per class | Measures how many flagged prompts are truly malicious, and how trustworthy clean predictions are. |
| Recall per class | Measures how many real attacks are caught. Injection recall is the primary security metric. |
| F1 per class | Summarizes the precision/recall tradeoff without hiding the individual values. |
| ROC-AUC | Measures how well the detector ranks attacks above clean prompts across thresholds. |
| Confusion matrix | Exposes false negatives and false positives directly. |
| Per-category detection rate | Shows which attack family is hardest: role override, instruction smuggling, data extraction, jailbreak, or indirect injection. |

Current local metrics on the generated starter split:

| Metric | Value |
|---|---:|
| Injection precision | 1.000 |
| Injection recall | 1.000 |
| Injection F1 | 1.000 |
| ROC-AUC | 1.000 |
| Accuracy | 1.000 |

Confusion matrix:

```text
[[112,   0],
 [  0, 113]]
```

The detector produces no false negatives on the starter split. This validates the
pipeline, but also shows why harder real-world examples are needed for a final
research claim.

## Red-Team Generator

The generator implements five evasion families:

- paraphrasing
- obfuscation
- role-play wrapping
- encoding tricks
- nested injection in benign context

Each variant is scored by the detector and marked as bypassed or blocked. The
default generator is deterministic for reproducible local and Colab runs. The
project also implements the problem statement's LLM-backed mode through
`pid redteam --provider gemini|cohere|openai|groq --chain-strategies`, reading
API keys from runtime environment variables such as `GEMINI_API_KEY` or
`COHERE_API_KEY`. The Colab notebook loads those values from Colab Secrets, not
from source code.

## Adversarial Loop

The loop runs for three iterations. When false negatives exist, they become the
primary red-team seeds. When the starter split has no false negatives, the loop
stress-tests already detected injections to exercise the attacker-defender
machinery.

Current local loop result:

- attack success rate: `0.0` in all three iterations
- detector recall: `1.0` in all three iterations
- red-team seed source: detected injections, because there were no false negatives

This is expected on the template-driven starter data. After adding public and
manual examples, the same loop should produce a more meaningful attack-success
curve.

## Game-Theoretic Analysis

The project now includes a strategic attacker/defender analysis. The attacker
chooses among red-team evasion strategies, while the defender chooses among
threshold policies. The loss matrix combines attacker bypass rate with a weighted
false-positive burden, so the defender minimizes worst-case security and
usability loss.

Run:

```powershell
pid game --dataset data/processed/dataset.csv --model-path artifacts/detector.joblib --output-dir reports
```

Outputs:

- `reports/game_payoff_matrix.csv`
- `reports/game_equilibrium.json`
- `reports/game_sensitivity.csv`
- `reports/game_theory_report.md`

This reframes the adversarial loop as a finite zero-sum security game and
provides a minimax equilibrium over attacker strategies and defender thresholds.

Current sensitivity result:

| False-positive weight | Equilibrium loss | Primary attacker | Primary defender threshold |
|---:|---:|---|---:|
| 0.25 | 0.031 | obfuscation | 0.25 |
| 1.00 | 0.100 | nested injection | 0.475 |

Interpretation: when false positives are cheap, the minimax defender becomes
strict and obfuscation is the binding attacker strategy. When false positives are
expensive, the defender moves toward the calibrated operating threshold and
nested injection becomes the most important adversarial pressure.

## Robustness Tests

The implemented robustness suite checks:

- category detection rates
- hardest attack category
- evasion-strategy attack success rate
- most effective evasion strategy
- sampled consistent failures and why they occur
- Base64 encoded injection variants
- Unicode lookalike substitutions
- multi-turn split attacks
- long benign-context embeddings

Current local edge-case detection is `1.0` for Base64, multi-turn, and long
benign-context tests, and `0.88` for Unicode-lookalike substitutions. The Unicode
gap is a useful next target for character-level features or transformer
fine-tuning.

The structured `reports/robustness_report.json` artifact now directly answers
the Step 5 review questions: which category is hardest, which evasion strategy
is most effective, where the detector fails, and whether the required edge
cases are detected.

## Curated Hard Suite

The project now includes a curated hard-suite benchmark with benign prompts that
look security-adjacent and subtler injection attempts. This is the more honest
local evaluation surface because it can reveal false positives and threshold
tradeoffs that the synthetic starter split hides.

Current hard-suite result at the selected threshold:

| Metric | Value |
|---|---:|
| Injection precision | 0.760 |
| Injection recall | 0.950 |
| Injection F1 | 0.844 |
| ROC-AUC | 0.943 |

Confusion matrix:

```text
[[14,  6],
 [ 1, 19]]
```

The threshold sweep identifies a recall-preserving hard-suite threshold near
`0.41` with full hard-suite recall and lower precision. This is useful for
security deployments where missed attacks are more expensive than review volume.

Run:

```powershell
pid benchmark --dataset data/processed/dataset.csv --model-path artifacts/detector.joblib --output-dir reports
```

Outputs:

- `reports/hard_case_metrics.json`
- `reports/hard_case_predictions.csv`
- `reports/hard_case_threshold_sweep.csv`
- `reports/local_evaluation_summary.md`

## Demo

The Streamlit app has two panels:

- Detector: input prompt, binary classification, category, score, feature signals
- Red-Team View: evasion variants, scores, bypass status, strategy names

Run:

```powershell
streamlit run src/prompt_injection_detector/app/streamlit_app.py
```

## Limitations

- The starter dataset is synthetic and template-driven.
- The local baseline is lexical; transformer fine-tuning is implemented but not
  run in the default local path.
- The MiniLM semantic baseline is implemented, but its results depend on
  downloading optional sentence-transformer weights and should be evaluated on
  the expanded public/manual corpus.
- No external HuggingFace upload is performed without credentials.
- No LLM-backed red-team generation is run without an API key.
- No demo video can be recorded automatically without a screen-recording workflow.

## Next Steps

1. Add public and manually reviewed examples.
2. Run MiniLM semantic-similarity evaluation in Colab and compare against TF-IDF.
3. Run transformer fine-tuning on DistilBERT or RoBERTa.
4. Run the LLM-backed red-team generator with reviewed outputs.
5. Upload the final dataset to HuggingFace.
6. Record a 2-3 minute demo video.
7. Update this report with real adversarial-loop curves from the expanded corpus.
