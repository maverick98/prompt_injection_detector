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
2. Recall-optimized detector training and evaluation.
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

The selected local model is Logistic Regression with a decision threshold of
`0.13`, chosen to prioritize injection recall.

## Evaluation

Current local metrics on the generated starter split:

| Metric | Value |
|---|---:|
| Injection precision | 0.983 |
| Injection recall | 1.000 |
| Injection F1 | 0.991 |
| ROC-AUC | 1.000 |
| Accuracy | 0.991 |

Confusion matrix:

```text
[[110,   2],
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

Each variant is scored by the detector and marked as bypassed or blocked.

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

## Robustness Tests

The implemented robustness suite checks:

- category detection rates
- Base64 encoded injection variants
- Unicode lookalike substitutions
- multi-turn split attacks
- long benign-context embeddings

Current local edge-case detection rate is `1.0` across generated edge cases.

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
- No external HuggingFace upload is performed without credentials.
- No LLM-backed red-team generation is run without an API key.
- No demo video can be recorded automatically without a screen-recording workflow.

## Next Steps

1. Add public and manually reviewed examples.
2. Run transformer fine-tuning on DistilBERT or RoBERTa.
3. Run the LLM-backed red-team generator with reviewed outputs.
4. Upload the final dataset to HuggingFace.
5. Record a 2-3 minute demo video.
6. Update this report with real adversarial-loop curves from the expanded corpus.

