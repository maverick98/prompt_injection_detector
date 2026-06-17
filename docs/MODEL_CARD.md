# Prompt Injection Detector Model Card

## Model Summary

The default trained detector is a recall-optimized classical text classifier:

- TF-IDF word n-grams from 1 to 3
- model comparison across Logistic Regression, RBF SVM, and Random Forest
- category classifier for detected injections
- threshold selected on validation data to prioritize injection recall

The locally generated artifact is written to:

```text
artifacts/detector.joblib
```

The artifact is ignored by Git because it is generated output.

## Current Local Evaluation

Evaluation was run on the generated 1,500-row synthetic starter dataset.

| Metric | Value |
|---|---:|
| Selected model | Logistic Regression |
| Decision threshold | 0.13 |
| Injection recall | 1.000 |
| Injection precision | 0.983 |
| Injection F1 | 0.991 |
| ROC-AUC | 1.000 |
| Accuracy | 0.991 |

Confusion matrix:

```text
[[110,   2],
 [  0, 113]]
```

Interpretation:

- zero false negatives on the starter split
- two false positives, which is acceptable for the security-first recall target
- metrics should be re-estimated after adding harder public/manual/LLM-reviewed data

## Category Results

Current detection rate on the starter split:

| Category | Detection Rate |
|---|---:|
| `data_extraction` | 1.000 |
| `indirect_injection` | 1.000 |
| `instruction_smuggling` | 1.000 |
| `jailbreak` | 1.000 |
| `role_override` | 1.000 |

## Robustness Edge Cases

Current detection rate on generated edge cases:

| Edge Case | Detection Rate |
|---|---:|
| Base64 encoded injections | 1.000 |
| Unicode lookalikes | 1.000 |
| Multi-turn split injection | 1.000 |
| Long benign-text embedding | 1.000 |

## Intended Use

The detector is intended as a defensive pre-filter for LLM application inputs and
as a research baseline for adversarial prompt-injection experiments.

## Limitations

- The default model is lexical and can miss novel attacks that do not share
  surface features with training data.
- Strong synthetic results do not imply production readiness.
- The transformer path is implemented but requires optional HuggingFace
  dependencies and compute.
- The LLM red-team generator is implemented but requires an API key.

## Safety Recommendation

Use this detector as one layer in a defense-in-depth system. Do not rely on it as
the only boundary around tools, secrets, documents, or agent actions.

