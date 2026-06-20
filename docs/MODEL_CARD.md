# Prompt Injection Detector Model Card

## Model Summary

The default trained detector is a recall-optimized classical text classifier:

- TF-IDF word n-grams from 1 to 3
- TF-IDF character n-grams from 3 to 5 for obfuscation robustness
- model comparison across Logistic Regression, RBF SVM, and Random Forest
- category classifier for detected injections
- threshold selected on validation data to prioritize injection recall

The project also includes optional research detectors:

- MiniLM semantic similarity using `sentence-transformers/all-MiniLM-L6-v2`
- DistilBERT/RoBERTa fine-tuning for supervised transformer classification

The locally generated artifact is written to:

```text
artifacts/detector.joblib
```

The artifact is ignored by Git because it is generated output.

## Current Local Evaluation

Evaluation was run on the generated 1,500-row synthetic starter dataset.

Metric policy:

- optimize first for injection recall, because a false negative is a missed attack
- report precision, recall, and F1 for both clean and injection classes
- report ROC-AUC to compare score separation independent of one threshold
- inspect the confusion matrix to identify false negatives and false positives
- inspect per-category detection rate to find the weakest injection family

| Metric | Value |
|---|---:|
| Selected model | Logistic Regression |
| Decision threshold | 0.475 |
| Injection recall | 1.000 |
| Injection precision | 1.000 |
| Injection F1 | 1.000 |
| ROC-AUC | 1.000 |
| Accuracy | 1.000 |

Confusion matrix:

```text
[[112,   0],
 [  0, 113]]
```

Interpretation:

- zero false negatives on the starter split
- zero false positives on the starter split
- metrics should be re-estimated after adding harder public/manual/LLM-reviewed data

## Curated Hard-Suite Evaluation

The project also includes a 40-case curated hard suite with security-adjacent
clean prompts and subtler injection attempts. At the selected threshold:

| Metric | Value |
|---|---:|
| Injection recall | 1.000 |
| Injection precision | 0.760 |
| Injection F1 | 0.844 |
| ROC-AUC | 0.943 |

Confusion matrix:

```text
[[14,  6],
 [ 1, 19]]
```

This is the more honest local benchmark because it exposes remaining false
positives on benign prompts that mention policies, system prompts, or untrusted
instructions. The hard-suite threshold sweep also shows a recall-preserving
threshold around `0.41` for deployments that prefer fewer missed attacks.

## Category Results

Current detection rate on the starter split:

| Category | Detection Rate |
|---|---:|
| `data_extraction` | 1.000 |
| `indirect_injection` | 1.000 |
| `instruction_smuggling` | 1.000 |
| `jailbreak` | 1.000 |
| `role_override` | 1.000 |

## MiniLM Semantic Similarity Baseline

The MiniLM path embeds prompts into dense semantic vectors and compares each
test prompt against clean and injection reference examples from the training
split. It is not a replacement for the fine-tuned transformer classifier; it is
a complementary retrieval-style detector that asks whether a prompt is
semantically closer to known attacks or known benign examples.

Run:

```text
pid evaluate-minilm --dataset data/processed/dataset.csv --metrics-out reports/minilm_semantic_metrics.json
```

The output contains the same security metrics as the classical and transformer
paths: precision/recall/F1 per class, ROC-AUC, confusion matrix, per-category
detection rate, validation operating points, and reference counts.

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
- The MiniLM semantic-similarity path depends on optional HuggingFace /
  sentence-transformer dependencies and should be calibrated on validation data.
- Strong synthetic results do not imply production readiness.
- The transformer path is implemented but requires optional HuggingFace
  dependencies and compute.
- The LLM red-team generator is implemented but requires an API key.

## Safety Recommendation

Use this detector as one layer in a defense-in-depth system. Do not rely on it as
the only boundary around tools, secrets, documents, or agent actions.

## Strategic Evaluation

For adaptive attackers, use `pid game` to compute a zero-sum payoff matrix over
red-team strategies and defender threshold policies. The equilibrium value is an
estimated worst-case defender loss that combines bypass risk and false-positive
burden under mixed attacker/defender play.

Current sensitivity analysis shows the primary attacker shifts from
`obfuscation` to `nested_injection` as false-positive cost increases from `0.25`
to `1.0`, while the defender threshold shifts from `0.25` to `0.475`.
