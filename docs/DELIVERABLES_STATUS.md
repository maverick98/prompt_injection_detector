# Deliverables Status

This file tracks the requested submission deliverables against the current
repository state.

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | Labeled dataset | Code-ready; external upload pending | `pid build-dataset --include-public`, `pid import-hf-deepset`, `pid export-hf-data-card`, and `pid upload-hf-dataset` are implemented. Actual HuggingFace upload requires `HF_TOKEN` and a target dataset repo. |
| 2 | Trained detector | Partially complete | Classical TF-IDF detector, MiniLM semantic baseline, and transformer fine-tuning/evaluation code are implemented. Classical metrics are generated locally. Transformer metrics require a Colab/GPU fine-tuning run. |
| 3 | Red-team generator | Complete | Rule-based generator has five strategies. Optional LLM-backed generator supports Gemini, Cohere, OpenAI, and Groq through runtime secrets. |
| 4 | Adversarial loop results | Complete locally; optional LLM loop pending | Three-round local loop is implemented and executed. LLM-backed loop is implemented but requires an API key runtime. |
| 5 | Robustness report | Complete | Category rates, edge cases, evasion effectiveness, hardest category, and failure explanations are implemented in `pid robust`. |
| 6 | Streamlit demo app | Complete | Root `streamlit_app.py` loads the two-panel Detector and Red-Team workflow, plus benchmark/game/research tabs. |
| 7 | GitHub repository | Complete | Clean package structure, tests, docs, `.gitignore`, README, and pushed GitHub remote. |
| 8 | Demo video | Script ready; recording pending | `docs/DEMO_SCRIPT.md` is written. A 2-3 minute screen recording still needs to be recorded manually. |

## Remaining External Actions

1. Run transformer fine-tuning in Colab and save `reports/transformer_metrics.json`.
2. Upload the final dataset to HuggingFace Datasets:

```powershell
$env:HF_TOKEN="your_huggingface_write_token"
pid upload-hf-dataset --repo-id your-user/prompt-injection-detector --dataset data/processed/dataset.csv --data-card reports/hf_data_card.md
```

3. Run the optional LLM-backed red team and adversarial loop with
   `GEMINI_API_KEY` or `COHERE_API_KEY` configured at runtime.
4. Record the 2-3 minute demo video using `docs/DEMO_SCRIPT.md`.
