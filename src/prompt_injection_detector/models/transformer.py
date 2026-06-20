from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _training_arguments_kwargs(training_arguments_cls: type, output_dir: Path, epochs: int) -> dict:
    """Build TrainingArguments kwargs across Transformers API versions."""

    kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
    }
    parameters = inspect.signature(training_arguments_cls).parameters
    if "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    return kwargs


def _trainer_kwargs(
    trainer_cls: type,
    *,
    model,
    args,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator,
    compute_metrics,
) -> dict:
    """Build Trainer kwargs across Transformers API versions."""

    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    parameters = inspect.signature(trainer_cls).parameters
    if "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    return kwargs


def fine_tune_transformer(
    train: pd.DataFrame,
    val: pd.DataFrame,
    output_dir: str | Path,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 2,
) -> Path:
    """Fine-tune a HuggingFace transformer for binary classification.

    This optional path is intentionally isolated so the classical pipeline works
    without GPU or transformer dependencies.
    """

    try:
        import evaluate
        import numpy as np
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install optional dependencies with `pip install -e .[hf]` to fine-tune transformers."
        ) from exc

    output_dir = Path(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = Dataset.from_pandas(train[["text", "label"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val[["text", "label"]], preserve_index=False)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    args = TrainingArguments(**_training_arguments_kwargs(TrainingArguments, output_dir, epochs))

    trainer = Trainer(
        **_trainer_kwargs(
            Trainer,
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def evaluate_transformer_model(
    model_dir: str | Path,
    frame: pd.DataFrame,
    output_path: str | Path | None = None,
    threshold: float = 0.5,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate a fine-tuned transformer with the same security metrics as the baseline."""

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install optional dependencies with `pip install -e .[hf]` to evaluate transformers."
        ) from exc

    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts = frame["text"].astype(str).tolist()
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.detach().cpu().numpy()
            if logits.shape[1] == 1:
                batch_scores = _sigmoid(logits[:, 0])
            else:
                exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
                probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                batch_scores = probabilities[:, 1]
            scores.extend(float(score) for score in batch_scores)

    y_true = frame["label"].to_numpy(dtype=int)
    score_array = np.asarray(scores)
    y_pred = (score_array >= threshold).astype(int)
    try:
        roc_auc = float(roc_auc_score(y_true, score_array))
    except ValueError:
        roc_auc = float("nan")
    metrics: dict[str, Any] = {
        "model_dir": str(model_dir),
        "threshold": float(threshold),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "scores": scores,
        "predictions": y_pred.tolist(),
    }
    injection_frame = frame[frame["label"].astype(int) == 1].copy()
    if not injection_frame.empty:
        injection_frame["detected"] = y_pred[frame["label"].astype(int).to_numpy() == 1].astype(bool)
        metrics["per_category_detection_rate"] = (
            injection_frame.groupby("category")["detected"].mean().sort_index().to_dict()
        )
    else:
        metrics["per_category_detection_rate"] = {}

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
