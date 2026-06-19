from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd


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
