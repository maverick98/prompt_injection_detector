from __future__ import annotations

from pathlib import Path

import pandas as pd


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
    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir

