from pathlib import Path

from prompt_injection_detector.models.transformer import _trainer_kwargs, _training_arguments_kwargs


class LegacyTrainingArguments:
    def __init__(
        self,
        output_dir,
        learning_rate,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        num_train_epochs,
        weight_decay,
        evaluation_strategy,
        save_strategy,
        load_best_model_at_end,
        metric_for_best_model,
    ):
        pass


class CurrentTrainingArguments:
    def __init__(
        self,
        output_dir,
        learning_rate,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        num_train_epochs,
        weight_decay,
        eval_strategy,
        save_strategy,
        load_best_model_at_end,
        metric_for_best_model,
    ):
        pass


def test_training_arguments_uses_legacy_evaluation_strategy_when_available():
    kwargs = _training_arguments_kwargs(LegacyTrainingArguments, Path("artifacts/model"), 2)

    assert kwargs["evaluation_strategy"] == "epoch"
    assert "eval_strategy" not in kwargs


def test_training_arguments_uses_current_eval_strategy_when_required():
    kwargs = _training_arguments_kwargs(CurrentTrainingArguments, Path("artifacts/model"), 2)

    assert kwargs["eval_strategy"] == "epoch"
    assert "evaluation_strategy" not in kwargs


class LegacyTrainer:
    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        tokenizer,
        data_collator,
        compute_metrics,
    ):
        pass


class CurrentTrainer:
    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        processing_class,
        data_collator,
        compute_metrics,
    ):
        pass


def test_trainer_kwargs_uses_legacy_tokenizer_when_available():
    kwargs = _trainer_kwargs(
        LegacyTrainer,
        model="model",
        args="args",
        train_dataset="train",
        eval_dataset="eval",
        tokenizer="tokenizer",
        data_collator="collator",
        compute_metrics="metrics",
    )

    assert kwargs["tokenizer"] == "tokenizer"
    assert "processing_class" not in kwargs


def test_trainer_kwargs_uses_current_processing_class_when_required():
    kwargs = _trainer_kwargs(
        CurrentTrainer,
        model="model",
        args="args",
        train_dataset="train",
        eval_dataset="eval",
        tokenizer="tokenizer",
        data_collator="collator",
        compute_metrics="metrics",
    )

    assert kwargs["processing_class"] == "tokenizer"
    assert "tokenizer" not in kwargs
