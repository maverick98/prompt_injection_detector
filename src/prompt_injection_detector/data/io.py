from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from prompt_injection_detector.schema import PromptSample, Split


REQUIRED_COLUMNS = {"text", "label", "category"}


def samples_to_frame(samples: list[PromptSample]) -> pd.DataFrame:
    return pd.DataFrame([sample.model_dump(mode="json") for sample in samples])


def save_samples(samples: list[PromptSample], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = samples_to_frame(samples)
    if path.suffix.lower() == ".jsonl":
        frame.to_json(path, orient="records", lines=True)
    else:
        frame.to_csv(path, index=False)
    return path


def load_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        frame = pd.read_json(path, lines=True)
    else:
        frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame


def stratified_split(
    frame: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        stratify=frame["category"],
        random_state=random_state,
    )
    adjusted_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        stratify=train_val["category"],
        random_state=random_state,
    )
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train["split"] = Split.TRAIN.value
    val["split"] = Split.VAL.value
    test["split"] = Split.TEST.value
    return pd.concat([train, val, test], ignore_index=True)

