import pandas as pd

from prompt_injection_detector.data.io import samples_to_frame, stratified_split
from prompt_injection_detector.data.synthetic import generate_synthetic_dataset


def test_synthetic_dataset_is_balanced_and_categorized():
    samples = generate_synthetic_dataset(50, 50)
    frame = samples_to_frame(samples)
    assert len(frame) == 100
    assert frame["label"].sum() == 50
    assert frame["category"].nunique() == 6


def test_stratified_split_adds_expected_splits():
    frame = samples_to_frame(generate_synthetic_dataset(50, 50))
    split = stratified_split(frame)
    assert set(split["split"]) == {"train", "val", "test"}


def test_stratified_split_handles_sparse_categories():
    frame = pd.DataFrame(
        {
            "text": [f"sample {index}" for index in range(20)],
            "label": [0, 1] * 10,
            "category": ["rare_a", "rare_b"] + ["clean", "role_override"] * 9,
        }
    )

    split = stratified_split(frame)

    assert len(split) == len(frame)
    assert set(split["split"]) == {"train", "val", "test"}
