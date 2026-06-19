from __future__ import annotations

from collections.abc import Iterable

from prompt_injection_detector.schema import InjectionCategory, PromptSample


DEEPSET_DATASET_ID = "deepset/prompt-injections"


def infer_injection_category(text: str, label: int) -> InjectionCategory:
    if int(label) == 0:
        return InjectionCategory.CLEAN

    lowered = text.lower()
    if any(term in lowered for term in ["system prompt", "prompt text", "hidden", "private", "memory"]):
        return InjectionCategory.DATA_EXTRACTION
    if any(term in lowered for term in ["ignore", "forget", "disregard", "stop", "previous instructions"]):
        return InjectionCategory.ROLE_OVERRIDE
    if any(term in lowered for term in ["pretend", "act as", "role", "fictional", "jailbreak", "unrestricted"]):
        return InjectionCategory.JAILBREAK
    if any(term in lowered for term in ["document", "web page", "email", "snippet", "retrieved", "reading this"]):
        return InjectionCategory.INDIRECT_INJECTION
    return InjectionCategory.INSTRUCTION_SMUGGLING


def rows_to_prompt_samples(
    rows: Iterable[dict],
    source: str = "public_hf_deepset",
) -> list[PromptSample]:
    samples: list[PromptSample] = []
    for index, row in enumerate(rows):
        text = str(row["text"]).strip()
        if not text:
            continue
        label = int(row["label"])
        metadata = {
            "public_dataset": DEEPSET_DATASET_ID,
            "original_index": index,
            "category_inference": "rule_based",
        }
        if "hf_split" in row:
            metadata["hf_split"] = str(row["hf_split"])
        samples.append(
            PromptSample(
                text=text,
                label=label,
                category=infer_injection_category(text, label),
                source=source,
                metadata=metadata,
            )
        )
    return samples


def load_deepset_prompt_injections() -> list[PromptSample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the HuggingFace datasets package to import public datasets: "
            "python -m pip install datasets"
        ) from exc

    dataset = load_dataset(DEEPSET_DATASET_ID)
    rows: list[dict] = []
    for split_name in dataset:
        split = dataset[split_name]
        for row in split:
            normalized = dict(row)
            normalized["hf_split"] = split_name
            rows.append(normalized)
    return rows_to_prompt_samples(rows)
