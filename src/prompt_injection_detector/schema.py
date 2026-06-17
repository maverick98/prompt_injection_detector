from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class InjectionCategory(str, Enum):
    ROLE_OVERRIDE = "role_override"
    INSTRUCTION_SMUGGLING = "instruction_smuggling"
    DATA_EXTRACTION = "data_extraction"
    JAILBREAK = "jailbreak"
    INDIRECT_INJECTION = "indirect_injection"
    CLEAN = "clean"


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class PromptSample(BaseModel):
    text: str = Field(min_length=1)
    label: int = Field(ge=0, le=1, description="1 means injection, 0 means clean")
    category: InjectionCategory
    source: str = "synthetic"
    split: Split | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Prediction(BaseModel):
    text: str
    is_injection: bool
    score: float
    category: InjectionCategory | None = None
    top_features: list[tuple[str, float]] = Field(default_factory=list)


class EvasionVariant(BaseModel):
    original_text: str
    variant_text: str
    strategy: str
    score: float | None = None
    bypassed: bool | None = None

