from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass
class Cost:
    input_usd: float
    output_usd: float

    @property
    def total_usd(self) -> float:
        return self.input_usd + self.output_usd


@dataclass
class TokenCount:
    tokens: int
    method: Literal["exact", "estimated"]
    tokenizer: str


@dataclass
class ModelInfo:
    id: str
    provider: str
    context_window: int
    tokenizer: str          # "tiktoken-cl100k", "hf:<repo>", "estimated"
    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None
    pricing_url: str
    updated_at: str
    chars_per_token: float = 4.0  # used when tokenizer == "estimated"


@dataclass
class ModelResult:
    model_id: str
    provider: str
    tokens: int
    token_method: Literal["exact", "estimated"]
    context_window: int
    fits_in_context: bool
    cost: Cost


@dataclass
class AnalysisResult:
    text_length: int
    tokens_out_assumed: int
    by_model: dict[str, ModelResult]
    cheapest: str
    largest_context: str
