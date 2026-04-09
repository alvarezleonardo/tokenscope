import pytest
from token_contador.core.pricing import PricingEngine
from token_contador.core.models import ModelInfo, Cost


def make_model(input_per_1m: float, output_per_1m: float) -> ModelInfo:
    return ModelInfo(
        id="test", provider="test", context_window=128000,
        tokenizer="tiktoken-cl100k", input_per_1m=input_per_1m,
        output_per_1m=output_per_1m, cached_input_per_1m=None,
        pricing_url="", updated_at="2026-04-09",
    )


def test_basic_cost():
    engine = PricingEngine()
    model = make_model(input_per_1m=2.50, output_per_1m=10.00)
    cost = engine.calculate(tokens_in=1000, tokens_out=500, model=model)
    assert isinstance(cost, Cost)
    assert abs(cost.input_usd - 0.0025) < 1e-9
    assert abs(cost.output_usd - 0.005) < 1e-9
    assert abs(cost.total_usd - 0.0075) < 1e-9


def test_zero_tokens():
    engine = PricingEngine()
    model = make_model(2.50, 10.00)
    cost = engine.calculate(tokens_in=0, tokens_out=0, model=model)
    assert cost.total_usd == 0.0


def test_large_text_cost():
    """1M tokens in + 100K out with GPT-4o pricing."""
    engine = PricingEngine()
    model = make_model(input_per_1m=2.50, output_per_1m=10.00)
    cost = engine.calculate(tokens_in=1_000_000, tokens_out=100_000, model=model)
    assert abs(cost.input_usd - 2.50) < 1e-9
    assert abs(cost.output_usd - 1.00) < 1e-9


def test_very_cheap_model():
    """Gemini Flash pricing."""
    engine = PricingEngine()
    model = make_model(input_per_1m=0.075, output_per_1m=0.30)
    cost = engine.calculate(tokens_in=21, tokens_out=500, model=model)
    assert cost.total_usd < 0.001
    assert cost.total_usd > 0
