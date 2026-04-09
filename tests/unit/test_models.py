from typing import Literal
from token_contador.core.models import Cost, TokenCount, ModelInfo, ModelResult, AnalysisResult


def test_cost_total():
    cost = Cost(input_usd=0.001, output_usd=0.005)
    assert cost.total_usd == 0.006


def test_token_count_fields():
    tc = TokenCount(tokens=42, method="exact", tokenizer="tiktoken-cl100k")
    assert tc.tokens == 42
    assert tc.method == "exact"


def test_model_info_fields():
    info = ModelInfo(
        id="gpt-4o",
        provider="openai",
        context_window=128000,
        tokenizer="tiktoken-cl100k",
        input_per_1m=2.50,
        output_per_1m=10.00,
        cached_input_per_1m=None,
        pricing_url="https://openai.com/pricing",
        updated_at="2026-04-09",
    )
    assert info.id == "gpt-4o"


def test_model_result_fits_in_context():
    cost = Cost(input_usd=0.0001, output_usd=0.001)
    result = ModelResult(
        model_id="gpt-4o",
        provider="openai",
        tokens=100,
        token_method="exact",
        context_window=128000,
        fits_in_context=True,
        cost=cost,
    )
    assert result.fits_in_context is True


def test_analysis_result_cheapest():
    cost_cheap = Cost(input_usd=0.00001, output_usd=0.0001)
    cost_expensive = Cost(input_usd=0.001, output_usd=0.01)
    r1 = ModelResult("gemini-flash", "google", 20, "estimated", 1000000, True, cost_cheap)
    r2 = ModelResult("gpt-4o", "openai", 19, "exact", 128000, True, cost_expensive)
    analysis = AnalysisResult(
        text_length=100,
        tokens_out_assumed=500,
        by_model={"gemini-flash": r1, "gpt-4o": r2},
        cheapest="gemini-flash",
        largest_context="gemini-flash",
    )
    assert analysis.cheapest == "gemini-flash"
