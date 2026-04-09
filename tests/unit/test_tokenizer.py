import pytest
from unittest.mock import patch, MagicMock
from token_contador.core.tokenizer import TokenizerEngine
from token_contador.core.models import ModelInfo, TokenCount


def make_model(tokenizer: str, chars_per_token: float = 4.0) -> ModelInfo:
    return ModelInfo(
        id="test-model", provider="test", context_window=128000,
        tokenizer=tokenizer, input_per_1m=1.0, output_per_1m=1.0,
        cached_input_per_1m=None, pricing_url="", updated_at="2026-04-09",
        chars_per_token=chars_per_token,
    )


def test_tiktoken_exact_count(sample_texts):
    engine = TokenizerEngine()
    model = make_model("tiktoken-cl100k")
    result = engine.count(sample_texts["corto"], model)
    assert isinstance(result, TokenCount)
    assert result.method == "exact"
    assert result.tokenizer == "tiktoken-cl100k"
    assert result.tokens > 0


def test_tiktoken_multiple_texts(sample_texts):
    engine = TokenizerEngine()
    model = make_model("tiktoken-cl100k")
    for name, text in sample_texts.items():
        result = engine.count(text, model)
        assert result.tokens > 0, f"Expected tokens > 0 for {name}"


def test_estimated_tokenizer():
    engine = TokenizerEngine()
    model = make_model("estimated", chars_per_token=4.0)
    result = engine.count("Hello world", model)
    assert result.method == "estimated"
    assert result.tokens > 0


def test_hf_tokenizer_falls_back_to_estimation_on_error():
    """Si el tokenizer HF no está disponible/falla, cae a estimación."""
    engine = TokenizerEngine()
    model = make_model("hf:meta-llama/Meta-Llama-3.1-8B")
    with patch("token_contador.core.tokenizer.AutoTokenizer") as mock_tok:
        mock_tok.from_pretrained.side_effect = Exception("no internet")
        result = engine.count("Hello world", model)
    assert result.method == "estimated"
    assert result.tokens > 0
