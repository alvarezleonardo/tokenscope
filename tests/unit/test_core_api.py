import pytest
from token_contador.core import analyze, compare
from token_contador.core.models import AnalysisResult, ModelResult


def test_analyze_returns_analysis_result(tmp_models_yaml):
    result = analyze("Hello world", yaml_path=tmp_models_yaml)
    assert isinstance(result, AnalysisResult)
    assert len(result.by_model) == 3
    assert result.cheapest in result.by_model
    assert result.largest_context in result.by_model


def test_analyze_all_results_have_tokens(tmp_models_yaml):
    result = analyze("Hello world", yaml_path=tmp_models_yaml)
    for model_id, mr in result.by_model.items():
        assert isinstance(mr, ModelResult)
        assert mr.tokens > 0
        assert mr.cost.total_usd >= 0


def test_analyze_fits_in_context(tmp_models_yaml):
    """Short text fits in all fixture models."""
    result = analyze("Hi", yaml_path=tmp_models_yaml)
    for mr in result.by_model.values():
        assert mr.fits_in_context is True


def test_analyze_filter_by_provider(tmp_models_yaml):
    result = analyze("Hello world", providers=["openai"], yaml_path=tmp_models_yaml)
    assert len(result.by_model) == 1
    assert "gpt-4o" in result.by_model


def test_analyze_custom_tokens_out(tmp_models_yaml):
    result = analyze("Hello world", tokens_out=1000, yaml_path=tmp_models_yaml)
    assert result.tokens_out_assumed == 1000


def test_compare_multiple_texts(tmp_models_yaml):
    texts = ["Short text", "A much longer text that has more words and tokens in it for comparison"]
    results = compare(texts, model_id="gpt-4o", yaml_path=tmp_models_yaml)
    assert len(results) == 2
    assert results[0].by_model["gpt-4o"].tokens < results[1].by_model["gpt-4o"].tokens


def test_cheapest_is_lowest_cost(tmp_models_yaml):
    result = analyze("Hello world", yaml_path=tmp_models_yaml)
    cheapest_cost = result.by_model[result.cheapest].cost.total_usd
    for mr in result.by_model.values():
        assert mr.cost.total_usd >= cheapest_cost
