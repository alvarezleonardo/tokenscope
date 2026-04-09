from __future__ import annotations
from token_contador.core.models import AnalysisResult, ModelResult
from token_contador.core.registry import ModelRegistry
from token_contador.core.tokenizer import TokenizerEngine
from token_contador.core.pricing import PricingEngine

_tokenizer = TokenizerEngine()
_pricing = PricingEngine()


def analyze(
    text: str,
    providers: list[str] | None = None,
    model_ids: list[str] | None = None,
    tokens_out: int = 500,
    yaml_path: str | None = None,
) -> AnalysisResult:
    registry = ModelRegistry(yaml_path=yaml_path)

    if model_ids:
        models = [registry.get(mid) for mid in model_ids]
    elif providers:
        models = registry.filter(providers=providers)
    else:
        models = registry.get_all()

    results: dict[str, ModelResult] = {}
    for model in models:
        tc = _tokenizer.count(text, model)
        cost = _pricing.calculate(tc.tokens, tokens_out, model)
        results[model.id] = ModelResult(
            model_id=model.id,
            provider=model.provider,
            tokens=tc.tokens,
            token_method=tc.method,
            context_window=model.context_window,
            fits_in_context=tc.tokens <= model.context_window,
            cost=cost,
        )

    cheapest = min(results, key=lambda k: results[k].cost.total_usd)
    largest = max(results, key=lambda k: results[k].context_window)

    return AnalysisResult(
        text_length=len(text),
        tokens_out_assumed=tokens_out,
        by_model=results,
        cheapest=cheapest,
        largest_context=largest,
    )


def compare(
    texts: list[str],
    model_id: str,
    tokens_out: int = 500,
    yaml_path: str | None = None,
) -> list[AnalysisResult]:
    return [
        analyze(text, model_ids=[model_id], tokens_out=tokens_out, yaml_path=yaml_path)
        for text in texts
    ]
