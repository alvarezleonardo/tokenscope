from token_contador.core.models import ModelInfo, Cost


class PricingEngine:
    def calculate(self, tokens_in: int, tokens_out: int, model: ModelInfo) -> Cost:
        input_usd = (tokens_in / 1_000_000) * model.input_per_1m
        output_usd = (tokens_out / 1_000_000) * model.output_per_1m
        return Cost(input_usd=input_usd, output_usd=output_usd)
