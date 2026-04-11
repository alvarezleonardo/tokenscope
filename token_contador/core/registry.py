from __future__ import annotations
import yaml
from pathlib import Path
from importlib.resources import files
from token_contador.core.models import ModelInfo

_CUSTOM_MODELS_PATH = Path.home() / ".token_contador" / "custom_models.yaml"


class ModelRegistry:
    def __init__(self, yaml_path: str | None = None):
        if yaml_path is None:
            data = files("token_contador.data").joinpath("models.yaml").read_text()
        else:
            with open(yaml_path) as f:
                data = f.read()
        raw = yaml.safe_load(data)
        self._models: dict[str, ModelInfo] = {}
        for entry in raw["models"]:
            self._models[entry["id"]] = self._parse_entry(entry)

        # Load custom models from ~/.token_contador/custom_models.yaml (if exists)
        if yaml_path is None and _CUSTOM_MODELS_PATH.exists():
            with open(_CUSTOM_MODELS_PATH) as f:
                custom_raw = yaml.safe_load(f.read())
            for entry in custom_raw.get("models", []):
                self._models[entry["id"]] = self._parse_entry(entry)

    @staticmethod
    def _parse_entry(entry: dict) -> ModelInfo:
        return ModelInfo(
            id=entry["id"],
            provider=entry["provider"],
            context_window=entry["context_window"],
            tokenizer=entry["tokenizer"],
            input_per_1m=entry["input_per_1m"],
            output_per_1m=entry["output_per_1m"],
            cached_input_per_1m=entry.get("cached_input_per_1m"),
            pricing_url=entry["pricing_url"],
            updated_at=entry["updated_at"],
            chars_per_token=float(entry.get("chars_per_token", 4.0)),
        )

    def get_all(self) -> list[ModelInfo]:
        return list(self._models.values())

    def get(self, model_id: str) -> ModelInfo:
        if model_id not in self._models:
            raise KeyError(f"unknown-model: '{model_id}' no encontrado en el registry")
        return self._models[model_id]

    def filter(
        self,
        provider: str | None = None,
        providers: list[str] | None = None,
        min_context: int | None = None,
    ) -> list[ModelInfo]:
        result = list(self._models.values())
        if provider:
            result = [m for m in result if m.provider == provider]
        if providers:
            result = [m for m in result if m.provider in providers]
        if min_context:
            result = [m for m in result if m.context_window >= min_context]
        return result
