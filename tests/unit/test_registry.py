import pytest
from token_contador.core.registry import ModelRegistry
from token_contador.core.models import ModelInfo


def test_load_models_from_yaml(tmp_models_yaml):
    registry = ModelRegistry(yaml_path=tmp_models_yaml)
    models = registry.get_all()
    assert len(models) == 3
    assert all(isinstance(m, ModelInfo) for m in models)


def test_get_by_id(tmp_models_yaml):
    registry = ModelRegistry(yaml_path=tmp_models_yaml)
    model = registry.get("gpt-4o")
    assert model.id == "gpt-4o"
    assert model.provider == "openai"
    assert model.input_per_1m == 2.50


def test_get_unknown_id_raises(tmp_models_yaml):
    registry = ModelRegistry(yaml_path=tmp_models_yaml)
    with pytest.raises(KeyError, match="unknown-model"):
        registry.get("unknown-model")


def test_filter_by_provider(tmp_models_yaml):
    registry = ModelRegistry(yaml_path=tmp_models_yaml)
    openai_models = registry.filter(provider="openai")
    assert len(openai_models) == 1
    assert openai_models[0].id == "gpt-4o"


def test_filter_by_min_context(tmp_models_yaml):
    registry = ModelRegistry(yaml_path=tmp_models_yaml)
    large_context = registry.filter(min_context=500000)
    assert len(large_context) == 1
    assert large_context[0].id == "gemini-2.0-flash"


def test_filter_multiple_providers(tmp_models_yaml):
    registry = ModelRegistry(yaml_path=tmp_models_yaml)
    models = registry.filter(providers=["openai", "google"])
    assert len(models) == 2


def test_default_yaml_loads():
    """Sin pasar yaml_path, carga el catálogo empaquetado."""
    registry = ModelRegistry()
    models = registry.get_all()
    assert len(models) > 10
    ids = [m.id for m in models]
    assert "gpt-4o" in ids
    assert "claude-sonnet-4-6" in ids
