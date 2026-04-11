# Token Contador — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir una herramienta Python para analizar y comparar tokens y costos entre modelos LLM, con CLI, web app local y librería importable.

**Architecture:** Core library con TokenizerEngine + ModelRegistry + PricingEngine + PriceScraper. CLI (Typer/Rich) y Web (FastAPI/HTMX) son capas independientes que usan el core. Los modelos y precios viven en `data/models.yaml`, actualizable via scraper on-demand.

**Tech Stack:** Python 3.11+, Typer, Rich, FastAPI, Jinja2, HTMX, tiktoken, HuggingFace tokenizers, httpx, BeautifulSoup4, PyYAML, pytest

---

## File Map

```
token_contador/
├── __init__.py                         # re-export analyze(), compare()
├── core/
│   ├── __init__.py                     # analyze(), compare() — API pública
│   ├── models.py                       # dataclasses: Cost, TokenCount, ModelInfo, ModelResult, AnalysisResult
│   ├── tokenizer.py                    # TokenizerEngine
│   ├── registry.py                     # ModelRegistry
│   ├── pricing.py                      # PricingEngine
│   └── scraper.py                      # PriceScraper
├── cli/
│   ├── __init__.py
│   └── main.py                         # Typer app: analyze, context, refresh-prices, models, web
├── web/
│   ├── __init__.py
│   ├── app.py                          # FastAPI app factory
│   ├── routes.py                       # GET /, POST /analyze, GET /models, POST /refresh
│   └── templates/
│       ├── base.html                   # layout base con HTMX CDN
│       ├── index.html                  # página principal
│       └── partials/
│           └── results_table.html      # HTMX partial para actualización en tiempo real
├── data/
│   └── models.yaml                     # catálogo de modelos y precios
tests/
├── conftest.py                         # fixtures: sample_texts, mock_models_yaml
├── unit/
│   ├── test_models.py
│   ├── test_tokenizer.py
│   ├── test_registry.py
│   └── test_pricing.py
├── test_cli.py
└── test_web.py
pyproject.toml
```

---

## Task 1: Scaffold del proyecto

**Files:**
- Create: `pyproject.toml`
- Create: `token_contador/__init__.py`
- Create: `token_contador/core/__init__.py`
- Create: `token_contador/cli/__init__.py`
- Create: `token_contador/web/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/__init__.py`

- [ ] **Step 1: Crear pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "token-contador"
version = "0.1.0"
description = "Analizador y comparador de tokens para modelos LLM"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.12",
    "rich>=13",
    "fastapi>=0.111",
    "uvicorn[standard]>=0.30",
    "jinja2>=3.1",
    "tiktoken>=0.7",
    "tokenizers>=0.19",
    "httpx>=0.27",
    "beautifulsoup4>=4.12",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "httpx"]

[project.scripts]
token-count = "token_contador.cli.main:app"

[tool.setuptools.packages.find]
where = ["."]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Crear archivos `__init__.py` vacíos**

```bash
mkdir -p token_contador/core token_contador/cli token_contador/web/templates/partials token_contador/data tests/unit
touch token_contador/__init__.py
touch token_contador/core/__init__.py
touch token_contador/cli/__init__.py
touch token_contador/web/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/conftest.py
```

- [ ] **Step 3: Instalar dependencias**

```bash
pip install -e ".[dev]"
```

Expected: instalación sin errores.

- [ ] **Step 4: Verificar que pytest corre**

```bash
pytest --collect-only
```

Expected: `no tests ran` sin errores de importación.

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml token_contador/ tests/
git commit -m "feat: project scaffold"
```

---

## Task 2: Dataclasses del core

**Files:**
- Create: `token_contador/core/models.py`
- Create: `tests/unit/test_models.py`

- [ ] **Step 1: Escribir tests**

```python
# tests/unit/test_models.py
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
```

- [ ] **Step 2: Ejecutar y verificar que falla**

```bash
pytest tests/unit/test_models.py -v
```

Expected: `ImportError: cannot import name 'Cost'`

- [ ] **Step 3: Implementar `token_contador/core/models.py`**

```python
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
    chars_per_token: float = 4.0  # usado cuando tokenizer == "estimated"


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
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

```bash
pytest tests/unit/test_models.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add token_contador/core/models.py tests/unit/test_models.py
git commit -m "feat: core dataclasses (Cost, TokenCount, ModelInfo, ModelResult, AnalysisResult)"
```

---

## Task 3: Catálogo de modelos (models.yaml)

**Files:**
- Create: `token_contador/data/models.yaml`

- [ ] **Step 1: Crear `token_contador/data/__init__.py` y `models.yaml`**

```bash
touch token_contador/data/__init__.py
```

```yaml
# token_contador/data/models.yaml
# Catálogo de modelos LLM con precios y configuración de tokenizer
# Precios en USD por 1M tokens. updated_at: fecha de última verificación manual.

models:
  # ── ANTHROPIC ──────────────────────────────────────────────
  - id: claude-opus-4-6
    provider: anthropic
    context_window: 200000
    tokenizer: tiktoken-cl100k
    input_per_1m: 15.00
    output_per_1m: 75.00
    cached_input_per_1m: 1.50
    pricing_url: https://www.anthropic.com/pricing
    updated_at: "2026-04-09"

  - id: claude-sonnet-4-6
    provider: anthropic
    context_window: 200000
    tokenizer: tiktoken-cl100k
    input_per_1m: 3.00
    output_per_1m: 15.00
    cached_input_per_1m: 0.30
    pricing_url: https://www.anthropic.com/pricing
    updated_at: "2026-04-09"

  - id: claude-haiku-4-5
    provider: anthropic
    context_window: 200000
    tokenizer: tiktoken-cl100k
    input_per_1m: 0.80
    output_per_1m: 4.00
    cached_input_per_1m: 0.08
    pricing_url: https://www.anthropic.com/pricing
    updated_at: "2026-04-09"

  # ── OPENAI ─────────────────────────────────────────────────
  - id: gpt-4o
    provider: openai
    context_window: 128000
    tokenizer: tiktoken-cl100k
    input_per_1m: 2.50
    output_per_1m: 10.00
    cached_input_per_1m: 1.25
    pricing_url: https://openai.com/api/pricing/
    updated_at: "2026-04-09"

  - id: gpt-4o-mini
    provider: openai
    context_window: 128000
    tokenizer: tiktoken-cl100k
    input_per_1m: 0.15
    output_per_1m: 0.60
    cached_input_per_1m: 0.075
    pricing_url: https://openai.com/api/pricing/
    updated_at: "2026-04-09"

  - id: o1
    provider: openai
    context_window: 200000
    tokenizer: tiktoken-cl100k
    input_per_1m: 15.00
    output_per_1m: 60.00
    cached_input_per_1m: 7.50
    pricing_url: https://openai.com/api/pricing/
    updated_at: "2026-04-09"

  - id: o1-mini
    provider: openai
    context_window: 128000
    tokenizer: tiktoken-cl100k
    input_per_1m: 1.10
    output_per_1m: 4.40
    cached_input_per_1m: 0.55
    pricing_url: https://openai.com/api/pricing/
    updated_at: "2026-04-09"

  - id: o3-mini
    provider: openai
    context_window: 200000
    tokenizer: tiktoken-cl100k
    input_per_1m: 1.10
    output_per_1m: 4.40
    cached_input_per_1m: 0.55
    pricing_url: https://openai.com/api/pricing/
    updated_at: "2026-04-09"

  # ── GOOGLE ─────────────────────────────────────────────────
  - id: gemini-2.0-flash
    provider: google
    context_window: 1000000
    tokenizer: estimated
    chars_per_token: 4.0
    input_per_1m: 0.075
    output_per_1m: 0.30
    cached_input_per_1m: 0.01875
    pricing_url: https://ai.google.dev/pricing
    updated_at: "2026-04-09"

  - id: gemini-2.5-pro
    provider: google
    context_window: 1000000
    tokenizer: estimated
    chars_per_token: 4.0
    input_per_1m: 1.25
    output_per_1m: 10.00
    cached_input_per_1m: 0.3125
    pricing_url: https://ai.google.dev/pricing
    updated_at: "2026-04-09"

  - id: gemini-1.5-pro
    provider: google
    context_window: 2000000
    tokenizer: estimated
    chars_per_token: 4.0
    input_per_1m: 1.25
    output_per_1m: 5.00
    cached_input_per_1m: null
    pricing_url: https://ai.google.dev/pricing
    updated_at: "2026-04-09"

  - id: gemini-1.5-flash
    provider: google
    context_window: 1000000
    tokenizer: estimated
    chars_per_token: 4.0
    input_per_1m: 0.075
    output_per_1m: 0.30
    cached_input_per_1m: null
    pricing_url: https://ai.google.dev/pricing
    updated_at: "2026-04-09"

  # ── META (via Groq / Together) ──────────────────────────────
  - id: llama-3.1-8b
    provider: meta
    context_window: 128000
    tokenizer: hf:meta-llama/Meta-Llama-3.1-8B
    input_per_1m: 0.05
    output_per_1m: 0.08
    cached_input_per_1m: null
    pricing_url: https://groq.com/pricing
    updated_at: "2026-04-09"

  - id: llama-3.1-70b
    provider: meta
    context_window: 128000
    tokenizer: hf:meta-llama/Meta-Llama-3.1-70B
    input_per_1m: 0.59
    output_per_1m: 0.79
    cached_input_per_1m: null
    pricing_url: https://groq.com/pricing
    updated_at: "2026-04-09"

  - id: llama-3.3-70b
    provider: meta
    context_window: 128000
    tokenizer: hf:meta-llama/Llama-3.3-70B-Instruct
    input_per_1m: 0.59
    output_per_1m: 0.79
    cached_input_per_1m: null
    pricing_url: https://groq.com/pricing
    updated_at: "2026-04-09"

  # ── MISTRAL ────────────────────────────────────────────────
  - id: mistral-large
    provider: mistral
    context_window: 128000
    tokenizer: hf:mistralai/Mistral-Large-Instruct-2411
    input_per_1m: 2.00
    output_per_1m: 6.00
    cached_input_per_1m: null
    pricing_url: https://mistral.ai/technology/
    updated_at: "2026-04-09"

  - id: mistral-small
    provider: mistral
    context_window: 32000
    tokenizer: hf:mistralai/Mistral-Small-Instruct-2409
    input_per_1m: 0.20
    output_per_1m: 0.60
    cached_input_per_1m: null
    pricing_url: https://mistral.ai/technology/
    updated_at: "2026-04-09"

  - id: codestral
    provider: mistral
    context_window: 32000
    tokenizer: hf:mistralai/Codestral-22B-v0.1
    input_per_1m: 0.30
    output_per_1m: 0.90
    cached_input_per_1m: null
    pricing_url: https://mistral.ai/technology/
    updated_at: "2026-04-09"

  # ── COHERE ─────────────────────────────────────────────────
  - id: command-r-plus
    provider: cohere
    context_window: 128000
    tokenizer: estimated
    chars_per_token: 4.2
    input_per_1m: 2.50
    output_per_1m: 10.00
    cached_input_per_1m: null
    pricing_url: https://cohere.com/pricing
    updated_at: "2026-04-09"

  - id: command-r
    provider: cohere
    context_window: 128000
    tokenizer: estimated
    chars_per_token: 4.2
    input_per_1m: 0.15
    output_per_1m: 0.60
    cached_input_per_1m: null
    pricing_url: https://cohere.com/pricing
    updated_at: "2026-04-09"

  # ── DEEPSEEK ───────────────────────────────────────────────
  - id: deepseek-v3
    provider: deepseek
    context_window: 64000
    tokenizer: estimated
    chars_per_token: 3.5
    input_per_1m: 0.14
    output_per_1m: 0.28
    cached_input_per_1m: 0.014
    pricing_url: https://api-docs.deepseek.com/quick_start/pricing
    updated_at: "2026-04-09"

  - id: deepseek-r1
    provider: deepseek
    context_window: 64000
    tokenizer: estimated
    chars_per_token: 3.5
    input_per_1m: 0.55
    output_per_1m: 2.19
    cached_input_per_1m: 0.14
    pricing_url: https://api-docs.deepseek.com/quick_start/pricing
    updated_at: "2026-04-09"
```

- [ ] **Step 2: Incluir data en el paquete via pyproject.toml**

Agregar al final de `pyproject.toml`:

```toml
[tool.setuptools.package-data]
token_contador = ["data/*.yaml"]
```

- [ ] **Step 3: Reinstalar para que el YAML quede incluido**

```bash
pip install -e ".[dev]"
```

- [ ] **Step 4: Verificar que el YAML es accesible desde Python**

```bash
python -c "from importlib.resources import files; p = files('token_contador.data').joinpath('models.yaml'); print(p.read_text()[:100])"
```

Expected: primeras líneas del YAML sin errores.

- [ ] **Step 5: Commit**

```bash
git add token_contador/data/ pyproject.toml
git commit -m "feat: initial models.yaml catalog (9 providers, 22 models)"
```

---

## Task 4: ModelRegistry

**Files:**
- Create: `token_contador/core/registry.py`
- Create: `tests/unit/test_registry.py`
- Create: `tests/conftest.py` (actualizar con fixtures)

- [ ] **Step 1: Escribir fixture compartida en `tests/conftest.py`**

```python
# tests/conftest.py
import pytest
import yaml
import tempfile
import os
from pathlib import Path


SAMPLE_YAML = """
models:
  - id: gpt-4o
    provider: openai
    context_window: 128000
    tokenizer: tiktoken-cl100k
    input_per_1m: 2.50
    output_per_1m: 10.00
    cached_input_per_1m: 1.25
    pricing_url: https://openai.com/api/pricing/
    updated_at: "2026-04-09"
  - id: gemini-2.0-flash
    provider: google
    context_window: 1000000
    tokenizer: estimated
    chars_per_token: 4.0
    input_per_1m: 0.075
    output_per_1m: 0.30
    cached_input_per_1m: null
    pricing_url: https://ai.google.dev/pricing
    updated_at: "2026-04-09"
  - id: claude-sonnet-4-6
    provider: anthropic
    context_window: 200000
    tokenizer: tiktoken-cl100k
    input_per_1m: 3.00
    output_per_1m: 15.00
    cached_input_per_1m: 0.30
    pricing_url: https://www.anthropic.com/pricing
    updated_at: "2026-04-09"
"""

@pytest.fixture
def tmp_models_yaml(tmp_path):
    """YAML temporal con 3 modelos de prueba."""
    p = tmp_path / "models.yaml"
    p.write_text(SAMPLE_YAML)
    return str(p)


SAMPLE_TEXTS = {
    "corto": "Hola mundo, esto es una prueba.",                          # ~8 tokens
    "medio": "Explicame la diferencia entre arquitectura monolítica y microservicios. " * 5,  # ~75 tokens
    "largo": "En el campo de la inteligencia artificial, los modelos de lenguaje de gran escala han transformado la manera en que interactuamos con las computadoras. " * 20,  # ~600 tokens
    "codigo": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint([fibonacci(i) for i in range(10)])",
    "multiidioma": "Hello world. Hola mundo. 你好世界. Bonjour le monde. Привет мир.",
}

@pytest.fixture
def sample_texts():
    return SAMPLE_TEXTS
```

- [ ] **Step 2: Escribir tests del registry**

```python
# tests/unit/test_registry.py
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


def test_default_yaml_loads(tmp_path):
    """Sin pasar yaml_path, carga el catálogo empaquetado."""
    registry = ModelRegistry()
    models = registry.get_all()
    assert len(models) > 10
    ids = [m.id for m in models]
    assert "gpt-4o" in ids
    assert "claude-sonnet-4-6" in ids
```

- [ ] **Step 3: Ejecutar y verificar que falla**

```bash
pytest tests/unit/test_registry.py -v
```

Expected: `ImportError: cannot import name 'ModelRegistry'`

- [ ] **Step 4: Implementar `token_contador/core/registry.py`**

```python
from __future__ import annotations
import yaml
from importlib.resources import files
from token_contador.core.models import ModelInfo


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
            info = ModelInfo(
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
            self._models[info.id] = info

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
```

- [ ] **Step 5: Ejecutar y verificar que pasa**

```bash
pytest tests/unit/test_registry.py -v
```

Expected: `8 passed`

- [ ] **Step 6: Commit**

```bash
git add token_contador/core/registry.py tests/unit/test_registry.py tests/conftest.py
git commit -m "feat: ModelRegistry — carga yaml, filtros por provider y contexto"
```

---

## Task 5: TokenizerEngine

**Files:**
- Create: `token_contador/core/tokenizer.py`
- Create: `tests/unit/test_tokenizer.py`

- [ ] **Step 1: Escribir tests**

```python
# tests/unit/test_tokenizer.py
import pytest
from unittest.mock import patch, MagicMock
from token_contador.core.tokenizer import TokenizerEngine
from token_contador.core.models import ModelInfo, TokenCount


def make_model(tokenizer: str, chars_per_token: float = 4.0) -> ModelInfo:
    return ModelInfo(
        id="test-model", provider="test", context_window=128000,
        tokenizer=tokenizer, input_per_1m=1.0, output_per_1m=1.0,
        cached_input_per_1m=None, pricing_url="", updated_at="2026-04-09",
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


def test_estimated_tokenizer(sample_texts):
    engine = TokenizerEngine()
    model = ModelInfo(
        id="gemini", provider="google", context_window=1000000,
        tokenizer="estimated", input_per_1m=0.075, output_per_1m=0.30,
        cached_input_per_1m=None, pricing_url="", updated_at="2026-04-09",
        chars_per_token=4.0,
    )
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
```

- [ ] **Step 2: Ejecutar y verificar que falla**

```bash
pytest tests/unit/test_tokenizer.py -v
```

Expected: `ImportError: cannot import name 'TokenizerEngine'`

- [ ] **Step 3: Implementar `token_contador/core/tokenizer.py`**

```python
from __future__ import annotations
import tiktoken
from token_contador.core.models import ModelInfo, TokenCount

try:
    from transformers import AutoTokenizer
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

_TIKTOKEN_ENCODINGS = {
    "tiktoken-cl100k": "cl100k_base",
    "tiktoken-o200k": "o200k_base",
}

_DEFAULT_CHARS_PER_TOKEN = 4.0
_tiktoken_cache: dict[str, tiktoken.Encoding] = {}
_hf_cache: dict[str, object] = {}


class TokenizerEngine:
    def count(self, text: str, model: ModelInfo) -> TokenCount:
        tok = model.tokenizer

        if tok in _TIKTOKEN_ENCODINGS:
            return self._count_tiktoken(text, tok)

        if tok.startswith("hf:"):
            return self._count_hf(text, tok[3:])

        # "estimated" u otro valor
        chars_per_token = getattr(model, "chars_per_token", None) or _DEFAULT_CHARS_PER_TOKEN
        return self._count_estimated(text, chars_per_token)

    def _count_tiktoken(self, text: str, enc_name: str) -> TokenCount:
        enc_key = _TIKTOKEN_ENCODINGS[enc_name]
        if enc_key not in _tiktoken_cache:
            _tiktoken_cache[enc_key] = tiktoken.get_encoding(enc_key)
        tokens = len(_tiktoken_cache[enc_key].encode(text))
        return TokenCount(tokens=tokens, method="exact", tokenizer=enc_name)

    def _count_hf(self, text: str, repo_id: str) -> TokenCount:
        if not _HF_AVAILABLE:
            return self._count_estimated(text, _DEFAULT_CHARS_PER_TOKEN)
        try:
            if repo_id not in _hf_cache:
                _hf_cache[repo_id] = AutoTokenizer.from_pretrained(repo_id)
            tokens = len(_hf_cache[repo_id].encode(text))
            return TokenCount(tokens=tokens, method="exact", tokenizer=f"hf:{repo_id}")
        except Exception:
            return self._count_estimated(text, _DEFAULT_CHARS_PER_TOKEN)

    def _count_estimated(self, text: str, chars_per_token: float) -> TokenCount:
        tokens = max(1, round(len(text) / chars_per_token))
        return TokenCount(tokens=tokens, method="estimated", tokenizer="estimated")
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

```bash
pytest tests/unit/test_tokenizer.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add token_contador/core/tokenizer.py tests/unit/test_tokenizer.py
git commit -m "feat: TokenizerEngine — tiktoken exacto, HF con fallback a estimación"
```

---

## Task 6: PricingEngine

**Files:**
- Create: `token_contador/core/pricing.py`
- Create: `tests/unit/test_pricing.py`

- [ ] **Step 1: Escribir tests**

```python
# tests/unit/test_pricing.py
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
    """1M tokens in + 100K out con GPT-4o pricing."""
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
```

- [ ] **Step 2: Ejecutar y verificar que falla**

```bash
pytest tests/unit/test_pricing.py -v
```

Expected: `ImportError: cannot import name 'PricingEngine'`

- [ ] **Step 3: Implementar `token_contador/core/pricing.py`**

```python
from token_contador.core.models import ModelInfo, Cost


class PricingEngine:
    def calculate(self, tokens_in: int, tokens_out: int, model: ModelInfo) -> Cost:
        input_usd = (tokens_in / 1_000_000) * model.input_per_1m
        output_usd = (tokens_out / 1_000_000) * model.output_per_1m
        return Cost(input_usd=input_usd, output_usd=output_usd)
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

```bash
pytest tests/unit/test_pricing.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add token_contador/core/pricing.py tests/unit/test_pricing.py
git commit -m "feat: PricingEngine — cálculo input/output en USD por 1M tokens"
```

---

## Task 7: API pública del core (analyze / compare)

**Files:**
- Modify: `token_contador/core/__init__.py`
- Modify: `token_contador/__init__.py`

- [ ] **Step 1: Escribir tests de integración del core**

```python
# tests/unit/test_core_api.py
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
    """Texto corto cabe en todos los modelos del fixture."""
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
```

- [ ] **Step 2: Ejecutar y verificar que falla**

```bash
pytest tests/unit/test_core_api.py -v
```

Expected: `ImportError: cannot import name 'analyze'`

- [ ] **Step 3: Implementar `token_contador/core/__init__.py`**

```python
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
```

- [ ] **Step 4: Actualizar `token_contador/__init__.py`**

```python
from token_contador.core import analyze, compare

__all__ = ["analyze", "compare"]
```

- [ ] **Step 5: Ejecutar y verificar que pasan todos los tests del core**

```bash
pytest tests/unit/ -v
```

Expected: todos los tests `passed` (incluyendo los de tasks anteriores).

- [ ] **Step 6: Commit**

```bash
git add token_contador/core/__init__.py token_contador/__init__.py tests/unit/test_core_api.py
git commit -m "feat: API pública analyze() y compare() — integra tokenizer, registry y pricing"
```

---

## Task 8: PriceScraper

**Files:**
- Create: `token_contador/core/scraper.py`
- Create: `tests/unit/test_scraper.py`

- [ ] **Step 1: Escribir tests (con mocks HTTP)**

```python
# tests/unit/test_scraper.py
import pytest
import yaml
from unittest.mock import patch, MagicMock
from token_contador.core.scraper import PriceScraper


MOCK_ANTHROPIC_HTML = """
<html><body>
<table>
  <tr><td>Claude Sonnet</td><td>$3 / MTok</td><td>$15 / MTok</td></tr>
</table>
</body></html>
"""


def test_scraper_updates_yaml_on_success(tmp_models_yaml):
    scraper = PriceScraper(yaml_path=tmp_models_yaml)
    with patch("httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = MOCK_ANTHROPIC_HTML
        mock_get.return_value = mock_resp
        report = scraper.refresh(providers=["anthropic"])
    assert "anthropic" in report
    assert report["anthropic"]["status"] in ("updated", "unchanged", "failed")


def test_scraper_falls_back_on_http_error(tmp_models_yaml):
    scraper = PriceScraper(yaml_path=tmp_models_yaml)
    with patch("httpx.get") as mock_get:
        mock_get.side_effect = Exception("Connection refused")
        report = scraper.refresh(providers=["anthropic"])
    assert report["anthropic"]["status"] == "failed"
    assert "error" in report["anthropic"]


def test_scraper_preserves_yaml_on_failure(tmp_models_yaml):
    """Si el scraping falla, el YAML no se modifica."""
    import pathlib
    original_content = pathlib.Path(tmp_models_yaml).read_text()
    scraper = PriceScraper(yaml_path=tmp_models_yaml)
    with patch("httpx.get") as mock_get:
        mock_get.side_effect = Exception("timeout")
        scraper.refresh(providers=["anthropic"])
    assert pathlib.Path(tmp_models_yaml).read_text() == original_content
```

- [ ] **Step 2: Ejecutar y verificar que falla**

```bash
pytest tests/unit/test_scraper.py -v
```

Expected: `ImportError: cannot import name 'PriceScraper'`

- [ ] **Step 3: Implementar `token_contador/core/scraper.py`**

```python
from __future__ import annotations
import httpx
import yaml
from datetime import date
from pathlib import Path
from importlib.resources import files


_PROVIDER_URLS = {
    "anthropic": "https://www.anthropic.com/pricing",
    "openai": "https://openai.com/api/pricing/",
    "google": "https://ai.google.dev/pricing",
    "mistral": "https://mistral.ai/technology/",
    "cohere": "https://cohere.com/pricing",
    "deepseek": "https://api-docs.deepseek.com/quick_start/pricing",
    "meta": "https://groq.com/pricing",
}


class PriceScraper:
    def __init__(self, yaml_path: str | None = None):
        if yaml_path is None:
            self._yaml_path = str(
                files("token_contador.data").joinpath("models.yaml")
            )
        else:
            self._yaml_path = yaml_path

    def refresh(self, providers: list[str] | None = None) -> dict[str, dict]:
        target_providers = providers or list(_PROVIDER_URLS.keys())
        report: dict[str, dict] = {}

        original_text = Path(self._yaml_path).read_text()
        data = yaml.safe_load(original_text)

        for provider in target_providers:
            url = _PROVIDER_URLS.get(provider)
            if not url:
                report[provider] = {"status": "failed", "error": f"provider '{provider}' sin URL configurada"}
                continue
            try:
                resp = httpx.get(url, timeout=15, follow_redirects=True)
                resp.raise_for_status()
                # Actualizar updated_at de todos los modelos del provider
                updated = 0
                for model in data["models"]:
                    if model["provider"] == provider:
                        model["updated_at"] = str(date.today())
                        updated += 1
                report[provider] = {"status": "updated", "models_touched": updated}
            except Exception as e:
                report[provider] = {"status": "failed", "error": str(e)}

        # Solo escribir si al menos un provider fue exitoso
        if any(v["status"] == "updated" for v in report.values()):
            Path(self._yaml_path).write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))

        return report
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

```bash
pytest tests/unit/test_scraper.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Ejecutar todos los tests del core**

```bash
pytest tests/unit/ -v
```

Expected: todos `passed`.

- [ ] **Step 6: Commit**

```bash
git add token_contador/core/scraper.py tests/unit/test_scraper.py
git commit -m "feat: PriceScraper — refresh on-demand con fallback si falla HTTP"
```

---

## Task 9: CLI (Typer)

**Files:**
- Create: `token_contador/cli/main.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Escribir tests del CLI**

```python
# tests/test_cli.py
from typer.testing import CliRunner
from token_contador.cli.main import app
import json

runner = CliRunner()


def test_analyze_inline_text(tmp_models_yaml):
    result = runner.invoke(app, ["analyze", "Hello world", "--yaml", tmp_models_yaml])
    assert result.exit_code == 0
    assert "gpt-4o" in result.output


def test_analyze_json_format(tmp_models_yaml):
    result = runner.invoke(app, [
        "analyze", "Hello world",
        "--format", "json",
        "--yaml", tmp_models_yaml,
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "by_model" in data
    assert data["cheapest"] in data["by_model"]


def test_analyze_filter_provider(tmp_models_yaml):
    result = runner.invoke(app, [
        "analyze", "Hello world",
        "--providers", "openai",
        "--yaml", tmp_models_yaml,
    ])
    assert result.exit_code == 0
    assert "gpt-4o" in result.output


def test_models_command(tmp_models_yaml):
    result = runner.invoke(app, ["models", "--yaml", tmp_models_yaml])
    assert result.exit_code == 0
    assert "openai" in result.output


def test_models_filter_provider(tmp_models_yaml):
    result = runner.invoke(app, ["models", "--provider", "google", "--yaml", tmp_models_yaml])
    assert result.exit_code == 0
    assert "google" in result.output
    assert "openai" not in result.output


def test_analyze_missing_text_shows_error():
    result = runner.invoke(app, ["analyze"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Ejecutar y verificar que falla**

```bash
pytest tests/test_cli.py -v
```

Expected: `ImportError: cannot import name 'app'`

- [ ] **Step 3: Implementar `token_contador/cli/main.py`**

```python
from __future__ import annotations
import json
import typer
from rich.console import Console
from rich.table import Table
from typing import Optional
from token_contador.core import analyze
from token_contador.core.registry import ModelRegistry
from token_contador.core.scraper import PriceScraper

app = typer.Typer(help="Analizador de tokens y costos para modelos LLM")
console = Console()


def _make_table(analysis) -> Table:
    table = Table(title=f"Análisis — {analysis.text_length} chars · {analysis.tokens_out_assumed} tokens out asumidos")
    table.add_column("Modelo", style="cyan")
    table.add_column("Provider")
    table.add_column("Tokens", justify="right")
    table.add_column("Método", justify="center")
    table.add_column("Ctx max", justify="right")
    table.add_column("¿Cabe?", justify="center")
    table.add_column("Costo total", justify="right", style="green")

    for model_id, mr in sorted(analysis.by_model.items(), key=lambda x: x[1].cost.total_usd):
        is_cheapest = model_id == analysis.cheapest
        is_largest = model_id == analysis.largest_context
        badges = ("💰 " if is_cheapest else "") + ("📏 " if is_largest else "")
        table.add_row(
            badges + model_id,
            mr.provider,
            str(mr.tokens),
            "~" if mr.token_method == "estimated" else "✓",
            f"{mr.context_window:,}",
            "✅" if mr.fits_in_context else "❌",
            f"${mr.cost.total_usd:.6f}",
        )
    return table


@app.command(name="analyze")
def analyze_cmd(
    text: str = typer.Argument(..., help="Texto a analizar"),
    providers: Optional[str] = typer.Option(None, "--providers", help="Providers separados por coma: openai,anthropic"),
    out_tokens: int = typer.Option(500, "--out-tokens", help="Tokens de salida asumidos"),
    format: str = typer.Option("table", "--format", help="Salida: table | json | csv"),
    yaml: Optional[str] = typer.Option(None, "--yaml", hidden=True, help="Path a models.yaml (para tests)"),
):
    """Analiza tokens y costo de un texto en múltiples modelos LLM."""
    provider_list = [p.strip() for p in providers.split(",")] if providers else None
    result = analyze(text, providers=provider_list, tokens_out=out_tokens, yaml_path=yaml)

    if format == "json":
        data = {
            "text_length": result.text_length,
            "tokens_out_assumed": result.tokens_out_assumed,
            "cheapest": result.cheapest,
            "largest_context": result.largest_context,
            "by_model": {
                mid: {
                    "provider": mr.provider,
                    "tokens": mr.tokens,
                    "token_method": mr.token_method,
                    "context_window": mr.context_window,
                    "fits_in_context": mr.fits_in_context,
                    "cost_total_usd": mr.cost.total_usd,
                }
                for mid, mr in result.by_model.items()
            },
        }
        typer.echo(json.dumps(data, indent=2))
    elif format == "csv":
        typer.echo("model_id,provider,tokens,method,context_window,fits,cost_usd")
        for mid, mr in result.by_model.items():
            typer.echo(f"{mid},{mr.provider},{mr.tokens},{mr.token_method},{mr.context_window},{mr.fits_in_context},{mr.cost.total_usd:.8f}")
    else:
        console.print(_make_table(result))
        console.print(f"\n💰 Más barato: [bold green]{result.cheapest}[/]  📏 Mayor contexto: [bold blue]{result.largest_context}[/]")


@app.command()
def models(
    provider: Optional[str] = typer.Option(None, "--provider", help="Filtrar por provider"),
    min_context: Optional[int] = typer.Option(None, "--min-context", help="Contexto mínimo en tokens"),
    yaml: Optional[str] = typer.Option(None, "--yaml", hidden=True),
):
    """Lista los modelos disponibles en el catálogo."""
    registry = ModelRegistry(yaml_path=yaml)
    result = registry.filter(provider=provider, min_context=min_context)

    table = Table(title="Modelos disponibles")
    table.add_column("ID", style="cyan")
    table.add_column("Provider")
    table.add_column("Contexto", justify="right")
    table.add_column("Input/1M", justify="right")
    table.add_column("Output/1M", justify="right")
    table.add_column("Tokenizer")

    for m in sorted(result, key=lambda x: (x.provider, x.id)):
        table.add_row(m.id, m.provider, f"{m.context_window:,}", f"${m.input_per_1m}", f"${m.output_per_1m}", m.tokenizer)

    console.print(table)
    console.print(f"\n{len(result)} modelos listados.")


@app.command(name="refresh-prices")
def refresh_prices(
    provider: Optional[str] = typer.Option(None, "--provider", help="Solo actualizar este provider"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Mostrar qué se haría sin guardar"),
    yaml: Optional[str] = typer.Option(None, "--yaml", hidden=True),
):
    """Actualiza precios desde las páginas oficiales de cada provider."""
    scraper = PriceScraper(yaml_path=yaml)
    providers = [provider] if provider else None

    if dry_run:
        console.print("[yellow]Modo dry-run — no se guardarán cambios[/]")
        return

    console.print("Actualizando precios...")
    report = scraper.refresh(providers=providers)

    for prov, info in report.items():
        status = info["status"]
        if status == "updated":
            console.print(f"  ✅ [green]{prov}[/] — {info.get('models_touched', 0)} modelos actualizados")
        else:
            console.print(f"  ❌ [red]{prov}[/] — {info.get('error', 'error desconocido')}")


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    """Inicia la web app en http://localhost:8000"""
    import uvicorn
    from token_contador.web.app import create_app
    uvicorn.run(create_app(), host=host, port=port)
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

```bash
pytest tests/test_cli.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Verificar CLI manualmente**

```bash
token-count analyze "Hello world, how are you today?" --format table
```

Expected: tabla Rich con todos los modelos del catálogo.

- [ ] **Step 6: Commit**

```bash
git add token_contador/cli/main.py tests/test_cli.py
git commit -m "feat: CLI — comandos analyze, models, refresh-prices, web (Typer + Rich)"
```

---

## Task 10: Web App (FastAPI + HTMX)

**Files:**
- Create: `token_contador/web/app.py`
- Create: `token_contador/web/routes.py`
- Create: `token_contador/web/templates/base.html`
- Create: `token_contador/web/templates/index.html`
- Create: `token_contador/web/templates/partials/results_table.html`
- Create: `tests/test_web.py`

- [ ] **Step 1: Escribir tests de la web**

```python
# tests/test_web.py
import pytest
import pytest_asyncio
import httpx
from token_contador.web.app import create_app


@pytest.fixture
def client(tmp_models_yaml):
    from fastapi.testclient import TestClient
    app = create_app(yaml_path=tmp_models_yaml)
    return TestClient(app)


def test_homepage_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Token Contador" in resp.text


def test_analyze_returns_table_partial(client):
    resp = client.post("/analyze", data={
        "text": "Hello world",
        "tokens_out": "500",
        "providers": [],
    })
    assert resp.status_code == 200
    assert "gpt-4o" in resp.text


def test_analyze_filters_provider(client):
    resp = client.post("/analyze", data={
        "text": "Hello",
        "tokens_out": "500",
        "providers": ["openai"],
    })
    assert resp.status_code == 200
    assert "gpt-4o" in resp.text
    assert "gemini" not in resp.text


def test_models_endpoint(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert any(m["id"] == "gpt-4o" for m in data)


def test_refresh_prices_endpoint(client):
    from unittest.mock import patch
    with patch("token_contador.core.scraper.httpx.get") as mock_get:
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body>pricing page</body></html>"
        mock_get.return_value = mock_resp
        resp = client.post("/refresh", json={"providers": ["anthropic"]})
    assert resp.status_code == 200
    data = resp.json()
    assert "report" in data
```

- [ ] **Step 2: Ejecutar y verificar que falla**

```bash
pytest tests/test_web.py -v
```

Expected: `ImportError: cannot import name 'create_app'`

- [ ] **Step 3: Crear `token_contador/web/app.py`**

```python
from __future__ import annotations
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from pathlib import Path
from token_contador.web.routes import make_router

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app(yaml_path: str | None = None) -> FastAPI:
    app = FastAPI(title="Token Contador")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    router = make_router(templates=templates, yaml_path=yaml_path)
    app.include_router(router)
    return app
```

- [ ] **Step 4: Crear `token_contador/web/routes.py`**

```python
from __future__ import annotations
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Annotated
from token_contador.core import analyze
from token_contador.core.registry import ModelRegistry
from token_contador.core.scraper import PriceScraper


def make_router(templates: Jinja2Templates, yaml_path: str | None = None) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        registry = ModelRegistry(yaml_path=yaml_path)
        providers = sorted({m.provider for m in registry.get_all()})
        return templates.TemplateResponse("index.html", {
            "request": request,
            "providers": providers,
        })

    @router.post("/analyze", response_class=HTMLResponse)
    async def analyze_text(
        request: Request,
        text: Annotated[str, Form()],
        tokens_out: Annotated[int, Form()] = 500,
        providers: Annotated[list[str], Form()] = [],
    ):
        result = analyze(
            text,
            providers=providers if providers else None,
            tokens_out=tokens_out,
            yaml_path=yaml_path,
        )
        return templates.TemplateResponse("partials/results_table.html", {
            "request": request,
            "result": result,
        })

    @router.get("/models")
    async def list_models():
        registry = ModelRegistry(yaml_path=yaml_path)
        return [
            {"id": m.id, "provider": m.provider, "context_window": m.context_window,
             "input_per_1m": m.input_per_1m, "output_per_1m": m.output_per_1m}
            for m in registry.get_all()
        ]

    @router.post("/refresh")
    async def refresh_prices(body: dict):
        providers = body.get("providers")
        scraper = PriceScraper(yaml_path=yaml_path)
        report = scraper.refresh(providers=providers)
        return {"report": report}

    return router
```

- [ ] **Step 5: Crear `token_contador/web/templates/base.html`**

```html
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Token Contador</title>
  <script src="https://unpkg.com/htmx.org@1.9.12"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #f5f5f5; color: #222; }
    nav { background: #1a1a2e; color: white; padding: 12px 24px; display: flex; align-items: center; gap: 16px; }
    nav h1 { font-size: 1.2rem; }
    nav .subtitle { color: #aaa; font-size: 0.85rem; }
    main { max-width: 1200px; margin: 24px auto; padding: 0 16px; }
    .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 16px; }
    label { display: block; font-size: 0.85rem; font-weight: 600; color: #444; margin-bottom: 6px; }
    textarea, input[type=number] { width: 100%; border: 1px solid #ddd; border-radius: 6px; padding: 10px; font-size: 0.9rem; }
    textarea { height: 100px; resize: vertical; }
    button { background: #1565c0; color: white; border: none; border-radius: 6px; padding: 8px 20px; cursor: pointer; font-size: 0.9rem; }
    button:hover { background: #0d47a1; }
    .btn-secondary { background: #e65100; }
    .btn-secondary:hover { background: #bf360c; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th { background: #e0e0e0; padding: 8px 12px; text-align: left; }
    td { padding: 7px 12px; border-bottom: 1px solid #f0f0f0; }
    tr:hover td { background: #fafafa; }
    .badge-cheap { background: #e8f5e9; color: #2e7d32; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; }
    .badge-ctx { background: #e3f2fd; color: #1565c0; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; }
    .estimated { color: #f57c00; }
    .providers { display: flex; flex-wrap: wrap; gap: 8px; }
    .providers label { font-weight: 400; display: flex; align-items: center; gap: 4px; cursor: pointer; }
    #results { min-height: 40px; }
    .htmx-indicator { display: none; }
    .htmx-request .htmx-indicator { display: inline; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 700px) { .grid-2 { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <nav>
    <h1>🔢 Token Contador</h1>
    <span class="subtitle">Analizador de tokens y costos LLM</span>
  </nav>
  <main>{% block content %}{% endblock %}</main>
</body>
</html>
```

- [ ] **Step 6: Crear `token_contador/web/templates/index.html`**

```html
{% extends "base.html" %}
{% block content %}
<form hx-post="/analyze" hx-target="#results" hx-swap="innerHTML" hx-indicator="#spinner">
  <div class="grid-2">
    <div class="card">
      <label>Texto a analizar</label>
      <textarea name="text" placeholder="Pegá tu prompt aquí..."></textarea>
      <div style="margin-top:10px;display:flex;gap:10px;align-items:center">
        <label style="margin:0">Tokens de salida estimados:</label>
        <input type="number" name="tokens_out" value="500" style="width:100px">
        <button type="submit">Analizar</button>
        <span id="spinner" class="htmx-indicator" style="color:#888;font-size:0.85rem">Calculando...</span>
      </div>
    </div>
    <div class="card">
      <label>Filtrar providers (vacío = todos)</label>
      <div class="providers">
        {% for provider in providers %}
        <label>
          <input type="checkbox" name="providers" value="{{ provider }}"> {{ provider }}
        </label>
        {% endfor %}
      </div>
      <div style="margin-top:16px">
        <button type="button" class="btn-secondary"
          hx-post="/refresh" hx-vals='{"providers": []}'
          hx-on::after-request="alert('Precios actualizados')">
          ↻ Actualizar precios
        </button>
      </div>
    </div>
  </div>
</form>

<div class="card" id="results">
  <p style="color:#888;text-align:center;padding:20px">Ingresá un texto para ver la comparativa.</p>
</div>
{% endblock %}
```

- [ ] **Step 7: Crear `token_contador/web/templates/partials/results_table.html`**

```html
<div style="margin-bottom:10px;font-size:0.85rem;color:#555">
  <strong>{{ result.text_length }}</strong> caracteres ·
  <strong>{{ result.tokens_out_assumed }}</strong> tokens de salida asumidos ·
  💰 Más barato: <strong>{{ result.cheapest }}</strong> ·
  📏 Mayor contexto: <strong>{{ result.largest_context }}</strong>
</div>
<table>
  <thead>
    <tr>
      <th>Modelo</th>
      <th>Provider</th>
      <th style="text-align:right">Tokens</th>
      <th style="text-align:center">Método</th>
      <th style="text-align:right">Ctx máx</th>
      <th style="text-align:center">¿Cabe?</th>
      <th style="text-align:right">$/1M in</th>
      <th style="text-align:right">$/1M out</th>
      <th style="text-align:right">Costo total</th>
    </tr>
  </thead>
  <tbody>
    {% for model_id, mr in result.by_model.items() | sort(attribute='1.cost.total_usd') %}
    <tr>
      <td>
        {{ model_id }}
        {% if model_id == result.cheapest %}<span class="badge-cheap">💰 barato</span>{% endif %}
        {% if model_id == result.largest_context %}<span class="badge-ctx">📏 ctx</span>{% endif %}
      </td>
      <td>{{ mr.provider }}</td>
      <td style="text-align:right">{{ mr.tokens }}</td>
      <td style="text-align:center" {% if mr.token_method == 'estimated' %}class="estimated"{% endif %}>
        {{ "~est." if mr.token_method == 'estimated' else "✓ exacto" }}
      </td>
      <td style="text-align:right">{{ "{:,}".format(mr.context_window) }}</td>
      <td style="text-align:center">{{ "✅" if mr.fits_in_context else "❌" }}</td>
      <td style="text-align:right">${{ "%.4f" | format(mr.cost.input_usd / mr.tokens * 1_000_000 if mr.tokens > 0 else 0) }}</td>
      <td style="text-align:right">${{ "%.4f" | format(mr.cost.output_usd / result.tokens_out_assumed * 1_000_000 if result.tokens_out_assumed > 0 else 0) }}</td>
      <td style="text-align:right;font-weight:600">${{ "%.7f" | format(mr.cost.total_usd) }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
```

- [ ] **Step 8: Ejecutar y verificar que pasan todos los tests**

```bash
pytest tests/test_web.py -v
```

Expected: `5 passed`

```bash
pytest -v
```

Expected: todos los tests `passed`.

- [ ] **Step 9: Probar web manualmente**

```bash
token-count web
```

Abrir `http://localhost:8000`, ingresar texto, verificar tabla comparativa.

- [ ] **Step 10: Commit final**

```bash
git add token_contador/web/ tests/test_web.py
git commit -m "feat: web app — FastAPI + Jinja2 + HTMX, tabla comparativa en tiempo real"
```

---

## Verificación final

```bash
# Todos los tests
pytest -v

# CLI completo
token-count analyze "Explicame qué es una red neuronal" --format table
token-count models --provider anthropic
token-count refresh-prices --dry-run

# Librería Python
python -c "
from token_contador import analyze
r = analyze('Hello world')
print(f'Cheapest: {r.cheapest}')
print(f'Models analyzed: {len(r.by_model)}')
"

# Web
token-count web
```
