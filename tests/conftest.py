import pytest
import yaml


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
    "corto": "Hola mundo, esto es una prueba.",
    "medio": "Explicame la diferencia entre arquitectura monolítica y microservicios. " * 5,
    "largo": "En el campo de la inteligencia artificial, los modelos de lenguaje de gran escala han transformado la manera en que interactuamos con las computadoras. " * 20,
    "codigo": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint([fibonacci(i) for i in range(10)])",
    "multiidioma": "Hello world. Hola mundo. 你好世界. Bonjour le monde. Привет мир.",
}

@pytest.fixture
def sample_texts():
    return SAMPLE_TEXTS
