# Token Contador

Herramienta para analizar y comparar el uso de tokens y costos entre modelos LLM.

**22 modelos · 9 providers · CLI + Web App + Librería Python**

## Instalación

```bash
pip install -e ".[dev]"
```

## CLI

### Analizar un texto

```bash
token-count analyze "Explicame qué es una red neuronal"
```

```
                   Análisis — 36 chars · 500 tokens out asumidos
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Modelo          ┃ Provider  ┃ Tokens ┃ Método ┃   Ctx max ┃ ¿Cabe? ┃ Costo total ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━┩
│ 💰 llama-3.1-8b │ meta      │     9  │   ~    │   128,000 │   ✅   │   $0.000005 │
│ deepseek-v3     │ deepseek  │    10  │   ~    │    64,000 │   ✅   │   $0.000015 │
│ gemini-2.0-fl.. │ google    │     9  │   ~    │ 1,000,000 │   ✅   │   $0.000151 │
│ gpt-4o          │ openai    │     9  │   ✓    │   128,000 │   ✅   │   $0.005023 │
│ claude-sonnet.. │ anthropic │     9  │   ✓    │   200,000 │   ✅   │   $0.007527 │
│ 📏 gemini-1.5-p │ google    │     9  │   ~    │ 2,000,000 │   ✅   │   $0.002523 │
└─────────────────┴───────────┴────────┴────────┴───────────┴────────┴─────────────┘

💰 Más barato: llama-3.1-8b    📏 Mayor contexto: gemini-1.5-pro
```

### Filtrar providers

```bash
token-count analyze "tu prompt" --providers anthropic,openai
```

### Salida JSON (para scripts)

```bash
token-count analyze "tu prompt" --format json
```

```json
{
  "text_length": 10,
  "tokens_out_assumed": 500,
  "cheapest": "llama-3.1-8b",
  "largest_context": "gemini-1.5-pro",
  "by_model": {
    "gpt-4o": {
      "provider": "openai",
      "tokens": 3,
      "token_method": "exact",
      "context_window": 128000,
      "fits_in_context": true,
      "cost_total_usd": 0.005008
    }
  }
}
```

### Salida CSV

```bash
token-count analyze "tu prompt" --format csv > resultados.csv
```

### Listar modelos disponibles

```bash
token-count models
token-count models --provider anthropic
token-count models --min-context 500000   # modelos con ventana ≥ 500K tokens
```

### Actualizar precios desde fuentes oficiales

```bash
token-count refresh-prices                    # todos los providers
token-count refresh-prices --provider openai  # solo OpenAI
token-count refresh-prices --dry-run          # ver qué haría sin cambios
```

### Iniciar web app

```bash
token-count web
# → http://localhost:8000
```

## Web App

```bash
token-count web
```

Abrí `http://localhost:8000` — interfaz visual con tabla comparativa en tiempo real:

- Pegá tu texto o cargá un archivo
- Seleccioná providers con checkboxes
- Ajustá tokens de salida estimados
- La tabla se actualiza sin recargar la página (HTMX)
- Botón para actualizar precios desde fuentes oficiales

## Librería Python

```python
from token_contador import analyze, compare

# Analizar un texto contra todos los modelos
result = analyze("Hola, ¿cómo estás?")

print(result.cheapest)           # "llama-3.1-8b"
print(result.largest_context)    # "gemini-1.5-pro"
print(result.text_length)        # 18

# Ver resultado de un modelo específico
mr = result.by_model["gpt-4o"]
print(mr.tokens)                 # 7
print(mr.token_method)           # "exact"
print(mr.cost.total_usd)         # 0.005018
print(mr.fits_in_context)        # True

# Filtrar por providers
result = analyze("tu texto", providers=["anthropic", "openai"])

# Cambiar tokens de salida asumidos (default: 500)
result = analyze("tu texto", tokens_out=2000)

# Analizar modelos específicos
result = analyze("tu texto", model_ids=["gpt-4o", "claude-sonnet-4-6"])

# Comparar variantes de un prompt contra el mismo modelo
textos = [
    "Explicame microservicios",
    "Explicame microservicios en 3 párrafos con ejemplos prácticos",
]
results = compare(textos, model_id="gpt-4o")
for i, r in enumerate(results):
    print(f"Texto {i+1}: {r.by_model['gpt-4o'].tokens} tokens — ${r.by_model['gpt-4o'].cost.total_usd:.6f}")
```

## Providers incluidos

| Provider   | Modelos                                                  | Tokenizer       |
|------------|----------------------------------------------------------|-----------------|
| Anthropic  | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5     | tiktoken exacto |
| OpenAI     | gpt-4o, gpt-4o-mini, o1, o1-mini, o3-mini               | tiktoken exacto |
| Google     | gemini-2.0-flash, gemini-2.5-pro, gemini-1.5-pro/flash  | estimado        |
| Meta       | llama-3.1-8b, llama-3.1-70b, llama-3.3-70b              | HF / estimado   |
| Mistral    | mistral-large, mistral-small, codestral                  | HF / estimado   |
| Cohere     | command-r-plus, command-r                                | estimado        |
| Deepseek   | deepseek-v3, deepseek-r1                                 | estimado        |

> **`~`** = conteo estimado (chars / ratio). **`✓`** = conteo exacto via tiktoken.

## Agregar modelos custom

Creá `~/.token_contador/custom_models.yaml` con el mismo formato que `models.yaml`:

```yaml
models:
  - id: mi-modelo-interno
    provider: mi-empresa
    context_window: 32000
    tokenizer: estimated
    chars_per_token: 4.0
    input_per_1m: 0.50
    output_per_1m: 1.50
    cached_input_per_1m: null
    pricing_url: https://mi-empresa.com/pricing
    updated_at: "2026-04-09"
```

## Desarrollo

```bash
# Tests
pytest

# Tests con verbose
pytest -v

# Solo unit tests
pytest tests/unit/

# Solo CLI tests
pytest tests/test_cli.py
```

## Arquitectura

```
token_contador/
├── core/          # Librería pura: tokenizer, registry, pricing, scraper
├── cli/           # CLI Typer con Rich tables
├── web/           # FastAPI + Jinja2 + HTMX
└── data/          # models.yaml — catálogo de modelos y precios
```

El core es independiente — CLI y Web son clientes del mismo core.
