# Token Contador — Design Spec
**Fecha:** 2026-04-09  
**Estado:** Aprobado

---

## Objetivo

Herramienta para analizar y comparar el uso de tokens y costos entre distintos modelos LLM. Cubre dos casos de uso principales:

- **Estimación de costos**: dado un texto, ¿cuánto cuesta procesarlo en cada modelo?
- **Comparativa de contexto**: ¿qué ventana de contexto ofrece cada modelo y cuánto del texto cabe?

---

## Arquitectura — Capas Independientes

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   CLI LAYER     │  │   WEB LAYER     │  │  PYTHON LIB     │
│  (Typer/Rich)   │  │ (FastAPI/HTMX)  │  │  (importable)   │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                     │
         └────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   CORE LIBRARY    │
                    │                   │
                    │ TokenizerEngine   │
                    │ ModelRegistry     │
                    │ PricingEngine     │
                    └─────────┬─────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
     ┌────────▼────────┐           ┌──────────▼──────────┐
     │   models.yaml   │           │   PriceScraper      │
     │  (fuente local) │           │  (refresh on-demand)│
     └─────────────────┘           └─────────────────────┘
```

### Estructura de directorios

```
token_contador/
├── core/
│   ├── __init__.py          # API pública: analyze(), compare()
│   ├── tokenizer.py         # TokenizerEngine
│   ├── registry.py          # ModelRegistry
│   ├── pricing.py           # PricingEngine
│   ├── scraper.py           # PriceScraper
│   └── models.py            # Dataclasses: ModelResult, AnalysisResult
├── cli/
│   ├── __init__.py
│   └── main.py              # Typer app con comandos: analyze, compare, refresh-prices
├── web/
│   ├── __init__.py
│   ├── app.py               # FastAPI app
│   ├── routes.py            # Endpoints REST + HTMX
│   └── templates/           # Jinja2 templates
│       ├── base.html
│       └── index.html
├── data/
│   └── models.yaml          # Catálogo de modelos y precios
├── tests/
│   ├── unit/
│   │   ├── test_tokenizer.py
│   │   ├── test_registry.py
│   │   └── test_pricing.py
│   ├── test_cli.py
│   └── test_web.py
├── pyproject.toml
└── README.md
```

---

## Componentes del Core

### TokenizerEngine

Cuenta tokens para un texto dado un modelo específico.

- **Modelos OpenAI / Claude con tiktoken**: usa `tiktoken` directamente → conteo exacto
- **Modelos HuggingFace (Meta Llama, Mistral, etc.)**: usa `tokenizers` (HF) → conteo exacto
- **Modelos sin tokenizer oficial** (Cohere, AI21, etc.): estimación basada en `ratio chars/token` definido en `models.yaml`
- Siempre indica en el resultado si el conteo es exacto o estimado

```python
engine = TokenizerEngine()
result = engine.count("Hola mundo", model="gpt-4o")
# TokenCount(tokens=3, method="exact", tokenizer="tiktoken")
```

La API pública del core expone dos funciones:
- `analyze(text, models=None, tokens_out=500)` → `AnalysisResult` — analiza un texto contra múltiples modelos
- `compare(texts: list[str], model, tokens_out=500)` → `list[AnalysisResult]` — compara múltiples textos contra un mismo modelo (útil para comparar variantes de un prompt)

### ModelRegistry

Fuente de verdad sobre los modelos disponibles.

- Lee `data/models.yaml` al inicio; cachea en memoria (TTL: 1 hora)
- Soporta filtros por provider, familia, contexto mínimo
- El usuario puede agregar modelos custom en `~/.token_contador/custom_models.yaml`

Campos por modelo en YAML:
```yaml
- id: claude-opus-4-5
  provider: anthropic
  context_window: 200000
  tokenizer: tiktoken-cl100k
  pricing:
    input_per_1m: 15.00
    output_per_1m: 75.00
    cached_input_per_1m: 1.50   # si aplica
  pricing_url: https://www.anthropic.com/pricing
  updated_at: "2026-04-09"
```

### PricingEngine

Calcula costo estimado dado un conteo de tokens.

```python
cost = pricing.calculate(
    tokens_in=21,
    tokens_out=500,   # estimado de salida, configurable
    model="claude-opus-4-5"
)
# Cost(input_usd=0.000315, output_usd=0.0375, total_usd=0.037815)
```

### PriceScraper

Actualiza precios desde fuentes oficiales de cada provider.

- Se ejecuta con `token-count refresh-prices` (CLI) o botón en la web
- Fuentes: páginas oficiales de pricing de cada provider (Anthropic, OpenAI, Google, etc.)
- Actualiza `data/models.yaml` con nuevos precios y `updated_at`
- **Fallback**: si el scraping falla para un provider, conserva precios anteriores y registra advertencia con timestamp de última actualización exitosa

### AnalysisResult (dataclasses)

```python
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
    cheapest: str       # model_id del más barato
    largest_context: str  # model_id con mayor ventana
```

---

## CLI

Comandos disponibles via `token-count`:

```bash
# Analizar texto inline
token-count analyze "texto aquí" [--models anthropic,openai] [--out-tokens 500]

# Analizar archivo
token-count analyze --file prompt.txt [--models all] [--format json|csv|table]

# Comparativa de contexto (sin costo, foco en ventana)
token-count context "texto" [--models all]

# Actualizar precios desde fuentes oficiales
token-count refresh-prices [--provider anthropic] [--dry-run]

# Listar modelos disponibles
token-count models [--provider openai] [--min-context 100000]
```

Output por defecto: tabla Rich en terminal con colores.  
`--format json` para integrar en scripts. `--format csv` para exportar.

---

## Web App

- **URL**: `http://localhost:8000`
- **Stack**: FastAPI + Jinja2 + HTMX (sin build steps, sin JS framework)
- **Inicio**: `token-count web` o `uvicorn token_contador.web.app:app`

### Funcionalidades

- Input de texto (textarea) o carga de archivo
- Selección de providers/modelos con checkboxes
- Slider para tokens de salida estimados (default: 500)
- Tabla comparativa actualizada en tiempo real vía HTMX
- Indicador de última actualización de precios + botón refresh
- Highlight del modelo más barato y el de mayor contexto
- Export de resultados como CSV o JSON

---

## Flujo de datos completo

```
1. Usuario ingresa texto
         │
2. TokenizerEngine.count(text, model) para cada modelo seleccionado
   ├── tokenizer exacto disponible → tiktoken / HF tokenizers
   └── no disponible → estimación con ratio del YAML
         │
3. ModelRegistry.get_models(filters) → lista de ModelInfo
         │
4. PricingEngine.calculate(tokens_in, tokens_out, model) → Cost
         │
5. Construir AnalysisResult con todos los ModelResult
         │
6. Renderizar:
   ├── CLI: Rich Table
   ├── Web: HTMX partial render de tabla
   └── Lib: devuelve AnalysisResult directamente
```

---

## Providers incluidos en models.yaml inicial

| Provider   | Modelos incluidos                                              |
|------------|---------------------------------------------------------------|
| Anthropic  | Claude Opus 4.5/4.6, Sonnet 4.6, Haiku 4.5                   |
| OpenAI     | GPT-4o, GPT-4o-mini, GPT-4-turbo, o1, o1-mini, o3-mini       |
| Google     | Gemini 2.0 Flash, Gemini 2.5 Pro, Gemini 1.5 Pro/Flash       |
| Meta       | Llama 3.1 8B/70B/405B, Llama 3.3 70B                         |
| Mistral    | Mistral Large, Mistral Small, Codestral                       |
| Cohere     | Command R+, Command R                                         |
| Deepseek   | DeepSeek V3, DeepSeek R1                                      |
| Groq       | (modelos hosteados: Llama, Mixtral via Groq)                  |
| Together   | (modelos open source via Together AI)                         |

El usuario puede agregar modelos custom via `~/.token_contador/custom_models.yaml`.

---

## Manejo de errores

| Escenario | Comportamiento |
|-----------|----------------|
| Scraping falla | Usa precios locales + muestra `[precios del YYYY-MM-DD]` |
| Tokenizer no disponible | Usa estimación + marca resultado como `~` (estimado) |
| Modelo custom con YAML malformado | Error claro: `campo 'pricing.input_per_1m' faltante en modelo 'mi-modelo'` |
| Archivo de entrada no existe | Error inmediato con mensaje claro |
| Provider desconocido en filtro | Advertencia + lista providers disponibles |

---

## Testing

```
tests/unit/
  test_tokenizer.py   — conteo exacto vs estimado, 5 textos fixture de distintos tamaños
  test_registry.py    — filtros, modelos custom, cache TTL
  test_pricing.py     — cálculos de costo, edge cases (tokens=0, precios=0)

tests/test_cli.py     — comandos con CliRunner de Typer
tests/test_web.py     — endpoints con httpx.AsyncClient contra FastAPI

Fixture de textos:
  - texto_corto: 50 chars (~12 tokens)
  - texto_medio: 500 chars (~125 tokens)
  - texto_largo: 5000 chars (~1250 tokens)
  - texto_codigo: snippet Python 300 chars
  - texto_multiidioma: mezcla ES/EN/ZH
```

---

## Instalación (target)

```bash
pip install token-contador

# CLI
token-count analyze "mi texto"

# Web
token-count web

# Librería
from token_contador import analyze
result = analyze("mi texto")
```

---

## Dependencias principales

| Dependencia | Uso |
|-------------|-----|
| `typer` + `rich` | CLI |
| `fastapi` + `uvicorn` | Web server |
| `jinja2` | Templates web |
| `tiktoken` | Tokenizer OpenAI/Anthropic |
| `tokenizers` | Tokenizer HuggingFace (Meta, Mistral) |
| `httpx` | Scraping de precios + tests |
| `beautifulsoup4` | Parsing HTML en scraper |
| `pyyaml` | Lectura de models.yaml |
| `pytest` | Tests |
