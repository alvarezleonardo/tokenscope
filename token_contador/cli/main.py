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
