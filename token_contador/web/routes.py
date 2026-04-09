from __future__ import annotations
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Annotated, Optional
from token_contador.core import analyze
from token_contador.core.registry import ModelRegistry
from token_contador.core.scraper import PriceScraper


def make_router(templates: Jinja2Templates, yaml_path: str | None = None) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        registry = ModelRegistry(yaml_path=yaml_path)
        providers = sorted({m.provider for m in registry.get_all()})
        return templates.TemplateResponse(request, "index.html", {
            "providers": providers,
        })

    @router.post("/analyze", response_class=HTMLResponse)
    async def analyze_text(
        request: Request,
        text: Annotated[str, Form()],
        tokens_out: Annotated[int, Form()] = 500,
        providers: Annotated[Optional[str], Form()] = None,
    ):
        provider_list = [p.strip() for p in providers.split(",")] if providers else None
        result = analyze(
            text,
            providers=provider_list,
            tokens_out=tokens_out,
            yaml_path=yaml_path,
        )
        return templates.TemplateResponse(request, "partials/results_table.html", {
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
