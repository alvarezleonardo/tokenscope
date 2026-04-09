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
