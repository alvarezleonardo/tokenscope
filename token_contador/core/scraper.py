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
                # Update updated_at for all models of this provider
                updated = 0
                for model in data["models"]:
                    if model["provider"] == provider:
                        model["updated_at"] = str(date.today())
                        updated += 1
                report[provider] = {"status": "updated", "models_touched": updated}
            except Exception as e:
                report[provider] = {"status": "failed", "error": str(e)}

        # Only write if at least one provider succeeded
        if any(v["status"] == "updated" for v in report.values()):
            Path(self._yaml_path).write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))

        return report
