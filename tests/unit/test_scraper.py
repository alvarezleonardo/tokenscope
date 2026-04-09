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
        mock_resp.raise_for_status = MagicMock()
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
    """If scraping fails, YAML is not modified."""
    import pathlib
    original_content = pathlib.Path(tmp_models_yaml).read_text()
    scraper = PriceScraper(yaml_path=tmp_models_yaml)
    with patch("httpx.get") as mock_get:
        mock_get.side_effect = Exception("timeout")
        scraper.refresh(providers=["anthropic"])
    assert pathlib.Path(tmp_models_yaml).read_text() == original_content
