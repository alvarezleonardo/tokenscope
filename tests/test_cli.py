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
