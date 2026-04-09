import pytest
from fastapi.testclient import TestClient
from token_contador.web.app import create_app


@pytest.fixture
def client(tmp_models_yaml):
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
    })
    assert resp.status_code == 200
    assert "gpt-4o" in resp.text


def test_analyze_filters_provider(client):
    resp = client.post("/analyze", data={
        "text": "Hello",
        "tokens_out": "500",
        "providers": "openai",
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
    from unittest.mock import patch, MagicMock
    with patch("httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body>pricing page</body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        resp = client.post("/refresh", json={"providers": ["anthropic"]})
    assert resp.status_code == 200
    data = resp.json()
    assert "report" in data
