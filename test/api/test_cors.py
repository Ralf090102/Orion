"""Tests for the CORS policy in backend/app.py.

Orion is a local RAG assistant talking to the user's own Ollama instance and
knowledge base -- an open ("*") CORS policy would let any website the user
visits in a browser make requests to it. These tests pin the policy to the
app's actual known origins (Vite dev server, Tauri production webview) and
guard against it silently widening back to "*".
"""

import pytest

from backend.app import ALLOWED_ORIGIN_REGEX, ALLOWED_ORIGINS


@pytest.mark.unit
def test_allowed_origins_is_not_wildcard():
    assert "*" not in ALLOWED_ORIGINS
    assert ALLOWED_ORIGIN_REGEX != ".*"


@pytest.mark.unit
@pytest.mark.parametrize("origin", ["http://localhost:5173", "https://tauri.localhost", "tauri://localhost"])
def test_preflight_allows_known_origins(client, origin):
    response = client.options(
        "/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.headers.get("access-control-allow-origin") == origin


@pytest.mark.unit
def test_preflight_rejects_unknown_origin(client):
    response = client.options(
        "/health",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert "access-control-allow-origin" not in response.headers


@pytest.mark.unit
def test_actual_request_from_unknown_origin_gets_no_cors_header(client):
    # The request itself isn't blocked server-side (CORS is enforced by the
    # browser/webview, not the server) but the response must omit the
    # allow-origin header so the calling browser refuses to expose the body.
    response = client.get("/health", headers={"Origin": "https://evil.example.com"})

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
