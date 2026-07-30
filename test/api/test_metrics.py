"""Tests for the request-tracking middleware and /api/metrics endpoint.

These exercise real middleware behavior through the TestClient (not just
the metrics_collector module in isolation), because the interesting bug
surface here is the wiring: does every request actually get recorded, is
it grouped by route template instead of raw path, does an error response
get counted correctly.
"""

import pytest


@pytest.mark.unit
def test_metrics_start_empty_after_reset(client):
    response = client.get("/api/metrics")

    body = response.json()["metrics"]
    assert body["total_requests"] == 0
    assert body["total_errors"] == 0
    assert body["uptime_seconds"] >= 0


@pytest.mark.unit
def test_metrics_count_prior_requests_but_not_the_request_fetching_them(client):
    client.get("/health")
    client.get("/health")

    response = client.get("/api/metrics")
    body = response.json()["metrics"]

    # Both /health calls are counted; this /api/metrics call isn't counted
    # in its own snapshot (it's recorded by middleware after the handler runs).
    assert body["total_requests"] == 2
    assert body["by_endpoint"]["GET /health"]["count"] == 2


@pytest.mark.unit
def test_metrics_group_by_route_template_not_raw_path(client):
    # /api/config?detailed=true and /api/config both resolve to the same route.
    client.get("/api/config")
    client.get("/api/config?detailed=true")

    response = client.get("/api/metrics")
    body = response.json()["metrics"]

    assert body["by_endpoint"]["GET /api/config"]["count"] == 2


@pytest.mark.unit
def test_metrics_track_server_errors(client, monkeypatch):
    def broken_db_stats():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("backend.api.health.get_database_stats", broken_db_stats)

    error_response = client.get("/api/db/stats")
    assert error_response.status_code == 500

    response = client.get("/api/metrics")
    body = response.json()["metrics"]

    assert body["total_errors"] == 1
    assert body["by_endpoint"]["GET /api/db/stats"]["error_count"] == 1


@pytest.mark.unit
def test_metrics_average_latency_is_nonnegative(client):
    client.get("/health")

    response = client.get("/api/metrics")
    body = response.json()["metrics"]

    assert body["average_latency_ms"] >= 0
    assert body["by_endpoint"]["GET /health"]["average_latency_ms"] >= 0
