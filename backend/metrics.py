"""In-memory request metrics for the Orion backend.

Orion runs as a single desktop-sidecar process (no multi-worker deployment,
no restarts between requests), so a process-local, non-persisted counter is
the right amount of complexity -- no Prometheus client, no external store.
If Orion ever runs as a multi-worker service, this would need to move to
something shared (e.g. Prometheus client with a multiprocess registry).
"""

import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class _EndpointStats:
    count: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0

    @property
    def average_latency_ms(self) -> float:
        return self.total_latency_ms / self.count if self.count else 0.0


class MetricsCollector:
    """Tracks request counts, latency, and errors per route."""

    def __init__(self):
        self._start_time = time.monotonic()
        self._by_endpoint: dict[str, _EndpointStats] = defaultdict(_EndpointStats)

    def record(self, endpoint: str, latency_ms: float, is_error: bool) -> None:
        stats = self._by_endpoint[endpoint]
        stats.count += 1
        stats.total_latency_ms += latency_ms
        if is_error:
            stats.error_count += 1

    def reset(self) -> None:
        self._start_time = time.monotonic()
        self._by_endpoint.clear()

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def total_requests(self) -> int:
        return sum(stats.count for stats in self._by_endpoint.values())

    @property
    def total_errors(self) -> int:
        return sum(stats.error_count for stats in self._by_endpoint.values())

    @property
    def average_latency_ms(self) -> float:
        total_count = self.total_requests
        if not total_count:
            return 0.0
        total_latency = sum(stats.total_latency_ms for stats in self._by_endpoint.values())
        return total_latency / total_count

    def snapshot(self) -> dict:
        return {
            "uptime_seconds": round(self.uptime_seconds, 2),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "average_latency_ms": round(self.average_latency_ms, 2),
            "by_endpoint": {
                endpoint: {
                    "count": stats.count,
                    "average_latency_ms": round(stats.average_latency_ms, 2),
                    "error_count": stats.error_count,
                }
                for endpoint, stats in sorted(self._by_endpoint.items())
            },
        }


# Process-wide singleton -- imported by both the tracking middleware
# (backend/app.py) and the /api/metrics route (backend/api/health.py).
metrics_collector = MetricsCollector()
