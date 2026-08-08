"""Bounded HTTP latency probe with explicit blocked semantics."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

PASSED = 0
FAILED = 1
BLOCKED = 2


def percentile(values: list[float], rank: int) -> float:
    if not values:
        raise ValueError("at least one latency sample is required")
    ordered = sorted(values)
    index = max(0, math.ceil((rank / 100) * len(ordered)) - 1)
    return ordered[index]


def _sample(url: str, timeout: float) -> float:
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("probe URL must use http or https")
    started = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        response.read()
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"HTTP {response.status}")
    return (time.perf_counter() - started) * 1000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-var", required=True)
    parser.add_argument("--path", default="/health")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--p95-limit-ms", type=float, default=2000)
    args = parser.parse_args(argv)
    base_url = os.getenv(args.env_var, "").strip().rstrip("/")
    if not base_url:
        print(f"BLOCKED: set {args.env_var}")
        return BLOCKED
    if args.samples < 1 or args.concurrency < 1:
        print("FAILED: samples and concurrency must be positive")
        return FAILED
    url = base_url + "/" + args.path.lstrip("/")
    latencies: list[float] = []
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(_sample, url, args.timeout) for _ in range(args.samples)]
            for future in as_completed(futures):
                latencies.append(future.result())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"BLOCKED: endpoint unavailable ({exc.__class__.__name__})")
        return BLOCKED
    except (RuntimeError, ValueError) as exc:
        print(f"FAILED: {exc}")
        return FAILED
    metrics = {
        "url": url,
        "samples": len(latencies),
        "p50_ms": round(percentile(latencies, 50), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
        "limit_p95_ms": args.p95_limit_ms,
    }
    print(json.dumps(metrics, ensure_ascii=False))
    return PASSED if metrics["p95_ms"] <= args.p95_limit_ms else FAILED


if __name__ == "__main__":
    raise SystemExit(main())
