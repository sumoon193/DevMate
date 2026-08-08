from scripts.devmate import http_probe


def test_missing_url_is_blocked(monkeypatch) -> None:
    monkeypatch.delenv("DEVMATE_PUBLIC_BASE_URL", raising=False)
    assert http_probe.main(["--env-var", "DEVMATE_PUBLIC_BASE_URL"]) == 2


def test_percentiles_are_nearest_rank() -> None:
    assert http_probe.percentile([10.0, 20.0, 30.0, 40.0], 95) == 40.0
