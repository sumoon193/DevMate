"""full 模式身份配置必须 fail closed。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api.dependencies import reset_container
from app.config import get_settings
from app.core.exceptions import ValidationError
from app.services.security.oidc import validate_full_mode_oidc


def test_full_mode_rejects_missing_oidc_configuration(monkeypatch) -> None:
    monkeypatch.setenv("APP_MODE", "full")
    monkeypatch.setenv("OIDC_ISSUER_URL", "")
    monkeypatch.setenv("OIDC_JWKS_URL", "")
    monkeypatch.setenv("OIDC_AUDIENCE", "")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValidationError, match="OIDC_ISSUER_URL"):
            validate_full_mode_oidc()
    finally:
        get_settings.cache_clear()
        reset_container()


def test_frozen_container_lock_includes_oidc_runtime_dependency() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    lock = (root / "uv.lock").read_text(encoding="utf-8")
    assert "PyJWT[crypto]" in pyproject
    assert 'name = "pyjwt"' in lock
