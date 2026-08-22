"""Keycloak OIDC bearer token verification for full mode."""

from __future__ import annotations

from contextvars import ContextVar
from functools import lru_cache
from typing import Any

from app.config import get_settings
from app.core.exceptions import PermissionError, ValidationError

_claims: ContextVar[dict[str, Any] | None] = ContextVar("devmate_oidc_claims", default=None)


def validate_full_mode_oidc() -> None:
    """full 模式缺少任一 OIDC 验证参数时拒绝启动。"""
    settings = get_settings()
    if settings.app_mode != "full":
        return
    missing = [
        name
        for name, value in (
            ("OIDC_ISSUER_URL", settings.oidc_issuer_url),
            ("OIDC_JWKS_URL", settings.oidc_jwks_url),
            ("OIDC_AUDIENCE", settings.oidc_audience),
        )
        if not value.strip()
    ]
    if missing:
        raise ValidationError("full mode requires " + ", ".join(missing))


@lru_cache(maxsize=4)
def _jwks_client(url: str) -> Any:
    from jwt import PyJWKClient

    return PyJWKClient(url, cache_keys=True, lifespan=300)


def verify_bearer(token: str) -> dict[str, Any]:
    import jwt

    settings = get_settings()
    issuer = settings.oidc_issuer_url.rstrip("/")
    validate_full_mode_oidc()
    try:
        signing_key = _jwks_client(settings.oidc_jwks_url).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=issuer,
            audience=settings.oidc_audience,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except (jwt.PyJWTError, OSError, ValueError) as exc:
        raise PermissionError("OIDC token verification failed") from exc
    tenant_id = claims.get("tenant_id")
    realm_access = claims.get("realm_access", {})
    roles = realm_access.get("roles", []) if isinstance(realm_access, dict) else []
    allowed = {"devmate-user", "devmate-approver", "devmate-admin"}
    if not isinstance(tenant_id, str) or not tenant_id:
        raise PermissionError("OIDC tenant_id claim is required")
    if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
        raise PermissionError("OIDC role claims are invalid")
    if not allowed.intersection(roles):
        raise PermissionError("DevMate role is required")
    return claims


def set_claims(claims: dict[str, Any]):
    return _claims.set(claims)


def reset_claims(token: Any) -> None:
    _claims.reset(token)


def current_tenant_id() -> str:
    claims = _claims.get()
    if claims is None:
        raise PermissionError("OIDC identity context is unavailable")
    return str(claims["tenant_id"])
