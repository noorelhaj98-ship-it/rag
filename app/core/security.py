"""Security, authentication, and RBAC."""

from typing import Any, Dict

from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import rbac_config

security = HTTPBearer(auto_error=False)


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> Dict[str, Any]:
    """Verify API token and return user info.

    If RBAC is disabled, returns anonymous admin user.
    """
    if not rbac_config.enabled:
        return {"name": "anonymous", "role": "admin"}

    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    token = credentials.credentials
    user = rbac_config.get_user(token)

    if not user:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return user


def require_permission(permission: str):
    """Dependency factory to require specific permission."""

    def checker(user: Dict[str, Any] = Depends(verify_token)) -> Dict[str, Any]:
        role = user.get("role", "readonly")
        if not rbac_config.has_permission(role, permission):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: '{permission}' required",
            )
        return user

    return checker
