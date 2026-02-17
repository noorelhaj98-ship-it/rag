"""FastAPI dependencies."""

from typing import Any, Dict

from fastapi import Depends

from app.core.security import require_permission, verify_token


# Re-export for convenience
verify_token = verify_token
require_permission = require_permission


# Common dependency aliases
get_current_user = verify_token
require_ask = Depends(require_permission("ask"))
require_history = Depends(require_permission("history"))
require_clear = Depends(require_permission("clear"))
require_admin = Depends(require_permission("admin"))
