"""Pydantic models for API requests and responses."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AskRequest(BaseModel):
    """Request model for the /ask endpoint."""

    question: str
    clear_history: bool = False


class Source(BaseModel):
    """Source document metadata."""

    score: float
    source_file: Optional[str]
    page_number: Optional[int]
    chunk_id: Optional[str]


class AskResponse(BaseModel):
    """Response model for the /ask endpoint."""

    answer: str
    sources: List[Source]


class HistoryResponse(BaseModel):
    """Response model for the /history endpoint."""

    history: List[Dict[str, str]]
    user: Optional[str]


class ClearResponse(BaseModel):
    """Response model for the /clear endpoint."""

    message: str
    user: Optional[str]


class APIInfoResponse(BaseModel):
    """Response model for the /api endpoint."""

    message: str
    rbac_enabled: bool
    endpoints: List[str]


class RBACStatusResponse(BaseModel):
    """Response model for the /admin/rbac endpoint."""

    rbac_enabled: bool
    configured_users: List[str]
    role_permissions: Dict[str, List[str]]
    current_user: Optional[str]
    current_role: Optional[str]
