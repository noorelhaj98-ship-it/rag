"""API route definitions."""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import rbac_config, settings
from app.core.conversation import conversation_manager
from app.core.logging import rag_logger
from app.core.security import require_permission
from app.models import (
    APIInfoResponse,
    AskRequest,
    AskResponse,
    ClearResponse,
    HistoryResponse,
    RBACStatusResponse,
    Source,
)
from app.services.embedding import embedding_service
from app.services.llm import llm_service
from app.services.retrieval import retrieval_service

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create router
router = APIRouter()


def compress_context(items: List[Dict[str, Any]]) -> str:
    """Compress retrieved items into context string for LLM.

    Args:
        items: Retrieved documents with text and metadata

    Returns:
        Formatted context string
    """
    blocks = []
    total = 0

    for item in items:
        # Normalize whitespace
        text = " ".join(item["text"].split())
        snippet = text[: settings.max_context_chars_per_chunk]

        block = (
            f"[source: {item.get('source_file')} | "
            f"page: {item.get('page_number')} | "
            f"chunk_id: {item.get('chunk_id')}]\n"
            f"{snippet}\n"
        )

        if total + len(block) > settings.max_context_total_chars:
            break

        blocks.append(block)
        total += len(block)

    return "\n".join(blocks)


@router.get("/", response_class=FileResponse)
def root():
    """Serve the HTML UI."""
    return FileResponse("index.html")


@router.get("/api", response_model=APIInfoResponse)
def api_info():
    """Get API information."""
    return {
        "message": "RAG API is running",
        "rbac_enabled": rbac_config.enabled,
        "endpoints": ["/ask", "/history", "/clear", "/admin/rbac"],
    }


@router.get("/admin/rbac", response_model=RBACStatusResponse)
@limiter.limit(settings.rate_limit)
def get_rbac_status(
    request: Request,
    user: Dict[str, Any] = Depends(require_permission("admin")),
):
    """Admin endpoint to check RBAC configuration."""
    return {
        "rbac_enabled": rbac_config.enabled,
        "configured_users": list(rbac_config.api_keys.keys())
        if rbac_config.enabled
        else [],
        "role_permissions": rbac_config.ROLE_PERMISSIONS,
        "current_user": user.get("name"),
        "current_role": user.get("role"),
    }


@router.post("/ask", response_model=AskResponse)
@limiter.limit(settings.rate_limit)
def ask(
    req: AskRequest,
    request: Request,
    user: Dict[str, Any] = Depends(require_permission("ask")),
):
    """Ask a question and get an answer based on retrieved documents."""
    # Clear history if requested
    if req.clear_history:
        conversation_manager.clear()

    question = req.question
    history = conversation_manager.get_context()

    # Generate embedding and retrieve documents
    query_vector = embedding_service.embed(question)
    print(f"[hybrid] Searching for: {question[:50]}...")
    items = retrieval_service.retrieve_hybrid(question, query_vector)
    print(f"[hybrid] Found {len(items)} results using dense + BM25 RRF")

    if not items:
        answer = "No relevant documents found."
        conversation_manager.add("user", question)
        conversation_manager.add("assistant", answer)

        rag_logger.log_interaction(
            {
                "question": question,
                "answer": answer,
                "sources_count": 0,
                "sources": [],
                "history_used": bool(history),
                "user": user.get("name"),
            }
        )
        return {"answer": answer, "sources": []}

    # Build context and generate answer
    context = compress_context(items)
    print(
        f"\n[final context] Using {len(items)} chunks, "
        f"total context length: {len(context)} chars"
    )
    print(f"[final context] Pages used: {[i['page_number'] for i in items]}")

    previous_answer = conversation_manager.get_last_answer()
    answer = llm_service.generate_answer(question, context, history, previous_answer)

    # Update conversation history
    conversation_manager.add("user", question)
    conversation_manager.add("assistant", answer)

    # Build sources list
    sources = [
        Source(
            score=i["score"],
            source_file=i.get("source_file"),
            page_number=i.get("page_number"),
            chunk_id=i.get("chunk_id"),
        )
        for i in items
    ]

    # Log interaction
    rag_logger.log_interaction(
        {
            "question": question,
            "answer": answer,
            "sources_count": len(sources),
            "sources": [
                {"score": s.score, "page_number": s.page_number} for s in sources
            ],
            "history_used": bool(history),
            "user": user.get("name"),
        }
    )

    return {"answer": answer, "sources": sources}


@router.get("/history", response_model=HistoryResponse)
@limiter.limit(settings.rate_limit)
def get_history(
    request: Request,
    user: Dict[str, Any] = Depends(require_permission("history")),
):
    """Get conversation history."""
    return {
        "history": conversation_manager.history,
        "user": user.get("name"),
    }


@router.post("/clear", response_model=ClearResponse)
@limiter.limit(settings.rate_limit)
def clear_history(
    request: Request,
    user: Dict[str, Any] = Depends(require_permission("clear")),
):
    """Clear conversation history."""
    conversation_manager.clear()

    rag_logger.log_action(
        "clear_history",
        {"user": user.get("name"), "role": user.get("role")},
    )

    return {"message": "Conversation history cleared", "user": user.get("name")}
