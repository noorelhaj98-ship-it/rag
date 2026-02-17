"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.routes import limiter, router
from app.config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="RAG API",
        description="Retrieval-Augmented Generation API with hybrid search",
        version="1.0.0",
    )

    # Attach rate limiter to app state
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, lambda req, exc: ...)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files (HTML UI)
    app.mount("/static", StaticFiles(directory="."), name="static")

    # Include API routes
    app.include_router(router)

    return app


# Create application instance
app = create_app()
