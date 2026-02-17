"""RAG API Server - Entry point.

This module serves as the entry point for the RAG application.
It imports and re-exports the FastAPI app from the modular app package.
"""

# Import the app from the modular structure
from app.main import app

# Re-export for compatibility with existing deployments
__all__ = ["app"]
