"""Structured logging setup with Logfire integration."""

import json
import logging
from datetime import datetime
from typing import Any, Dict

import logfire

from app.config import settings


class RAGLogger:
    """Structured logger for RAG interactions."""

    def __init__(self):
        self._logger = logging.getLogger("rag_server")
        self._logger.setLevel(logging.INFO)

        # File handler for JSONL logs
        file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(file_handler)

        # Logfire setup
        if settings.logfire_token:
            logfire.configure(token=settings.logfire_token, send_to_logfire=True)
        else:
            logfire.configure(send_to_logfire=False)

    def log_interaction(self, entry: Dict[str, Any]) -> None:
        """Log a RAG interaction to both file and Logfire."""
        entry["timestamp"] = datetime.utcnow().isoformat()
        self._logger.info(json.dumps(entry, ensure_ascii=False))
        logfire.info("rag_query", **entry)

    def log_action(self, action: str, details: Dict[str, Any]) -> None:
        """Log a generic action."""
        entry = {
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
            **details,
        }
        self._logger.info(json.dumps(entry, ensure_ascii=False))
        logfire.info(f"rag_{action}", **details)


# Global logger instance
rag_logger = RAGLogger()
