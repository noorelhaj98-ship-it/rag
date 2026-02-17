"""Conversation history management."""

from typing import Dict, List
from threading import Lock

from app.config import settings


class ConversationManager:
    """Thread-safe conversation history manager."""

    def __init__(self, max_history: int = None):
        self.max_history = max_history or settings.max_history
        self._history: List[Dict[str, str]] = []
        self._lock = Lock()

    def add(self, role: str, content: str) -> None:
        """Add a message to history."""
        with self._lock:
            self._history.append({"role": role, "content": content})
            # Keep last N exchanges (2 messages per exchange)
            max_messages = self.max_history * 2
            if len(self._history) > max_messages:
                self._history.pop(0)

    def get_context(self, last_n_exchanges: int = 3) -> str:
        """Get formatted conversation context for prompting."""
        with self._lock:
            if not self._history:
                return ""

            lines = []
            # Get last N exchanges (2 messages per exchange)
            recent = self._history[-(last_n_exchanges * 2) :]
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content']}")
            return "\n".join(lines)

    def get_last_answer(self) -> str:
        """Get the last assistant answer for follow-up questions."""
        with self._lock:
            for msg in reversed(self._history):
                if msg["role"] == "assistant":
                    return msg["content"]
            return ""

    def clear(self) -> None:
        """Clear all conversation history."""
        with self._lock:
            self._history.clear()

    @property
    def history(self) -> List[Dict[str, str]]:
        """Get a copy of the full history."""
        with self._lock:
            return self._history.copy()


# Global conversation manager instance
conversation_manager = ConversationManager()
