"""Application configuration using Pydantic Settings."""

import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Embedding Service
    embedding_service_type: str = Field(default="aragemma", alias="EMBEDDING_SERVICE_TYPE")
    embedding_api_url: Optional[str] = Field(default=None, alias="EMBEDDING_API_URL")
    embed_timeout: int = Field(default=120, alias="EMBED_TIMEOUT")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="realsoft_chunks_hybrid", alias="QDRANT_COLLECTION")

    # DeepSeek LLM
    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    # Retrieval Config
    top_k: int = Field(default=6, alias="TOP_K")
    top_k_dense: int = Field(default=100, alias="TOP_K_DENSE")
    top_k_keyword: int = Field(default=100, alias="TOP_K_KEYWORD")
    top_k_final: int = Field(default=6, alias="TOP_K_FINAL")
    fusion_type: str = Field(default="RRF", alias="FUSION_TYPE")
    rrf_k: int = Field(default=60, alias="RRF_K")

    # Context Limits
    max_context_chars_per_chunk: int = Field(default=450, alias="MAX_CONTEXT_CHARS_PER_CHUNK")
    max_context_total_chars: int = Field(default=2400, alias="MAX_CONTEXT_TOTAL_CHARS")

    # Reranker
    reranker_enabled: bool = Field(default=True, alias="RERANKER_ENABLED")
    reranker_top_k: int = Field(default=30, alias="RERANKER_TOP_K")
    reranker_model: str = Field(default="cross-encoder", alias="RERANKER_MODEL")

    # Rate Limiting
    rate_limit: str = Field(default="10/minute", alias="RATE_LIMIT")

    # Logging
    logfire_token: Optional[str] = Field(default=None, alias="LOGFIRE_TOKEN")

    # Paths
    system_prompt_path: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "system_prompt.yaml",
        alias="SYSTEM_PROMPT_PATH",
    )
    log_file: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "rag_logs.jsonl",
        alias="LOG_FILE",
    )

    # Conversation
    max_history: int = Field(default=10, alias="MAX_HISTORY")

    @property
    def default_system_prompt(self) -> str:
        """Default system prompt when file is not available."""
        return (
            "Answer ONLY using the provided context. "
            "Be concise and specific: 1-2 sentences maximum. "
            "If not in context, say: 'Not found in the provided documents.'"
        )

    def load_system_prompt(self) -> str:
        """Load system prompt from YAML file."""
        import yaml

        try:
            with open(self.system_prompt_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and "content" in data:
                    return data["content"].strip()
                if isinstance(data, str):
                    return data.strip()
        except Exception:
            pass
        return self.default_system_prompt


class Role(str):
    """Role enumeration for RBAC."""

    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"


class RBACConfig:
    """RBAC configuration and permission management."""

    ROLE_PERMISSIONS: Dict[str, List[str]] = {
        Role.ADMIN: ["ask", "history", "clear", "admin"],
        Role.USER: ["ask", "history", "clear"],
        Role.READONLY: ["history"],
    }

    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.enabled = bool(self.api_keys)

    def _load_api_keys(self) -> Dict[str, Dict[str, str]]:
        """Load API keys from environment variable.

        Format: API_KEYS=user:key1:role1,user:key2:role2
        Example: API_KEYS=user:abc123:user,admin:xyz789:admin
        """
        keys_config = os.environ.get("API_KEYS", "")
        api_keys = {}
        if keys_config:
            for entry in keys_config.split(","):
                parts = entry.strip().split(":")
                if len(parts) == 3:
                    name, key, role = parts
                    api_keys[key] = {"name": name, "role": role}
        return api_keys

    def get_user(self, token: str) -> Optional[Dict[str, str]]:
        """Get user info by API token."""
        return self.api_keys.get(token)

    def has_permission(self, role: str, permission: str) -> bool:
        """Check if role has specific permission."""
        allowed = self.ROLE_PERMISSIONS.get(role, [])
        return permission in allowed


# Global settings instance
settings = Settings()
rbac_config = RBACConfig()
