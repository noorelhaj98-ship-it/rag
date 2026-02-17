"""LLM service for generating answers."""

import requests

from app.config import settings


class LLMService:
    """DeepSeek LLM client for answer generation."""

    def __init__(self):
        self.api_key = settings.deepseek_api_key
        self.base_url = settings.deepseek_base_url
        self.model = settings.deepseek_model
        self.system_prompt = settings.load_system_prompt()

    def generate_answer(
        self,
        question: str,
        context: str,
        history: str = "",
        previous_answer: str = "",
    ) -> str:
        """Generate answer using DeepSeek LLM.

        Args:
            question: User question
            context: Retrieved document context
            history: Previous conversation history
            previous_answer: Last assistant answer for follow-up

        Returns:
            Generated answer text

        Raises:
            RuntimeError: If API key is missing or request fails
        """
        if not self.api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY")

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build conversation-aware prompt
        user_content_parts = []

        if history:
            user_content_parts.append(f"Previous conversation:\n{history}")

        if previous_answer:
            user_content_parts.append(f"My previous answer:\n{previous_answer}")

        user_content_parts.append(f"Document Context:\n{context}")
        user_content_parts.append(f"Current question: {question}\n\nAnswer:")

        user_content = "\n\n".join(user_content_parts)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "max_tokens": 200,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


# Global LLM service instance
llm_service = LLMService()
