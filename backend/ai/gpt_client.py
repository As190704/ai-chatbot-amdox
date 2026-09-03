"""
ai/gpt_client.py — OpenAI GPT integration.

Falls back gracefully (returns None) when:
  - No API key is configured
  - The API call fails for any reason

The caller (response_engine) will then try the local LLM.
"""

from __future__ import annotations

import logging
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are AMDOX AI, an intelligent, friendly, and helpful AI assistant.

Key traits:
- You are conversational, warm, and professional
- You give concise but thorough answers
- You use markdown formatting (bold, bullet points) when it aids clarity
- You acknowledge when you're uncertain rather than guessing
- You adapt your tone: enthusiastic for happy users, empathetic for frustrated ones
- You remember the conversation context and refer back to it naturally
- You use occasional emojis to make responses feel friendly (not excessive)

When you don't know something, say so honestly and suggest alternatives.
Never claim to have real-time data (weather, news, stock prices) unless told you do.
"""


class GPTClient:
    """
    Thin wrapper around the OpenAI chat-completions API.

    Usage:
        client = GPTClient()
        reply = await client.generate(messages=[...])
    """

    def __init__(self) -> None:
        self._client = None
        self._available = False
        self._init()

    def _init(self) -> None:
        if not settings.has_openai:
            logger.info("GPTClient: No OpenAI API key — GPT responses disabled.")
            return
        try:
            from openai import AsyncOpenAI  # noqa: PLC0415
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
            self._available = True
            logger.info(f"GPTClient: Initialised with model '{settings.openai_model}'.")
        except ImportError:
            logger.warning("GPTClient: openai package not installed. Run: pip install openai")

    @property
    def is_available(self) -> bool:
        return self._available

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Optional[str]:
        """
        Call the OpenAI chat completion API and return the assistant's message.
        Returns None on any error so callers can fall back.
        """
        if not self._available or self._client is None:
            return None

        try:
            response = await self._client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(f"GPTClient: API call failed — {exc}")
            return None

    async def generate_with_context(
        self,
        user_message: str,
        history: list[dict],
        sentiment_hint: str = "clear and concise",
    ) -> Optional[str]:
        """
        Convenience method that builds a messages list from history and calls generate().

        Args:
            user_message:  The latest user input.
            history:       Previous turns as [{"role":..., "content":...}, ...].
            sentiment_hint: Tone adjustment string injected into the system prompt.
        """
        system = SYSTEM_PROMPT + f"\n\nResponse tone: Be {sentiment_hint}."
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return await self.generate(messages)

_gpt_client: Optional[GPTClient] = None


def get_gpt_client() -> GPTClient:
    global _gpt_client
    if _gpt_client is None:
        _gpt_client = GPTClient()
    return _gpt_client
