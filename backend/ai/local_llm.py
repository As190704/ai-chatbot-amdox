"""
ai/local_llm.py — HuggingFace local model fallback.

Default model: facebook/blenderbot-400M-distill
  - ~400 MB download on first use
  - Runs on CPU, no GPU required
  - Conversational model fine-tuned for chat

Configured via LOCAL_MODEL_NAME in .env.
Set PRELOAD_LOCAL_MODEL=true to load eagerly at startup.

If the model is unavailable (not downloaded / import error),
generate() returns None so the caller can use the last-resort fallback.
"""

from __future__ import annotations

import logging
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


class LocalLLM:
    """
    Wraps a HuggingFace conversational (or text-generation) model.

    Usage:
        llm = LocalLLM()
        reply = llm.generate("What is the speed of light?")
    """

    def __init__(self, eager_load: bool = False) -> None:
        self._pipeline = None
        self._model_name = settings.local_model_name
        self._available = False

        if eager_load:
            self._load()

    # ── Private ───────────────────────────────────────────────

    def _load(self) -> None:
        try:
            from transformers import pipeline  # noqa: PLC0415
            logger.info(f"LocalLLM: Loading '{self._model_name}' …")

            # BlenderBot → conversational pipeline
            if "blenderbot" in self._model_name.lower():
                self._pipeline = pipeline(
                    "conversational",
                    model=self._model_name,
                    tokenizer=self._model_name,
                )
            else:
                # Generic text-generation fallback
                self._pipeline = pipeline(
                    "text-generation",
                    model=self._model_name,
                    tokenizer=self._model_name,
                    max_new_tokens=200,
                    pad_token_id=50256,
                )

            self._available = True
            logger.info(f"LocalLLM: '{self._model_name}' loaded successfully.")
        except Exception as exc:
            logger.warning(f"LocalLLM: Could not load model — {exc}")
            self._pipeline = None
            self._available = False

    # ── Public ────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return self._available

    def generate(self, user_input: str, history: Optional[list[dict]] = None) -> Optional[str]:
        """
        Generate a response for *user_input*.

        Lazy-loads the model on first call.
        Returns None if unavailable.
        """
        if self._pipeline is None:
            self._load()

        if not self._available or self._pipeline is None:
            return None

        try:
            from transformers import Conversation  # noqa: PLC0415

            if "blenderbot" in self._model_name.lower():
                # Build conversation object
                convo = Conversation(user_input)
                result = self._pipeline(convo)
                reply = result.generated_responses[-1] if result.generated_responses else None
                return reply.strip() if reply else None
            else:
                # Text-generation pipeline
                prompt = user_input
                if history:
                    ctx = "\n".join(f"{t['role'].title()}: {t['content']}" for t in history[-4:])
                    prompt = f"{ctx}\nUser: {user_input}\nAssistant:"

                outputs = self._pipeline(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
                generated = outputs[0]["generated_text"]
                # Strip the prompt from the generated output
                if "Assistant:" in generated:
                    generated = generated.split("Assistant:")[-1]
                return generated.strip()

        except Exception as exc:
            logger.warning(f"LocalLLM: Generation failed — {exc}")
            return None


# ── Module-level singleton ─────────────────────────────────────────────────────
_local_llm: Optional[LocalLLM] = None


def get_local_llm() -> LocalLLM:
    global _local_llm
    if _local_llm is None:
        _local_llm = LocalLLM(eager_load=settings.preload_local_model)
    return _local_llm
