"""
ai/response_engine.py — Central orchestration layer.

Fallback chain:
  1. Rule-based engine  (instant, deterministic)
  2. OpenAI GPT         (if API key configured)
  3. Local HuggingFace  (lazy-loaded on first use)
  4. Static fallback    (always succeeds)

Each step records which layer generated the response via `generated_by`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ai.gpt_client import get_gpt_client
from ai.local_llm import get_local_llm
from ai.rule_based import get_rule_engine
from nlp.context_manager import ConversationContext, get_context_manager
from nlp.intent_classifier import IntentResult, get_classifier
from nlp.ner_extractor import get_ner_extractor
from nlp.sentiment_analyzer import SentimentResult, get_sentiment_analyzer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChatResponse:
    text: str
    intent: IntentResult
    sentiment: SentimentResult
    entities: list[dict]
    generated_by: str                  # rule | gpt | local_llm | fallback
    response_time_ms: float
    session_id: str
    context_turns: int = 0
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_RESPONSES = [
    "I'm sorry, I didn't quite understand that. Could you try rephrasing? 🤔",
    "Hmm, that's a tricky one! Could you give me a bit more context?",
    "I want to help but I'm not sure what you mean. Could you elaborate?",
    "Let me think… I'm still learning! Could you ask that differently?",
]

import random


class ResponseEngine:
    """
    Orchestrates the full NLP → response pipeline.

    Usage:
        engine = ResponseEngine()
        response = await engine.process("hello!", session_id="abc123")
    """

    def __init__(self) -> None:
        self._classifier = get_classifier()
        self._ner = get_ner_extractor()
        self._sentiment = get_sentiment_analyzer()
        self._ctx_mgr = get_context_manager()
        self._rule_engine = get_rule_engine()
        self._gpt = get_gpt_client()
        self._local_llm = get_local_llm()

    async def process(self, user_message: str, session_id: str) -> ChatResponse:
        start = time.perf_counter()

        # ── Step 1: NLP pipeline ──────────────────────────────
        intent_result = self._classifier.classify(user_message)
        sentiment_result = self._sentiment.analyze(user_message)
        entities = self._ner.extract_as_dicts(user_message)

        # ── Step 2: Update context ────────────────────────────
        ctx: ConversationContext = self._ctx_mgr.get_or_create(session_id)
        ctx.add_user_turn(
            content=user_message,
            intent=intent_result.intent,
            sentiment=sentiment_result.label,
        )

        # ── Step 3: Generate response ─────────────────────────
        reply_text, generated_by = await self._generate_reply(
            user_message=user_message,
            intent=intent_result.intent,
            sentiment=sentiment_result,
            ctx=ctx,
        )

        # ── Step 4: Update context with bot reply ─────────────
        ctx.add_bot_turn(reply_text)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return ChatResponse(
            text=reply_text,
            intent=intent_result,
            sentiment=sentiment_result,
            entities=entities,
            generated_by=generated_by,
            response_time_ms=round(elapsed_ms, 2),
            session_id=session_id,
            context_turns=len(ctx.turns),
            metadata={
                "is_follow_up": ctx.is_follow_up(),
                "topic_switches": ctx.topic_switches,
            },
        )

    async def _generate_reply(
        self,
        user_message: str,
        intent: str,
        sentiment: SentimentResult,
        ctx: ConversationContext,
    ) -> tuple[str, str]:
        """
        Try each fallback layer in order.
        Returns (reply_text, source_label).
        """

        # ── Layer 1: Rule-based ───────────────────────────────
        rule_reply = self._rule_engine.get_response(user_message, intent)
        if rule_reply:
            logger.debug(f"Rule-based responded for intent='{intent}'")
            return rule_reply, "rule"

        # ── Layer 2: GPT ──────────────────────────────────────
        if self._gpt.is_available:
            history = [t.to_openai_message() for t in ctx.get_history(max_turns=8)]
            gpt_reply = await self._gpt.generate_with_context(
                user_message=user_message,
                history=history,
                sentiment_hint=sentiment.tone_hint,
            )
            if gpt_reply:
                logger.debug("GPT responded.")
                return gpt_reply, "gpt"

        # ── Layer 3: Local LLM ────────────────────────────────
        history_dicts = [t.to_openai_message() for t in ctx.get_history(max_turns=4)]
        local_reply = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._local_llm.generate(user_message, history_dicts),
        )
        if local_reply:
            logger.debug("LocalLLM responded.")
            return local_reply, "local_llm"

        # ── Layer 4: Static fallback ──────────────────────────
        logger.debug("Using static fallback.")
        return random.choice(_FALLBACK_RESPONSES), "fallback"


# ── Module-level singleton ─────────────────────────────────────────────────────
_response_engine: Optional[ResponseEngine] = None


def get_response_engine() -> ResponseEngine:
    global _response_engine
    if _response_engine is None:
        _response_engine = ResponseEngine()
    return _response_engine
