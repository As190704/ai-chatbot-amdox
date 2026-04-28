"""
nlp/context_manager.py — In-memory conversation context store.

Maintains a sliding window of recent turns per session so the response
engine can:
  - Reference previous messages (e.g., "tell me more about it")
  - Detect topic continuity vs. topic switches
  - Pass structured history to GPT as a messages[] array

Thread-safety: each session is stored in a plain dict; FastAPI's async
event loop is single-threaded so no explicit locking is required for
normal use. If you run with multiple workers, switch to Redis.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """One user↔bot exchange."""
    role: str          # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    intent: Optional[str] = None
    sentiment_label: Optional[str] = None

    def to_openai_message(self) -> dict:
        """Format for OpenAI messages[] payload."""
        return {"role": self.role, "content": self.content}


@dataclass
class ConversationContext:
    """Sliding-window history for one session."""
    session_id: str
    turns: deque[Turn] = field(default_factory=lambda: deque(maxlen=settings.context_window * 2))
    last_intent: Optional[str] = None
    last_sentiment: Optional[str] = None
    topic_switches: int = 0

    # ── Accessors ─────────────────────────────────────────────

    def add_user_turn(self, content: str, intent: Optional[str] = None, sentiment: Optional[str] = None) -> None:
        turn = Turn(role="user", content=content, intent=intent, sentiment_label=sentiment)
        if intent and self.last_intent and intent != self.last_intent:
            self.topic_switches += 1
        self.last_intent = intent or self.last_intent
        self.last_sentiment = sentiment or self.last_sentiment
        self.turns.append(turn)

    def add_bot_turn(self, content: str) -> None:
        self.turns.append(Turn(role="assistant", content=content))

    def get_history(self, max_turns: Optional[int] = None) -> list[Turn]:
        turns = list(self.turns)
        if max_turns:
            turns = turns[-max_turns:]
        return turns

    def to_openai_messages(self, system_prompt: str, max_turns: int = 10) -> list[dict]:
        """
        Build an OpenAI-compatible messages list:
          [{"role": "system", "content": ...}, {"role": "user", ...}, ...]
        """
        msgs = [{"role": "system", "content": system_prompt}]
        msgs.extend(t.to_openai_message() for t in self.get_history(max_turns))
        return msgs

    @property
    def user_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "user"]

    @property
    def bot_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "assistant"]

    @property
    def last_user_message(self) -> Optional[str]:
        msgs = self.user_turns
        return msgs[-1].content if msgs else None

    @property
    def previous_user_message(self) -> Optional[str]:
        msgs = self.user_turns
        return msgs[-2].content if len(msgs) >= 2 else None

    def is_follow_up(self) -> bool:
        """Heuristic: message is a follow-up if it's very short or contains anaphora."""
        last = self.last_user_message or ""
        follow_up_tokens = {"it", "that", "this", "more", "again", "why", "how", "when", "who", "where"}
        words = set(last.lower().split())
        return len(last.split()) <= 3 or bool(words & follow_up_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Session store
# ─────────────────────────────────────────────────────────────────────────────

_SESSION_TTL_SECONDS = 3600  # 1 hour of inactivity → purge


class ContextManager:
    """
    Central store for all active ConversationContext objects.

    Usage:
        ctx_mgr = get_context_manager()
        ctx = ctx_mgr.get_or_create(session_id)
        ctx.add_user_turn("hello")
    """

    def __init__(self) -> None:
        self._store: dict[str, ConversationContext] = {}
        self._last_seen: dict[str, float] = {}

    def get_or_create(self, session_id: str) -> ConversationContext:
        self._last_seen[session_id] = time.time()
        if session_id not in self._store:
            self._store[session_id] = ConversationContext(session_id=session_id)
        return self._store[session_id]

    def exists(self, session_id: str) -> bool:
        return session_id in self._store

    def reset(self, session_id: str) -> None:
        """Clear history for a session (new chat button)."""
        self._store.pop(session_id, None)
        self._last_seen.pop(session_id, None)

    def purge_stale(self) -> int:
        """Remove sessions inactive for > TTL. Returns count removed."""
        now = time.time()
        stale = [sid for sid, ts in self._last_seen.items() if now - ts > _SESSION_TTL_SECONDS]
        for sid in stale:
            self._store.pop(sid, None)
            self._last_seen.pop(sid, None)
        return len(stale)

    @property
    def active_session_count(self) -> int:
        return len(self._store)


# ── Module-level singleton ─────────────────────────────────────────────────────
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager
