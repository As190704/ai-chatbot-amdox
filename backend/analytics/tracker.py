"""
analytics/tracker.py — Logs every chatbot interaction to SQLite.

Called by the FastAPI route handlers after each successful exchange.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from database.db import db_session
from database.models import IntentLog, Message, SentimentLog, Session

logger = logging.getLogger(__name__)


class InteractionTracker:
    """
    Persists chat turns (user + bot messages) with their NLP metadata.

    Usage:
        tracker = InteractionTracker()
        tracker.log_exchange(
            session_id="abc",
            user_message="hello",
            bot_reply="Hi there!",
            intent="greet",
            intent_confidence=0.92,
            entities=[],
            sentiment_label="positive",
            polarity=0.5,
            subjectivity=0.6,
            generated_by="rule",
            response_time_ms=12.4,
        )
    """

    def ensure_session(self, db, session_id: str, user_agent: Optional[str] = None) -> Session:
        """Get or create the Session row for *session_id*."""
        session = db.get(Session, session_id)
        if session is None:
            session = Session(
                id=session_id,
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow(),
                user_agent=user_agent,
                message_count=0,
            )
            db.add(session)
            db.flush()
        else:
            session.last_active = datetime.utcnow()
        return session

    def log_exchange(
        self,
        session_id: str,
        user_message: str,
        bot_reply: str,
        intent: str,
        intent_confidence: float,
        entities: list[dict],
        sentiment_label: str,
        polarity: float,
        subjectivity: float,
        sentiment_emotion: str,
        generated_by: str,
        response_time_ms: float,
        user_agent: Optional[str] = None,
    ) -> Optional[int]:
        """
        Write a full exchange (user turn + bot turn + logs) to the DB.
        Returns the bot Message.id on success, None on error.
        """
        try:
            with db_session() as db:
                # Ensure session exists
                session = self.ensure_session(db, session_id, user_agent)
                session.message_count += 2  # user + bot

                # ── User message ───────────────────────────────
                user_msg = Message(
                    session_id=session_id,
                    role="user",
                    content=user_message,
                    timestamp=datetime.utcnow(),
                )
                db.add(user_msg)
                db.flush()

                # Intent log (on user message)
                intent_log = IntentLog(
                    message_id=user_msg.id,
                    intent=intent,
                    confidence=intent_confidence,
                    entities=json.dumps(entities) if entities else None,
                )
                db.add(intent_log)

                # Sentiment log (on user message)
                sentiment_log = SentimentLog(
                    message_id=user_msg.id,
                    polarity=polarity,
                    subjectivity=subjectivity,
                    label=sentiment_label,
                    emotion=sentiment_emotion,
                )
                db.add(sentiment_log)

                # ── Bot message ────────────────────────────────
                bot_msg = Message(
                    session_id=session_id,
                    role="bot",
                    content=bot_reply,
                    timestamp=datetime.utcnow(),
                    response_time_ms=response_time_ms,
                    generated_by=generated_by,
                )
                db.add(bot_msg)
                db.flush()

                return bot_msg.id

        except Exception as exc:
            logger.error(f"InteractionTracker: Failed to log exchange — {exc}", exc_info=True)
            return None


# ── Module-level singleton ─────────────────────────────────────────────────────
_tracker: Optional[InteractionTracker] = None


def get_tracker() -> InteractionTracker:
    global _tracker
    if _tracker is None:
        _tracker = InteractionTracker()
    return _tracker
