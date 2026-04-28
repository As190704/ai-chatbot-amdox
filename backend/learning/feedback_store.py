"""
learning/feedback_store.py — Stores user thumbs-up/down feedback.

The stored ratings are used by the RL trainer to weight rule-based
response selection and to compute the satisfaction metric.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from database.db import db_session
from database.models import Feedback, Message

logger = logging.getLogger(__name__)


class FeedbackStore:
    """
    Persists per-message user ratings.

    Usage:
        store = FeedbackStore()
        store.record(message_id=42, rating=True, comment="Very helpful!")
    """

    def record(
        self,
        message_id: int,
        rating: bool,
        comment: Optional[str] = None,
    ) -> bool:
        """
        Save a feedback rating for a bot message.

        Args:
            message_id: The bot Message.id being rated.
            rating:     True = 👍, False = 👎
            comment:    Optional free-text comment.

        Returns:
            True on success, False on error (e.g. message not found).
        """
        try:
            with db_session() as db:
                # Verify the message exists and is a bot message
                msg = db.get(Message, message_id)
                if msg is None:
                    logger.warning(f"FeedbackStore: message_id={message_id} not found.")
                    return False
                if msg.role != "bot":
                    logger.warning(f"FeedbackStore: message_id={message_id} is not a bot message.")
                    return False

                # Upsert: replace existing feedback if any
                existing = db.query(Feedback).filter(Feedback.message_id == message_id).first()
                if existing:
                    existing.rating = rating
                    existing.comment = comment
                    existing.created_at = datetime.utcnow()
                    logger.debug(f"FeedbackStore: Updated feedback for message {message_id}.")
                else:
                    fb = Feedback(
                        message_id=message_id,
                        rating=rating,
                        comment=comment,
                        created_at=datetime.utcnow(),
                    )
                    db.add(fb)
                    logger.debug(f"FeedbackStore: Recorded {'👍' if rating else '👎'} for message {message_id}.")

                return True

        except Exception as exc:
            logger.error(f"FeedbackStore: Failed to record feedback — {exc}", exc_info=True)
            return False

    def get_for_message(self, message_id: int) -> Optional[dict]:
        """Retrieve feedback for a single message, or None if not rated."""
        try:
            with db_session() as db:
                fb = db.query(Feedback).filter(Feedback.message_id == message_id).first()
                if fb is None:
                    return None
                return {
                    "message_id": message_id,
                    "rating": fb.rating,
                    "comment": fb.comment,
                    "created_at": fb.created_at.isoformat(),
                }
        except Exception as exc:
            logger.error(f"FeedbackStore: get_for_message error — {exc}")
            return None


# ── Module-level singleton ─────────────────────────────────────────────────────
_store: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    global _store
    if _store is None:
        _store = FeedbackStore()
    return _store
