"""
analytics/aggregator.py — Computes dashboard metrics from the SQLite DB.

Returns structured data consumed by the frontend analytics dashboard.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, text

from database.db import db_session
from database.models import Feedback, IntentLog, Message, SentimentLog, Session

logger = logging.getLogger(__name__)


class AnalyticsAggregator:
    """
    Computes all metrics needed for the analytics dashboard.

    Usage:
        agg = AnalyticsAggregator()
        data = agg.get_dashboard_data()
    """

    def get_dashboard_data(self) -> dict[str, Any]:
        """Return a single dict with all dashboard metrics."""
        try:
            with db_session() as db:
                return {
                    "overview": self._overview(db),
                    "messages_over_time": self._messages_over_time(db),
                    "intent_distribution": self._intent_distribution(db),
                    "sentiment_distribution": self._sentiment_distribution(db),
                    "emotion_distribution": self._emotion_distribution(db),
                    "response_sources": self._response_sources(db),
                    "top_entities": self._top_entities(db),
                    "satisfaction": self._satisfaction(db),
                    "avg_response_time": self._avg_response_time(db),
                    "hourly_activity": self._hourly_activity(db),
                    "generated_at": datetime.utcnow().isoformat(),
                }
        except Exception as exc:
            logger.error(f"AnalyticsAggregator error: {exc}", exc_info=True)
            return {"error": str(exc)}

    # ── Metric helpers ────────────────────────────────────────

    def _overview(self, db) -> dict:
        total_messages = db.query(func.count(Message.id)).scalar() or 0
        total_sessions = db.query(func.count(Session.id)).scalar() or 0
        user_messages = db.query(func.count(Message.id)).filter(Message.role == "user").scalar() or 0
        bot_messages = db.query(func.count(Message.id)).filter(Message.role == "bot").scalar() or 0

        # Active sessions in last 24h
        cutoff = datetime.utcnow() - timedelta(hours=24)
        active_today = (
            db.query(func.count(Session.id))
            .filter(Session.last_active >= cutoff)
            .scalar() or 0
        )

        return {
            "total_messages": total_messages,
            "total_sessions": total_sessions,
            "user_messages": user_messages,
            "bot_messages": bot_messages,
            "active_sessions_today": active_today,
        }

    def _messages_over_time(self, db, days: int = 7) -> list[dict]:
        """Messages grouped by day for the last N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        rows = (
            db.query(
                func.date(Message.timestamp).label("day"),
                func.count(Message.id).label("count"),
            )
            .filter(Message.timestamp >= cutoff)
            .group_by(func.date(Message.timestamp))
            .order_by(func.date(Message.timestamp))
            .all()
        )
        return [{"date": str(r.day), "count": r.count} for r in rows]

    def _intent_distribution(self, db, limit: int = 10) -> list[dict]:
        rows = (
            db.query(IntentLog.intent, func.count(IntentLog.id).label("count"))
            .group_by(IntentLog.intent)
            .order_by(func.count(IntentLog.id).desc())
            .limit(limit)
            .all()
        )
        return [{"intent": r.intent, "count": r.count} for r in rows]

    def _sentiment_distribution(self, db) -> dict[str, int]:
        rows = (
            db.query(SentimentLog.label, func.count(SentimentLog.id).label("count"))
            .group_by(SentimentLog.label)
            .all()
        )
        return {r.label: r.count for r in rows}

    def _emotion_distribution(self, db) -> list[dict]:
        rows = (
            db.query(SentimentLog.emotion, func.count(SentimentLog.id).label("count"))
            .group_by(SentimentLog.emotion)
            .order_by(func.count(SentimentLog.id).desc())
            .all()
        )
        return [{"emotion": r.emotion, "count": r.count} for r in rows]

    def _response_sources(self, db) -> list[dict]:
        rows = (
            db.query(Message.generated_by, func.count(Message.id).label("count"))
            .filter(Message.role == "bot", Message.generated_by.isnot(None))
            .group_by(Message.generated_by)
            .all()
        )
        return [{"source": r.generated_by, "count": r.count} for r in rows]

    def _top_entities(self, db, limit: int = 10) -> list[dict]:
        """Parse entity JSON blobs and aggregate by entity text."""
        rows = db.query(IntentLog.entities).filter(IntentLog.entities.isnot(None)).all()
        counter: Counter = Counter()
        for row in rows:
            try:
                entities = json.loads(row.entities)
                for ent in entities:
                    key = f"{ent['text']} ({ent['label']})"
                    counter[key] += 1
            except Exception:
                continue
        return [{"entity": k, "count": v} for k, v in counter.most_common(limit)]

    def _satisfaction(self, db) -> dict:
        total = db.query(func.count(Feedback.id)).scalar() or 0
        positive = db.query(func.count(Feedback.id)).filter(Feedback.rating.is_(True)).scalar() or 0
        negative = total - positive
        rate = round(positive / total * 100, 1) if total > 0 else None
        return {
            "total_ratings": total,
            "positive": positive,
            "negative": negative,
            "satisfaction_rate": rate,
        }

    def _avg_response_time(self, db) -> float | None:
        result = (
            db.query(func.avg(Message.response_time_ms))
            .filter(Message.role == "bot", Message.response_time_ms.isnot(None))
            .scalar()
        )
        return round(float(result), 2) if result else None

    def _hourly_activity(self, db, days: int = 7) -> list[dict]:
        cutoff = datetime.utcnow() - timedelta(days=days)
        rows = (
            db.query(
                func.strftime("%H", Message.timestamp).label("hour"),
                func.count(Message.id).label("count"),
            )
            .filter(Message.timestamp >= cutoff, Message.role == "user")
            .group_by(func.strftime("%H", Message.timestamp))
            .order_by(func.strftime("%H", Message.timestamp))
            .all()
        )
        return [{"hour": int(r.hour), "count": r.count} for r in rows]


# ── Module-level singleton ─────────────────────────────────────────────────────
_aggregator = None


def get_aggregator() -> AnalyticsAggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = AnalyticsAggregator()
    return _aggregator
