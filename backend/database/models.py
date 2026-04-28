"""
database/models.py — SQLAlchemy ORM table definitions.

Tables:
  sessions      — one row per chat session (browser tab / user)
  messages      — every message (user + bot) in every session
  intent_logs   — classified intent for each user message
  sentiment_logs— sentiment scores per message
  feedback      — thumbs up/down ratings per bot response
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Sessions
# ─────────────────────────────────────────────────────────────────────────────

class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(64), primary_key=True)            # UUID string
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_agent = Column(String(512), nullable=True)       # browser info if available
    message_count = Column(Integer, default=0)

    # Relationships
    messages: list[Message] = relationship("Message", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Session id={self.id} msgs={self.message_count}>"


# ─────────────────────────────────────────────────────────────────────────────
# Messages
# ─────────────────────────────────────────────────────────────────────────────

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.id"), nullable=False)
    role = Column(String(8), nullable=False)              # "user" | "bot"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    response_time_ms = Column(Float, nullable=True)       # latency for bot messages

    # Which AI layer generated this response
    generated_by = Column(String(32), nullable=True)      # rule | gpt | local_llm | fallback

    # Relationships
    session: Session = relationship("Session", back_populates="messages")
    intent_log: IntentLog = relationship("IntentLog", back_populates="message", uselist=False, cascade="all, delete-orphan")
    sentiment_log: SentimentLog = relationship("SentimentLog", back_populates="message", uselist=False, cascade="all, delete-orphan")
    feedback: Feedback = relationship("Feedback", back_populates="message", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        preview = self.content[:40].replace("\n", " ")
        return f"<Message id={self.id} role={self.role} '{preview}...'>"


# ─────────────────────────────────────────────────────────────────────────────
# Intent Logs
# ─────────────────────────────────────────────────────────────────────────────

class IntentLog(Base):
    __tablename__ = "intent_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False, unique=True)
    intent = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    entities = Column(Text, nullable=True)               # JSON-serialised entity list

    message: Message = relationship("Message", back_populates="intent_log")

    def __repr__(self) -> str:
        return f"<IntentLog intent={self.intent} conf={self.confidence:.2f}>"


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Logs
# ─────────────────────────────────────────────────────────────────────────────

class SentimentLog(Base):
    __tablename__ = "sentiment_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False, unique=True)
    polarity = Column(Float, nullable=False)             # -1.0 → +1.0
    subjectivity = Column(Float, nullable=False)         # 0.0 → 1.0
    label = Column(String(16), nullable=False)           # positive | neutral | negative
    emotion = Column(String(32), nullable=True)          # joy | anger | sadness | etc.

    message: Message = relationship("Message", back_populates="sentiment_log")

    def __repr__(self) -> str:
        return f"<SentimentLog label={self.label} polarity={self.polarity:.2f}>"


# ─────────────────────────────────────────────────────────────────────────────
# User Feedback
# ─────────────────────────────────────────────────────────────────────────────

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False, unique=True)
    rating = Column(Boolean, nullable=False)             # True = 👍  False = 👎
    comment = Column(Text, nullable=True)                # optional free-text
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    message: Message = relationship("Message", back_populates="feedback")

    def __repr__(self) -> str:
        thumb = "👍" if self.rating else "👎"
        return f"<Feedback msg={self.message_id} {thumb}>"
