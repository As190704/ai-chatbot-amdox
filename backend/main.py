"""
main.py — FastAPI application entry point.

Routes:
  GET  /                        → Health check
  GET  /api/health              → Detailed health status
  POST /api/chat                → REST chat endpoint
  WS   /ws/chat/{session_id}    → WebSocket real-time chat
  POST /api/feedback            → Submit thumbs up/down
  GET  /api/analytics           → Dashboard data
  GET  /api/analytics/sessions  → Active session count
  DELETE /api/session/{id}      → Reset conversation context
  GET  /api/intents             → List all known intents

Run with:
    python main.py
    # or
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ai.response_engine import get_response_engine
from analytics.aggregator import get_aggregator
from analytics.tracker import get_tracker
from config import settings
from database.db import init_db
from learning.feedback_store import get_feedback_store
from learning.rl_trainer import get_rl_trainer
from nlp.context_manager import get_context_manager
from nlp.intent_classifier import INTENT_EXAMPLES, get_classifier

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 AMDOX AI Chatbot starting …")

    # Create DB tables
    init_db()
    logger.info("✅ Database tables ready.")

    # Warm-up intent classifier (builds TF-IDF matrix)
    get_classifier()
    logger.info("✅ Intent classifier ready.")

    # Schedule stale session purge every 30 minutes
    async def _purge_loop():
        while True:
            await asyncio.sleep(1800)
            removed = get_context_manager().purge_stale()
            if removed:
                logger.info(f"🗑  Purged {removed} stale session(s).")

    asyncio.create_task(_purge_loop())

    logger.info(f"✅ AMDOX AI ready at http://{settings.app_host}:{settings.app_port}")
    yield

    logger.info("👋 AMDOX AI shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AMDOX AI Chatbot",
    description="Dynamic AI Chatbot API — NLP, Sentiment Analysis, Multi-layer AI responses",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the frontend (file:// or localhost) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User's message")
    session_id: Optional[str] = Field(None, description="Session ID (generated if omitted)")

    model_config = {"json_schema_extra": {"example": {"message": "Hello! How are you?"}}}


class ChatResponseSchema(BaseModel):
    reply: str
    session_id: str
    intent: str
    intent_confidence: float
    sentiment: dict
    entities: list[dict]
    generated_by: str
    response_time_ms: float
    message_id: Optional[int]
    timestamp: str


class FeedbackRequest(BaseModel):
    message_id: int
    rating: bool                   # True = 👍, False = 👎
    comment: Optional[str] = None
    intent: Optional[str] = None   # to update RL weights


class FeedbackResponse(BaseModel):
    success: bool
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "name": "AMDOX AI Chatbot",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "docs": "/docs",
    }


@app.get("/api/health", tags=["Health"])
async def health():
    engine = get_response_engine()
    return {
        "status": "healthy",
        "openai_available": engine._gpt.is_available,
        "local_llm_available": engine._local_llm.is_available,
        "active_sessions": get_context_manager().active_session_count,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── REST Chat ─────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponseSchema, tags=["Chat"])
async def chat(req: ChatRequest, request: Request):
    """
    Send a message and receive an AI-generated response.
    Creates a new session automatically if session_id is omitted.
    """
    session_id = req.session_id or str(uuid.uuid4())
    user_agent = request.headers.get("user-agent")

    engine = get_response_engine()
    chat_resp = await engine.process(req.message, session_id)

    # Persist to DB (fire-and-forget style — errors don't break the response)
    tracker = get_tracker()
    message_id = tracker.log_exchange(
        session_id=session_id,
        user_message=req.message,
        bot_reply=chat_resp.text,
        intent=chat_resp.intent.intent,
        intent_confidence=chat_resp.intent.confidence,
        entities=chat_resp.entities,
        sentiment_label=chat_resp.sentiment.label,
        polarity=chat_resp.sentiment.polarity,
        subjectivity=chat_resp.sentiment.subjectivity,
        sentiment_emotion=chat_resp.sentiment.emotion,
        generated_by=chat_resp.generated_by,
        response_time_ms=chat_resp.response_time_ms,
        user_agent=user_agent,
    )

    return ChatResponseSchema(
        reply=chat_resp.text,
        session_id=session_id,
        intent=chat_resp.intent.intent,
        intent_confidence=round(chat_resp.intent.confidence, 4),
        sentiment=chat_resp.sentiment.to_dict(),
        entities=chat_resp.entities,
        generated_by=chat_resp.generated_by,
        response_time_ms=chat_resp.response_time_ms,
        message_id=message_id,
        timestamp=datetime.utcnow().isoformat(),
    )


# ── WebSocket Chat ─────────────────────────────────────────────────────────────

@app.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str):
    """
    Real-time WebSocket chat endpoint.
    Client sends: {"message": "..."}
    Server sends: {same as REST /api/chat response}
    """
    await websocket.accept()
    logger.info(f"WS connected: session={session_id}")

    engine = get_response_engine()
    tracker = get_tracker()

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                user_message = payload.get("message", "").strip()
                if not user_message:
                    await websocket.send_json({"error": "Empty message"})
                    continue
            except json.JSONDecodeError:
                user_message = data.strip()

            chat_resp = await engine.process(user_message, session_id)

            message_id = tracker.log_exchange(
                session_id=session_id,
                user_message=user_message,
                bot_reply=chat_resp.text,
                intent=chat_resp.intent.intent,
                intent_confidence=chat_resp.intent.confidence,
                entities=chat_resp.entities,
                sentiment_label=chat_resp.sentiment.label,
                polarity=chat_resp.sentiment.polarity,
                subjectivity=chat_resp.sentiment.subjectivity,
                sentiment_emotion=chat_resp.sentiment.emotion,
                generated_by=chat_resp.generated_by,
                response_time_ms=chat_resp.response_time_ms,
            )

            await websocket.send_json({
                "reply": chat_resp.text,
                "session_id": session_id,
                "intent": chat_resp.intent.intent,
                "intent_confidence": round(chat_resp.intent.confidence, 4),
                "sentiment": chat_resp.sentiment.to_dict(),
                "entities": chat_resp.entities,
                "generated_by": chat_resp.generated_by,
                "response_time_ms": chat_resp.response_time_ms,
                "message_id": message_id,
                "timestamp": datetime.utcnow().isoformat(),
            })

    except WebSocketDisconnect:
        logger.info(f"WS disconnected: session={session_id}")


# ── Feedback ───────────────────────────────────────────────────────────────────

@app.post("/api/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(req: FeedbackRequest):
    """Record a thumbs up/down rating for a bot message."""
    store = get_feedback_store()
    success = store.record(
        message_id=req.message_id,
        rating=req.rating,
        comment=req.comment,
    )

    if success and req.intent:
        trainer = get_rl_trainer()
        trainer.update(intent=req.intent, reward=req.rating)

    return FeedbackResponse(
        success=success,
        message="Feedback recorded. Thank you! 🙏" if success else "Failed to record feedback.",
    )


# ── Analytics ──────────────────────────────────────────────────────────────────

@app.get("/api/analytics", tags=["Analytics"])
async def get_analytics():
    """Return all dashboard metrics."""
    agg = get_aggregator()
    return agg.get_dashboard_data()


@app.get("/api/analytics/sessions", tags=["Analytics"])
async def get_active_sessions():
    return {"active_sessions": get_context_manager().active_session_count}


# ── Session management ─────────────────────────────────────────────────────────

@app.delete("/api/session/{session_id}", tags=["Session"])
async def reset_session(session_id: str):
    """Clear the conversation context for a session (new chat)."""
    get_context_manager().reset(session_id)
    return {"success": True, "message": f"Session '{session_id}' reset."}


# ── Intent catalogue ───────────────────────────────────────────────────────────

@app.get("/api/intents", tags=["Info"])
async def list_intents():
    """Return all known intent labels and example utterances."""
    return {
        "intents": list(INTENT_EXAMPLES.keys()),
        "total": len(INTENT_EXAMPLES),
        "examples": INTENT_EXAMPLES,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
