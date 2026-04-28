"""
nlp/sentiment_analyzer.py — Sentiment & Emotion detection using TextBlob.

Outputs:
  - polarity    float  -1.0 (very negative) → +1.0 (very positive)
  - subjectivity float  0.0 (objective) → 1.0 (subjective/opinionated)
  - label       str    "positive" | "neutral" | "negative"
  - emotion     str    coarse emotion mapped from polarity+subjectivity

Emotion mapping heuristic:
  polarity > 0.5  → joy
  polarity > 0.2  → optimism
  polarity > 0.05 → content
  polarity > -0.05 → neutral
  polarity > -0.2  → dissatisfied
  polarity > -0.5  → frustration / anger (combined with subjectivity)
  polarity ≤ -0.5  → anger | sadness (subjectivity decides)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from textblob import TextBlob


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    polarity: float
    subjectivity: float
    label: str       # positive | neutral | negative
    emotion: str     # joy | optimism | content | neutral | dissatisfied | frustration | anger | sadness

    # Emoji shorthand for UI display
    @property
    def emoji(self) -> str:
        mapping = {
            "joy": "😊",
            "optimism": "🙂",
            "content": "😌",
            "neutral": "😐",
            "dissatisfied": "😕",
            "frustration": "😤",
            "anger": "😠",
            "sadness": "😢",
        }
        return mapping.get(self.emotion, "😐")

    # Response tone hint used by the response engine
    @property
    def tone_hint(self) -> str:
        if self.label == "positive":
            return "enthusiastic and warm"
        if self.label == "negative":
            return "empathetic and supportive"
        return "clear and concise"

    def to_dict(self) -> dict:
        return {
            "polarity": round(self.polarity, 4),
            "subjectivity": round(self.subjectivity, 4),
            "label": self.label,
            "emotion": self.emotion,
            "emoji": self.emoji,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Wraps TextBlob sentiment analysis with coarse emotion classification.

    Usage:
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("I love this! It's amazing!")
        # SentimentResult(polarity=0.625, subjectivity=0.6, label='positive', emotion='joy')
    """

    # Thresholds for label classification
    POSITIVE_THRESHOLD = 0.08
    NEGATIVE_THRESHOLD = -0.08

    def analyze(self, text: str) -> SentimentResult:
        """Analyse *text* and return a SentimentResult."""
        blob = TextBlob(text)
        polarity: float = blob.sentiment.polarity
        subjectivity: float = blob.sentiment.subjectivity

        label = self._classify_label(polarity)
        emotion = self._classify_emotion(polarity, subjectivity)

        return SentimentResult(
            polarity=polarity,
            subjectivity=subjectivity,
            label=label,
            emotion=emotion,
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyse multiple texts at once."""
        return [self.analyze(t) for t in texts]

    # ── Private helpers ───────────────────────────────────────

    def _classify_label(self, polarity: float) -> str:
        if polarity > self.POSITIVE_THRESHOLD:
            return "positive"
        if polarity < self.NEGATIVE_THRESHOLD:
            return "negative"
        return "neutral"

    @staticmethod
    def _classify_emotion(polarity: float, subjectivity: float) -> str:
        if polarity > 0.5:
            return "joy"
        if polarity > 0.2:
            return "optimism"
        if polarity > 0.05:
            return "content"
        if polarity > -0.05:
            return "neutral"
        if polarity > -0.2:
            return "dissatisfied"
        if polarity > -0.5:
            return "frustration"
        # Very negative — subjectivity determines anger vs sadness
        return "anger" if subjectivity > 0.5 else "sadness"


# ── Module-level singleton ─────────────────────────────────────────────────────
_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer
