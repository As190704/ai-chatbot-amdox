"""
nlp/intent_classifier.py — TF-IDF + cosine-similarity intent recognition.

How it works:
  1. A set of training examples (utterances + intent label) is defined.
  2. At startup they are vectorised into a TF-IDF matrix.
  3. Incoming messages are projected into the same space; the most
     similar training utterance wins (cosine similarity).
  4. If the winning score is below INTENT_CONFIDENCE_THRESHOLD the
     intent is marked as "unknown" so the LLM fallback is triggered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import settings

# ─────────────────────────────────────────────────────────────────────────────
# Training data  (intent → list of representative utterances)
# ─────────────────────────────────────────────────────────────────────────────

INTENT_EXAMPLES: dict[str, list[str]] = {
    "greet": [
        "hello", "hi", "hey", "good morning", "good evening",
        "howdy", "what's up", "how are you", "greetings", "hi there",
        "hey bot", "hello there", "yo", "sup",
    ],
    "farewell": [
        "bye", "goodbye", "see you", "see ya", "later", "take care",
        "farewell", "quit", "exit", "cya", "good night", "talk to you later",
    ],
    "thanks": [
        "thank you", "thanks", "thank you so much", "many thanks",
        "cheers", "much appreciated", "grateful", "thx", "ty",
    ],
    "help": [
        "help", "help me", "what can you do", "how do you work",
        "what are your features", "can you help me", "i need help",
        "show me what you can do", "what do you know",
    ],
    "weather": [
        "what is the weather", "weather today", "is it raining",
        "temperature outside", "forecast", "will it rain tomorrow",
        "weather in new york", "how hot is it", "is it cold today",
    ],
    "time": [
        "what time is it", "current time", "tell me the time",
        "what is the date today", "what day is it", "today's date",
    ],
    "joke": [
        "tell me a joke", "make me laugh", "say something funny",
        "joke please", "do you know any jokes", "entertain me",
    ],
    "news": [
        "latest news", "what is happening in the world", "news today",
        "current events", "headlines", "tell me the news",
    ],
    "definition": [
        "what is", "define", "meaning of", "explain", "tell me about",
        "what does mean", "definition of", "describe",
    ],
    "math": [
        "calculate", "what is 2 plus 2", "solve", "compute",
        "math problem", "how much is", "add subtract multiply divide",
    ],
    "recommendation": [
        "recommend", "suggest", "what should i", "best option",
        "what do you recommend", "give me suggestions", "advise me",
    ],
    "personal": [
        "what is your name", "who are you", "are you a robot",
        "are you human", "who made you", "your creator", "your age",
        "tell me about yourself", "what are you",
    ],
    "capabilities": [
        "what can you do", "list your features", "your abilities",
        "what are you capable of", "show me features",
    ],
    "complaint": [
        "this is wrong", "bad response", "you are useless",
        "i am frustrated", "terrible", "awful", "unhelpful",
        "not working", "broken", "I hate this",
    ],
    "affirmation": [
        "yes", "yeah", "yep", "correct", "right", "sure", "ok",
        "okay", "absolutely", "of course", "definitely", "agreed",
    ],
    "negation": [
        "no", "nope", "not really", "i disagree", "wrong", "incorrect",
        "negative", "nah",
    ],
    "smalltalk": [
        "how are you doing", "what are you up to", "how's life",
        "bored", "just chatting", "talk to me", "let's chat",
        "i am lonely", "keep me company",
    ],
    "search": [
        "search for", "look up", "find me", "google", "i want to know about",
        "give me information about", "research",
    ],
    "unknown": [
        "asdfghjkl", "xyzzy", "qwerty123",  # anchor for low-similarity matches
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentResult:
    intent: str
    confidence: float
    all_scores: dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Lightweight TF-IDF intent classifier.

    Usage:
        clf = IntentClassifier()
        result = clf.classify("hello there!")
        # IntentResult(intent='greet', confidence=0.87, ...)
    """

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer="word",
            min_df=1,
            sublinear_tf=True,
        )
        self._intents: list[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._build()

    # ── Private ───────────────────────────────────────────────

    def _preprocess(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _build(self) -> None:
        """Fit the vectorizer on all training utterances."""
        corpus: list[str] = []
        labels: list[str] = []
        for intent, utterances in INTENT_EXAMPLES.items():
            for utt in utterances:
                corpus.append(self._preprocess(utt))
                labels.append(intent)

        self._intents = labels
        self._matrix = self._vectorizer.fit_transform(corpus)

    # ── Public ────────────────────────────────────────────────

    def classify(self, text: str) -> IntentResult:
        """
        Classify *text* and return an IntentResult.
        If the best score < threshold → intent = 'unknown'.
        """
        cleaned = self._preprocess(text)
        vec = self._vectorizer.transform([cleaned])
        sims: np.ndarray = cosine_similarity(vec, self._matrix)[0]

        # Group by intent (take max similarity per intent)
        intent_scores: dict[str, float] = {}
        for idx, score in enumerate(sims):
            intent = self._intents[idx]
            if intent not in intent_scores or score > intent_scores[intent]:
                intent_scores[intent] = float(score)

        # Remove the "unknown" anchor — we handle it via threshold
        intent_scores.pop("unknown", None)

        if not intent_scores:
            return IntentResult(intent="unknown", confidence=0.0)

        best_intent = max(intent_scores, key=intent_scores.__getitem__)
        best_score = intent_scores[best_intent]

        if best_score < settings.intent_confidence_threshold:
            return IntentResult(intent="unknown", confidence=best_score, all_scores=intent_scores)

        return IntentResult(intent=best_intent, confidence=best_score, all_scores=intent_scores)

    def top_n(self, text: str, n: int = 3) -> list[tuple[str, float]]:
        """Return top-n (intent, score) pairs."""
        result = self.classify(text)
        sorted_scores = sorted(result.all_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:n]


# ── Module-level singleton ─────────────────────────────────────────────────────
_classifier: Optional[IntentClassifier] = None


def get_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier
