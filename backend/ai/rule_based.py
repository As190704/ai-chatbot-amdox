"""
ai/rule_based.py — Pattern-matching & FAQ response engine.

Structure:
  PATTERNS  → list of (regex, response_list) pairs, checked first.
  FAQ_PAIRS → dict of canonical question → answer, matched via simple
               string containment after normalisation.
  INTENT_RESPONSES → dict of intent label → list of canned responses;
               one is chosen at random for variety.

The engine returns None if no rule matches, signalling the caller to
escalate to an LLM.
"""

from __future__ import annotations

import random
import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Pattern rules  (checked in order — put specifics before generals)
# ─────────────────────────────────────────────────────────────────────────────

PATTERNS: list[tuple[str, list[str]]] = [
    # ── Math ──────────────────────────────────────────────────
    (
        r"what(?:'s| is) (\d+)\s*[\+plus]+\s*(\d+)",
        ["That's {0} + {1} = {result}! 🧮"],
    ),
    (
        r"what(?:'s| is) (\d+)\s*[\-minus]+\s*(\d+)",
        ["Let me calculate: {0} - {1} = {result} ✅"],
    ),
    (
        r"what(?:'s| is) (\d+)\s*[\*x×times]+\s*(\d+)",
        ["{0} × {1} = {result} 🎉"],
    ),
    (
        r"what(?:'s| is) (\d+)\s*[\/÷divided by]+\s*(\d+)",
        ["{0} ÷ {1} = {result:.4g} 📐"],
    ),
    # ── Time / Date ────────────────────────────────────────────
    (
        r"\b(time|clock)\b",
        ["__CURRENT_TIME__"],
    ),
    (
        r"\b(date|today|day)\b",
        ["__CURRENT_DATE__"],
    ),
    # ── Joke ──────────────────────────────────────────────────
    (
        r"\b(joke|funny|laugh|humor)\b",
        [
            "Why don't scientists trust atoms? Because they make up everything! 😄",
            "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads. 🍫",
            "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
            "What do you call a fake noodle? An impasta! 🍝",
            "Why did the scarecrow win an award? Because he was outstanding in his field! 🌾",
            "How does a computer get drunk? It takes screenshots! 📸",
            "Why do Java developers wear glasses? Because they don't C#! 👓",
        ],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Intent-based responses  (used when intent is known but no pattern matches)
# ─────────────────────────────────────────────────────────────────────────────

INTENT_RESPONSES: dict[str, list[str]] = {
    "greet": [
        "Hello! 👋 How can I assist you today?",
        "Hey there! Great to see you. What's on your mind?",
        "Hi! I'm AMDOX AI, your intelligent assistant. How can I help? 🤖",
        "Greetings! Ready to assist you. What would you like to know?",
        "Hello! 😊 I'm here and ready. Ask me anything!",
    ],
    "farewell": [
        "Goodbye! Have a wonderful day! 👋",
        "See you later! Feel free to come back anytime. 😊",
        "Take care! It was great chatting with you. 🌟",
        "Farewell! Don't hesitate to reach out whenever you need help.",
        "Bye! Have a fantastic day ahead! ✨",
    ],
    "thanks": [
        "You're very welcome! 😊 Is there anything else I can help with?",
        "Happy to help! Let me know if you need anything else.",
        "My pleasure! That's what I'm here for. 🤖",
        "Anytime! Feel free to ask more questions.",
        "Glad I could assist! 🌟",
    ],
    "help": [
        (
            "Here's what I can help you with:\n\n"
            "• 💬 **General conversation** — ask me anything!\n"
            "• 🧮 **Math** — arithmetic calculations\n"
            "• 😂 **Jokes** — I know a few good ones\n"
            "• 🕒 **Time & Date** — current time and date\n"
            "• 📖 **Definitions** — explain concepts\n"
            "• 💡 **Recommendations** — suggestions on various topics\n"
            "• 😊 **Sentiment-aware** — I adapt to your mood\n\n"
            "Just type naturally and I'll do my best!"
        )
    ],
    "personal": [
        (
            "I'm **AMDOX AI** 🤖 — a Dynamic AI Chatbot powered by NLP and Machine Learning.\n\n"
            "I understand natural language, remember our conversation, detect your sentiment, "
            "and learn from interactions. I was built to make conversations smarter and more helpful!"
        ),
        "I'm AMDOX AI, your intelligent assistant! I use NLP, ML, and generative AI to chat naturally. 🧠",
    ],
    "capabilities": [
        (
            "My capabilities include:\n\n"
            "🧠 **NLP** — Intent recognition & Named Entity Extraction\n"
            "💬 **Context Memory** — I remember our conversation\n"
            "😊 **Sentiment Analysis** — I detect your mood\n"
            "🤖 **AI Responses** — Rule-based, GPT, and local LLM\n"
            "📊 **Analytics** — Tracks conversation patterns\n"
            "👍 **Self-Learning** — Improves from your feedback"
        )
    ],
    "complaint": [
        "I'm really sorry to hear that! 😔 I'm constantly learning and improving. Your feedback helps me get better.",
        "I apologise for the frustration. Let me try to help you better — could you rephrase your question?",
        "I understand your frustration and I'm sorry. Please let me know what went wrong so I can improve.",
    ],
    "affirmation": [
        "Great! 😊 How can I help you further?",
        "Awesome! Let me know what you'd like to explore next.",
        "Perfect! What else can I do for you?",
    ],
    "negation": [
        "I see. Could you clarify what you're looking for? I want to make sure I give you the right answer.",
        "Got it! Let me know how I can do better. What would you like instead?",
        "Understood. Please tell me more so I can help accurately.",
    ],
    "smalltalk": [
        "I'm doing great, thanks for asking! 😊 I love chatting. What's on your mind today?",
        "As an AI I don't have feelings, but I'm fully charged and ready to help! ⚡ What about you?",
        "Always happy to chat! Tell me something interesting or ask me anything. 🌟",
    ],
    "recommendation": [
        "I'd love to give you a recommendation! Could you tell me a bit more about what you're looking for? 🤔",
        "Great question! To give you the best suggestion, could you provide a little more context?",
    ],
    "search": [
        "I'd search that for you, but I don't have live internet access. Try Google or let me answer from what I know! 🔍",
        "I can answer from my training knowledge, but for the very latest info you'd want to search online. What specifically would you like to know?",
    ],
    "math": [
        "I can do basic arithmetic! Try asking something like 'what is 25 + 37?' or 'calculate 144 / 12'. 🧮",
    ],
    "news": [
        "I don't have live news access, but I can discuss topics I know about. What subject are you interested in? 📰",
    ],
    "weather": [
        "I don't have live weather data, but you can check weather.com or your phone's weather app for accurate forecasts! 🌤️",
    ],
    "unknown": [
        "Hmm, that's an interesting one! Could you rephrase or give me a bit more context? I want to help you accurately. 🤔",
        "I'm not quite sure I understood that. Could you try rephrasing? I'm still learning! 🧠",
        "That's a great question — let me think... Could you elaborate a bit more so I can give you a better answer?",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# FAQ pairs  (exact/partial match)
# ─────────────────────────────────────────────────────────────────────────────

FAQ_PAIRS: list[tuple[str, str]] = [
    ("what is artificial intelligence", "Artificial Intelligence (AI) is the simulation of human intelligence in machines, enabling them to learn, reason, and solve problems. 🤖"),
    ("what is machine learning", "Machine Learning is a subset of AI where models learn from data to make predictions or decisions without being explicitly programmed. 📊"),
    ("what is deep learning", "Deep Learning uses multi-layered neural networks to model complex patterns in data — it's the engine behind image recognition, NLP, and more. 🧠"),
    ("what is nlp", "Natural Language Processing (NLP) is the field of AI that enables computers to understand, interpret, and generate human language. 💬"),
    ("what is python", "Python is a versatile, high-level programming language known for its simplicity and vast ecosystem, especially popular in data science and AI. 🐍"),
    ("capital of india", "The capital of India is **New Delhi** 🇮🇳"),
    ("capital of france", "The capital of France is **Paris** 🇫🇷"),
    ("capital of usa", "The capital of the USA is **Washington, D.C.** 🇺🇸"),
    ("who is elon musk", "Elon Musk is a tech entrepreneur known for founding SpaceX, Tesla, Neuralink, and xAI. He's one of the world's most influential innovators. 🚀"),
    ("what is chatgpt", "ChatGPT is an AI conversational model developed by OpenAI, trained on a large corpus of text to generate human-like responses. 🤖"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedEngine:
    """
    Tries pattern matching → FAQ matching → intent-canned response.
    Returns None if nothing matches (signal to escalate to LLM).
    """

    def __init__(self) -> None:
        # Pre-compile all regex patterns
        self._compiled: list[tuple[re.Pattern, list[str]]] = [
            (re.compile(pat, re.IGNORECASE), responses)
            for pat, responses in PATTERNS
        ]

    # ── Public ────────────────────────────────────────────────

    def get_response(self, text: str, intent: str = "unknown") -> Optional[str]:
        """
        Try all matching strategies and return a response string, or None.
        """
        # 1. Regex patterns (highest priority)
        pattern_resp = self._match_pattern(text)
        if pattern_resp:
            return pattern_resp

        # 2. FAQ pairs
        faq_resp = self._match_faq(text)
        if faq_resp:
            return faq_resp

        # 3. Intent-canned responses
        intent_resp = self._match_intent(intent)
        if intent_resp:
            return intent_resp

        return None

    # ── Private ───────────────────────────────────────────────

    def _match_pattern(self, text: str) -> Optional[str]:
        import datetime as dt
        for pattern, responses in self._compiled:
            m = pattern.search(text)
            if m:
                template = random.choice(responses)

                # Special dynamic substitutions
                if "__CURRENT_TIME__" in template:
                    return f"The current time is **{dt.datetime.now().strftime('%H:%M:%S')}** ⏰"
                if "__CURRENT_DATE__" in template:
                    return f"Today is **{dt.datetime.now().strftime('%A, %B %d, %Y')}** 📅"

                # Math substitution
                if "{result}" in template:
                    try:
                        a, b = float(m.group(1)), float(m.group(2))
                        op_char = re.search(r"[\+\-\*\/x×÷]", text.lower())
                        op = op_char.group(0) if op_char else "+"
                        result: float
                        if op in ("+", "plus"):
                            result = a + b
                        elif op in ("-", "minus"):
                            result = a - b
                        elif op in ("*", "x", "×", "times"):
                            result = a * b
                        else:
                            result = a / b if b != 0 else float("inf")
                        return template.format(
                            int(a) if a == int(a) else a,
                            int(b) if b == int(b) else b,
                            result=int(result) if result == int(result) else result,
                        )
                    except Exception:
                        pass

                return template
        return None

    def _match_faq(self, text: str) -> Optional[str]:
        normalised = text.lower().strip()
        for question, answer in FAQ_PAIRS:
            if question in normalised or normalised in question:
                return answer
        return None

    def _match_intent(self, intent: str) -> Optional[str]:
        responses = INTENT_RESPONSES.get(intent)
        if responses:
            return random.choice(responses)
        return None


# ── Module-level singleton ─────────────────────────────────────────────────────
_engine: Optional[RuleBasedEngine] = None


def get_rule_engine() -> RuleBasedEngine:
    global _engine
    if _engine is None:
        _engine = RuleBasedEngine()
    return _engine
