"""
nlp/ner_extractor.py — Named Entity Recognition using spaCy.

Extracts entities such as persons, organisations, locations, dates,
products, money, etc. from user messages.

spaCy model required:
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# spaCy lazy-load (avoids heavy startup cost when NER is not needed)
# ─────────────────────────────────────────────────────────────────────────────

_nlp = None  # lazy


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy  # noqa: PLC0415
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model 'en_core_web_sm' loaded.")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm\n"
                "NER will be disabled until the model is installed."
            )
            _nlp = None
    return _nlp


# ─────────────────────────────────────────────────────────────────────────────
# Entity label mapping — spaCy label → human-friendly category
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP: dict[str, str] = {
    "PERSON":   "person",
    "ORG":      "organization",
    "GPE":      "location",
    "LOC":      "location",
    "DATE":     "date",
    "TIME":     "time",
    "MONEY":    "money",
    "PRODUCT":  "product",
    "EVENT":    "event",
    "WORK_OF_ART": "work_of_art",
    "LANGUAGE": "language",
    "NORP":     "group",
    "FAC":      "facility",
    "LAW":      "law",
    "PERCENT":  "percentage",
    "QUANTITY": "quantity",
    "CARDINAL": "number",
    "ORDINAL":  "ordinal",
}


@dataclass
class Entity:
    text: str
    label: str          # human-friendly label from LABEL_MAP
    start: int          # character offset in original text
    end: int

    def to_dict(self) -> dict:
        return {"text": self.text, "label": self.label, "start": self.start, "end": self.end}


# ─────────────────────────────────────────────────────────────────────────────
# Extractor
# ─────────────────────────────────────────────────────────────────────────────

class NERExtractor:
    """
    Named Entity Recogniser wrapping spaCy's en_core_web_sm pipeline.

    Usage:
        extractor = NERExtractor()
        entities = extractor.extract("Book a flight to Paris on Friday")
        # [Entity(text='Paris', label='location', ...), Entity(text='Friday', label='date', ...)]
    """

    def extract(self, text: str) -> list[Entity]:
        """
        Run NER on *text* and return a list of Entity objects.
        Returns an empty list if spaCy model is unavailable.
        """
        nlp = _get_nlp()
        if nlp is None:
            return []

        doc = nlp(text)
        entities: list[Entity] = []

        for ent in doc.ents:
            label = LABEL_MAP.get(ent.label_, ent.label_.lower())
            entities.append(
                Entity(
                    text=ent.text,
                    label=label,
                    start=ent.start_char,
                    end=ent.end_char,
                )
            )

        return entities

    def extract_as_dicts(self, text: str) -> list[dict]:
        """Convenience wrapper returning plain dicts for JSON serialisation."""
        return [e.to_dict() for e in self.extract(text)]

    def group_by_label(self, text: str) -> dict[str, list[str]]:
        """
        Return a dict mapping entity category → list of entity texts.

        Example:
            {"location": ["Paris", "London"], "date": ["Friday"]}
        """
        grouped: dict[str, list[str]] = {}
        for ent in self.extract(text):
            grouped.setdefault(ent.label, []).append(ent.text)
        return grouped


# ── Module-level singleton ─────────────────────────────────────────────────────
_extractor: Optional[NERExtractor] = None


def get_ner_extractor() -> NERExtractor:
    global _extractor
    if _extractor is None:
        _extractor = NERExtractor()
    return _extractor
