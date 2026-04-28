"""
learning/rl_trainer.py — Reinforcement-learning-flavoured response reweighting.

Current approach: a simple reward-based weight update on rule-based intent
responses. When a user rates a bot response 👍 (positive) the intent that
produced it receives a small weight boost; 👎 produces a weight penalty.

The weights are persisted to a JSON sidecar file so they survive restarts.
A future version could train a proper bandit / RLHF pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Weights are stored next to the backend module
WEIGHTS_PATH = Path(__file__).parent.parent / "data" / "intent_weights.json"

REWARD_STEP = 0.05     # boost per thumbs-up
PENALTY_STEP = 0.08    # penalty per thumbs-down
MIN_WEIGHT = 0.1
MAX_WEIGHT = 2.0
DEFAULT_WEIGHT = 1.0


class RLTrainer:
    """
    Maintains a weight table: {intent → float} used to bias response
    selection in the rule-based engine.

    Usage:
        trainer = RLTrainer()
        trainer.update(intent="greet", reward=True)
        weight = trainer.get_weight("greet")
    """

    def __init__(self) -> None:
        self._weights: dict[str, float] = {}
        self._load()

    # ── Private ───────────────────────────────────────────────

    def _load(self) -> None:
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if WEIGHTS_PATH.exists():
            try:
                with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
                    self._weights = json.load(f)
                logger.info(f"RLTrainer: Loaded weights for {len(self._weights)} intents.")
            except Exception as exc:
                logger.warning(f"RLTrainer: Could not load weights — {exc}. Starting fresh.")
                self._weights = {}
        else:
            logger.info("RLTrainer: No existing weights file. Starting fresh.")

    def _save(self) -> None:
        try:
            with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
                json.dump(self._weights, f, indent=2)
        except Exception as exc:
            logger.error(f"RLTrainer: Could not save weights — {exc}")

    # ── Public ────────────────────────────────────────────────

    def get_weight(self, intent: str) -> float:
        return self._weights.get(intent, DEFAULT_WEIGHT)

    def update(self, intent: str, reward: bool) -> float:
        """
        Apply a reward (+) or penalty (-) to *intent*.
        Returns the new weight.
        """
        current = self._weights.get(intent, DEFAULT_WEIGHT)
        if reward:
            new_weight = min(current + REWARD_STEP, MAX_WEIGHT)
            logger.debug(f"RLTrainer: Reward '{intent}' → {current:.2f} → {new_weight:.2f}")
        else:
            new_weight = max(current - PENALTY_STEP, MIN_WEIGHT)
            logger.debug(f"RLTrainer: Penalty '{intent}' → {current:.2f} → {new_weight:.2f}")

        self._weights[intent] = round(new_weight, 4)
        self._save()
        return self._weights[intent]

    def get_all_weights(self) -> dict[str, float]:
        return dict(self._weights)

    def reset(self) -> None:
        """Reset all weights to defaults and save."""
        self._weights = {}
        self._save()
        logger.info("RLTrainer: All weights reset.")


# ── Module-level singleton ─────────────────────────────────────────────────────
_trainer: Optional[RLTrainer] = None


def get_rl_trainer() -> RLTrainer:
    global _trainer
    if _trainer is None:
        _trainer = RLTrainer()
    return _trainer
