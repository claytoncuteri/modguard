"""Toxicity classifier using a HuggingFace transformer model.

Wraps the martin-ha/toxic-comment-model to produce multi-label toxicity
scores across categories: toxic, severe_toxic, obscene, threat, insult,
and identity_hate. The model is lazy-loaded on first use to avoid slow
startup when toxicity classification is not needed.
"""

from __future__ import annotations

import logging
from typing import Optional

from modguard.classifiers.base import BaseClassifier
from modguard.config import ModelConfig
from modguard.models import ToxicityResult

logger = logging.getLogger(__name__)

# Toxicity categories aligned with the Jigsaw dataset labels
TOXICITY_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


class ToxicityClassifier(BaseClassifier):
    """Transformer-based toxicity classifier.

    Uses a HuggingFace text-classification pipeline to score text across
    multiple toxicity dimensions. The model is loaded lazily on first
    classify() call.

    Attributes:
        model_name: HuggingFace model identifier.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        """Initialize the toxicity classifier.

        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        self._config = config or ModelConfig()
        self.model_name = self._config.toxicity_model
        self._pipeline = None

    def _load_model(self) -> None:
        """Lazy-load the HuggingFace classification pipeline.

        Called automatically on first classify() invocation. Logs a
        warning if the model cannot be loaded.
        """
        try:
            from transformers import pipeline as hf_pipeline

            logger.info("Loading toxicity model: %s", self.model_name)
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self.model_name,
                device=self._config.device if self._config.device != "auto" else -1,
                top_k=None,
            )
            logger.info("Toxicity model loaded successfully.")
        except Exception as exc:
            logger.warning(
                "Failed to load toxicity model '%s': %s. "
                "Falling back to zero scores.",
                self.model_name,
                exc,
            )
            self._pipeline = None

    def classify(self, text: str) -> ToxicityResult:
        """Classify text for toxicity across multiple categories.

        On first call, loads the ML model. If the model is unavailable,
        returns zero scores for all categories.

        Args:
            text: The text content to classify.

        Returns:
            A ToxicityResult with per-category scores and an overall score.
        """
        if self._pipeline is None:
            self._load_model()

        if self._pipeline is None:
            # Model unavailable, return neutral result
            return ToxicityResult(
                labels={label: 0.0 for label in TOXICITY_LABELS},
                overall_score=0.0,
                model_name=self.model_name,
            )

        try:
            # Truncate long texts to avoid model input limits
            truncated = text[:512]
            results = self._pipeline(truncated)

            # Parse model output into label scores
            labels: dict[str, float] = {}
            if results and isinstance(results[0], list):
                for item in results[0]:
                    label = item["label"].lower()
                    score = float(item["score"])
                    # Map model labels to our standard categories
                    if label in ("toxic", "label_1", "positive"):
                        labels["toxic"] = score
                    elif label in ("non-toxic", "label_0", "negative"):
                        labels["toxic"] = 1.0 - score
                    else:
                        labels[label] = score
            elif results and isinstance(results[0], dict):
                label = results[0]["label"].lower()
                score = float(results[0]["score"])
                if label in ("toxic", "label_1", "positive"):
                    labels["toxic"] = score
                elif label in ("non-toxic", "label_0", "negative"):
                    labels["toxic"] = 1.0 - score

            # Fill missing categories with zero
            for cat in TOXICITY_LABELS:
                if cat not in labels:
                    labels[cat] = 0.0

            # Overall score is the maximum across all categories
            overall = max(labels.values()) if labels else 0.0

            return ToxicityResult(
                labels=labels,
                overall_score=overall,
                model_name=self.model_name,
            )

        except Exception as exc:
            logger.warning("Toxicity classification failed: %s", exc)
            return ToxicityResult(
                labels={label: 0.0 for label in TOXICITY_LABELS},
                overall_score=0.0,
                model_name=self.model_name,
            )
