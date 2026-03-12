"""Sentiment classifier with context-aware flag detection.

Uses DistilBERT fine-tuned on SST-2 for sentiment scoring, then applies
heuristic checks for sarcasm and backhanded compliments. Negative sentiment
combined with context flags raises the moderation score.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from modguard.classifiers.base import BaseClassifier
from modguard.config import ModelConfig
from modguard.models import SentimentResult

logger = logging.getLogger(__name__)

# Patterns for detecting sarcasm and backhanded compliments
_SARCASM_PATTERNS = [
    re.compile(r"\boh\s+sure\b", re.IGNORECASE),
    re.compile(r"\byeah\s+right\b", re.IGNORECASE),
    re.compile(r"\bwow\s+so\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+a\s+surprise\b", re.IGNORECASE),
    re.compile(r"\bsure\s+jan\b", re.IGNORECASE),
    re.compile(r"\btotally\b.*\bnot\b", re.IGNORECASE),
    re.compile(r"/s\b", re.IGNORECASE),
]

_BACKHANDED_PATTERNS = [
    re.compile(r"\bnot\s+bad\s+for\b", re.IGNORECASE),
    re.compile(r"\bpretty\s+good\s+for\b", re.IGNORECASE),
    re.compile(r"\bfor\s+someone\s+who\b", re.IGNORECASE),
    re.compile(r"\byou['re]*\s+actually\s+(?:pretty|quite|kind of)\b", re.IGNORECASE),
    re.compile(r"\bno\s+offense\s+but\b", re.IGNORECASE),
    re.compile(r"\bjust\s+saying\b", re.IGNORECASE),
    re.compile(r"\bI['m]*\s+surprised\s+(?:that|you)\b", re.IGNORECASE),
]


class SentimentClassifier(BaseClassifier):
    """Sentiment analysis classifier with context flag detection.

    Combines transformer-based sentiment scoring with heuristic pattern
    matching to detect sarcasm and backhanded compliments that might
    evade pure sentiment analysis.

    Attributes:
        model_name: HuggingFace model identifier for sentiment analysis.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        """Initialize the sentiment classifier.

        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        self._config = config or ModelConfig()
        self.model_name = self._config.sentiment_model
        self._pipeline = None

    def _load_model(self) -> None:
        """Lazy-load the HuggingFace sentiment pipeline."""
        try:
            from transformers import pipeline as hf_pipeline

            logger.info("Loading sentiment model: %s", self.model_name)
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self._config.device if self._config.device != "auto" else -1,
            )
            logger.info("Sentiment model loaded successfully.")
        except Exception as exc:
            logger.warning(
                "Failed to load sentiment model '%s': %s. "
                "Falling back to neutral scores.",
                self.model_name,
                exc,
            )
            self._pipeline = None

    def _detect_context_flags(self, text: str) -> list[str]:
        """Detect contextual flags like sarcasm and backhanded compliments.

        Args:
            text: The text to analyze.

        Returns:
            List of context flag strings.
        """
        flags: list[str] = []

        for pattern in _SARCASM_PATTERNS:
            if pattern.search(text):
                flags.append("sarcasm")
                break

        for pattern in _BACKHANDED_PATTERNS:
            if pattern.search(text):
                flags.append("backhanded_compliment")
                break

        return flags

    def _estimate_subjectivity(self, text: str) -> float:
        """Estimate text subjectivity using simple heuristics.

        Checks for opinion indicators, personal pronouns, and hedging
        language to approximate subjectivity.

        Args:
            text: The text to analyze.

        Returns:
            Subjectivity score from 0.0 (objective) to 1.0 (subjective).
        """
        subjectivity_markers = [
            r"\bI\s+think\b",
            r"\bI\s+feel\b",
            r"\bI\s+believe\b",
            r"\bin\s+my\s+opinion\b",
            r"\bpersonally\b",
            r"\bseems\s+like\b",
            r"\bprobably\b",
            r"\bmaybe\b",
            r"\bI\s+guess\b",
        ]

        hits = sum(
            1 for marker in subjectivity_markers
            if re.search(marker, text, re.IGNORECASE)
        )
        # Normalize: each marker contributes roughly 0.15
        return min(1.0, hits * 0.15)

    def classify(self, text: str) -> SentimentResult:
        """Classify text sentiment and detect contextual flags.

        Args:
            text: The text content to classify.

        Returns:
            A SentimentResult with sentiment score, subjectivity,
            and any detected context flags.
        """
        if self._pipeline is None:
            self._load_model()

        context_flags = self._detect_context_flags(text)
        subjectivity = self._estimate_subjectivity(text)

        if self._pipeline is None:
            return SentimentResult(
                sentiment_score=0.0,
                subjectivity=subjectivity,
                context_flags=context_flags,
                raw_label="NEUTRAL",
                raw_score=0.5,
            )

        try:
            truncated = text[:512]
            result = self._pipeline(truncated)[0]

            raw_label = result["label"]
            raw_score = float(result["score"])

            # Convert to -1.0 to 1.0 scale
            if raw_label.upper() == "POSITIVE":
                sentiment_score = raw_score
            elif raw_label.upper() == "NEGATIVE":
                sentiment_score = -raw_score
            else:
                sentiment_score = 0.0

            return SentimentResult(
                sentiment_score=sentiment_score,
                subjectivity=subjectivity,
                context_flags=context_flags,
                raw_label=raw_label,
                raw_score=raw_score,
            )

        except Exception as exc:
            logger.warning("Sentiment classification failed: %s", exc)
            return SentimentResult(
                sentiment_score=0.0,
                subjectivity=subjectivity,
                context_flags=context_flags,
                raw_label="ERROR",
                raw_score=0.0,
            )
