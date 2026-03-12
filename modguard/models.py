"""Data models for moderation pipeline results.

Each classifier layer produces a typed result, and the ensemble combines
them into a final ModerationResult with a decision, confidence score,
and human-readable explanation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Decision(str, Enum):
    """Possible moderation decisions."""

    APPROVE = "APPROVE"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    REJECT = "REJECT"


@dataclass
class RuleResult:
    """Result from the rule-based classifier.

    Attributes:
        matched_rules: List of rule names that matched the input.
        severity: Severity level from 0.0 (none) to 1.0 (critical).
        confidence: Confidence in the result. Always 1.0 for rule matches
            since rules are deterministic.
        details: Optional additional details about matched rules.
    """

    matched_rules: list[str] = field(default_factory=list)
    severity: float = 0.0
    confidence: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "matched_rules": self.matched_rules,
            "severity": self.severity,
            "confidence": self.confidence,
            "details": self.details,
        }


@dataclass
class ToxicityResult:
    """Result from the toxicity classifier.

    Attributes:
        labels: Dictionary mapping toxicity categories to their scores.
            Categories include: toxic, severe_toxic, obscene, threat,
            insult, identity_hate.
        overall_score: Aggregate toxicity score from 0.0 to 1.0.
        model_name: Name of the model used for classification.
    """

    labels: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    model_name: str = "martin-ha/toxic-comment-model"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "labels": self.labels,
            "overall_score": self.overall_score,
            "model_name": self.model_name,
        }


@dataclass
class SentimentResult:
    """Result from the sentiment classifier.

    Attributes:
        sentiment_score: Score from -1.0 (very negative) to 1.0 (very positive).
        subjectivity: How subjective the text is, from 0.0 to 1.0.
        context_flags: List of contextual flags such as 'sarcasm' or
            'backhanded_compliment'.
        raw_label: The raw label from the model.
        raw_score: The raw confidence score from the model.
    """

    sentiment_score: float = 0.0
    subjectivity: float = 0.0
    context_flags: list[str] = field(default_factory=list)
    raw_label: str = ""
    raw_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sentiment_score": self.sentiment_score,
            "subjectivity": self.subjectivity,
            "context_flags": self.context_flags,
            "raw_label": self.raw_label,
            "raw_score": self.raw_score,
        }


@dataclass
class ModerationResult:
    """Final moderation result combining all classifier layers.

    Attributes:
        id: Unique identifier for this moderation result.
        text: The original text that was moderated.
        decision: The moderation decision (APPROVE, FLAG_FOR_REVIEW, REJECT).
        confidence: Overall confidence in the decision from 0.0 to 1.0.
        layer_results: Dictionary of results from each classifier layer.
        processing_time_ms: Time taken to process, in milliseconds.
        explanation: Human-readable explanation of the decision.
        timestamp: Unix timestamp when the result was created.
    """

    text: str = ""
    decision: Decision = Decision.APPROVE
    confidence: float = 0.0
    layer_results: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    explanation: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        serialized_layers = {}
        for name, result in self.layer_results.items():
            if hasattr(result, "to_dict"):
                serialized_layers[name] = result.to_dict()
            else:
                serialized_layers[name] = result

        return {
            "id": self.id,
            "text": self.text,
            "decision": self.decision.value,
            "confidence": round(self.confidence, 4),
            "layer_results": serialized_layers,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "explanation": self.explanation,
            "timestamp": self.timestamp,
        }
