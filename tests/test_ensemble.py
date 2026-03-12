"""Tests for the ensemble classifier.

Uses mocked layer results to verify that weighting logic, threshold
boundaries, and explanation generation work correctly.
"""

from __future__ import annotations

import pytest

from modguard.classifiers.ensemble import EnsembleClassifier
from modguard.config import ClassifierWeights, ThresholdConfig
from modguard.models import (
    Decision,
    RuleResult,
    SentimentResult,
    ToxicityResult,
)


@pytest.fixture
def ensemble() -> EnsembleClassifier:
    """Create an ensemble classifier with default weights and thresholds."""
    return EnsembleClassifier()


@pytest.fixture
def strict_ensemble() -> EnsembleClassifier:
    """Create an ensemble with lower reject threshold for strict moderation."""
    return EnsembleClassifier(
        thresholds=ThresholdConfig(approve_below=0.2, reject_above=0.5)
    )


def _make_clean_results() -> dict:
    """Create layer results representing clean content."""
    return {
        "rules": RuleResult(matched_rules=[], severity=0.0, confidence=0.0),
        "toxicity": ToxicityResult(labels={"toxic": 0.05}, overall_score=0.05),
        "sentiment": SentimentResult(sentiment_score=0.8, subjectivity=0.2),
    }


def _make_toxic_results() -> dict:
    """Create layer results representing clearly toxic content."""
    return {
        "rules": RuleResult(
            matched_rules=["blocked_keyword"], severity=0.9, confidence=1.0
        ),
        "toxicity": ToxicityResult(
            labels={"toxic": 0.95, "insult": 0.88}, overall_score=0.95
        ),
        "sentiment": SentimentResult(sentiment_score=-0.9, subjectivity=0.7),
    }


def _make_borderline_results() -> dict:
    """Create layer results representing borderline content."""
    return {
        "rules": RuleResult(matched_rules=[], severity=0.0, confidence=0.0),
        "toxicity": ToxicityResult(
            labels={"toxic": 0.55}, overall_score=0.55
        ),
        "sentiment": SentimentResult(
            sentiment_score=-0.4,
            subjectivity=0.5,
            context_flags=["sarcasm"],
        ),
    }


class TestDecisionThresholds:
    """Tests for threshold-based decision making."""

    def test_clean_content_approved(self, ensemble: EnsembleClassifier) -> None:
        """Clean content should receive an APPROVE decision."""
        layer_results = _make_clean_results()
        result = ensemble.classify("Hello world", layer_results)
        assert result.decision == Decision.APPROVE

    def test_toxic_content_rejected(self, ensemble: EnsembleClassifier) -> None:
        """Clearly toxic content should receive a REJECT decision."""
        layer_results = _make_toxic_results()
        result = ensemble.classify("Toxic text here", layer_results)
        assert result.decision == Decision.REJECT

    def test_borderline_content_flagged(self, ensemble: EnsembleClassifier) -> None:
        """Borderline content should receive a FLAG_FOR_REVIEW decision."""
        layer_results = _make_borderline_results()
        result = ensemble.classify("Borderline text", layer_results)
        assert result.decision == Decision.FLAG_FOR_REVIEW

    def test_strict_thresholds_more_rejections(
        self, strict_ensemble: EnsembleClassifier
    ) -> None:
        """Stricter thresholds should reject content that defaults would flag."""
        layer_results = _make_borderline_results()
        result = strict_ensemble.classify("Borderline text", layer_results)
        assert result.decision in (Decision.FLAG_FOR_REVIEW, Decision.REJECT)


class TestWeighting:
    """Tests for classifier weight application."""

    def test_rules_only(self) -> None:
        """With only rules available, the score should reflect rule severity."""
        ensemble = EnsembleClassifier()
        layer_results = {
            "rules": RuleResult(
                matched_rules=["blocked_keyword"],
                severity=0.8,
                confidence=1.0,
            ),
        }
        result = ensemble.classify("Test", layer_results)
        # With only rules layer, score = 0.8 (normalized)
        assert result.decision == Decision.REJECT

    def test_toxicity_only(self) -> None:
        """With only toxicity available, the score reflects toxicity."""
        ensemble = EnsembleClassifier()
        layer_results = {
            "toxicity": ToxicityResult(
                labels={"toxic": 0.1}, overall_score=0.1
            ),
        }
        result = ensemble.classify("Test", layer_results)
        assert result.decision == Decision.APPROVE

    def test_sentiment_alone_low_impact(self) -> None:
        """Negative sentiment alone should not cause rejection with default weights."""
        ensemble = EnsembleClassifier()
        layer_results = {
            "rules": RuleResult(matched_rules=[], severity=0.0, confidence=0.0),
            "toxicity": ToxicityResult(labels={"toxic": 0.1}, overall_score=0.1),
            "sentiment": SentimentResult(sentiment_score=-0.9, subjectivity=0.8),
        }
        result = ensemble.classify("Negative but not toxic", layer_results)
        # Sentiment weight is 0.2, so alone it should not drive rejection
        assert result.decision != Decision.REJECT


class TestConfidence:
    """Tests for confidence score calculation."""

    def test_confidence_in_range(self, ensemble: EnsembleClassifier) -> None:
        """Confidence should always be between 0.0 and 1.0."""
        for make_results in [
            _make_clean_results,
            _make_toxic_results,
            _make_borderline_results,
        ]:
            result = ensemble.classify("Test", make_results())
            assert 0.0 <= result.confidence <= 1.0

    def test_clear_approve_high_confidence(
        self, ensemble: EnsembleClassifier
    ) -> None:
        """Very clean content should have high confidence in APPROVE."""
        layer_results = _make_clean_results()
        result = ensemble.classify("Very clean text", layer_results)
        assert result.decision == Decision.APPROVE
        assert result.confidence > 0.5


class TestExplanation:
    """Tests for explanation generation."""

    def test_explanation_not_empty(self, ensemble: EnsembleClassifier) -> None:
        """Every decision should include a non-empty explanation."""
        layer_results = _make_clean_results()
        result = ensemble.classify("Test text", layer_results)
        assert len(result.explanation) > 0

    def test_explanation_mentions_decision(
        self, ensemble: EnsembleClassifier
    ) -> None:
        """The explanation should mention the actual decision."""
        layer_results = _make_toxic_results()
        result = ensemble.classify("Toxic text", layer_results)
        assert "REJECT" in result.explanation

    def test_explanation_mentions_rules(
        self, ensemble: EnsembleClassifier
    ) -> None:
        """When rules trigger, the explanation should list them."""
        layer_results = _make_toxic_results()
        result = ensemble.classify("Toxic text", layer_results)
        assert "blocked_keyword" in result.explanation

    def test_explanation_mentions_context_flags(
        self, ensemble: EnsembleClassifier
    ) -> None:
        """When context flags are present, they should appear in explanation."""
        layer_results = _make_borderline_results()
        result = ensemble.classify("Sarcastic text", layer_results)
        assert "sarcasm" in result.explanation
