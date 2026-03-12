"""End-to-end tests for the moderation pipeline.

Uses mocked ML models to test the full pipeline flow without requiring
model downloads or GPU access. Verifies that text flows through all
layers and produces valid ModerationResult objects.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from modguard.config import PipelineConfig
from modguard.models import Decision, ModerationResult, SentimentResult, ToxicityResult
from modguard.pipeline import ModerationPipeline


def _mock_toxicity_classify(text: str) -> ToxicityResult:
    """Produce a mock toxicity result based on text content."""
    if "toxic" in text.lower() or "hate" in text.lower():
        return ToxicityResult(
            labels={"toxic": 0.9, "insult": 0.7},
            overall_score=0.9,
        )
    return ToxicityResult(
        labels={"toxic": 0.05},
        overall_score=0.05,
    )


def _mock_sentiment_classify(text: str) -> SentimentResult:
    """Produce a mock sentiment result based on text content."""
    if "hate" in text.lower() or "terrible" in text.lower():
        return SentimentResult(
            sentiment_score=-0.8,
            subjectivity=0.6,
            raw_label="NEGATIVE",
            raw_score=0.9,
        )
    return SentimentResult(
        sentiment_score=0.7,
        subjectivity=0.3,
        raw_label="POSITIVE",
        raw_score=0.85,
    )


@pytest.fixture
def pipeline() -> ModerationPipeline:
    """Create a pipeline with mocked ML classifiers."""
    config = PipelineConfig()
    p = ModerationPipeline(config)

    # Mock the ML classifiers
    if p._toxicity_classifier is not None:
        p._toxicity_classifier.classify = MagicMock(
            side_effect=_mock_toxicity_classify
        )
    if p._sentiment_classifier is not None:
        p._sentiment_classifier.classify = MagicMock(
            side_effect=_mock_sentiment_classify
        )

    return p


@pytest.fixture
def rules_only_pipeline() -> ModerationPipeline:
    """Create a pipeline with only rule-based classification."""
    config = PipelineConfig(enable_toxicity=False, enable_sentiment=False)
    return ModerationPipeline(config)


class TestPipelineModerate:
    """Tests for the synchronous moderate method."""

    def test_returns_moderation_result(
        self, pipeline: ModerationPipeline
    ) -> None:
        """moderate() should return a ModerationResult instance."""
        result = pipeline.moderate("Hello, world!")
        assert isinstance(result, ModerationResult)

    def test_clean_text_approved(self, pipeline: ModerationPipeline) -> None:
        """Clean text should be approved."""
        result = pipeline.moderate("This is a perfectly normal message.")
        assert result.decision == Decision.APPROVE

    def test_toxic_text_rejected(self, pipeline: ModerationPipeline) -> None:
        """Text containing toxicity indicators should be rejected."""
        result = pipeline.moderate("I hate you, toxic person!")
        assert result.decision in (Decision.REJECT, Decision.FLAG_FOR_REVIEW)

    def test_includes_processing_time(
        self, pipeline: ModerationPipeline
    ) -> None:
        """The result should include a positive processing time."""
        result = pipeline.moderate("Any text here.")
        assert result.processing_time_ms > 0

    def test_includes_text(self, pipeline: ModerationPipeline) -> None:
        """The result should store the original input text."""
        text = "Remember this exact text."
        result = pipeline.moderate(text)
        assert result.text == text

    def test_includes_explanation(self, pipeline: ModerationPipeline) -> None:
        """The result should include a non-empty explanation."""
        result = pipeline.moderate("Some text to moderate.")
        assert len(result.explanation) > 0

    def test_layer_results_present(
        self, pipeline: ModerationPipeline
    ) -> None:
        """The result should contain layer results for all enabled layers."""
        result = pipeline.moderate("Test text.")
        assert "rules" in result.layer_results
        assert "toxicity" in result.layer_results
        assert "sentiment" in result.layer_results

    def test_result_serializable(self, pipeline: ModerationPipeline) -> None:
        """The result should be serializable to a dictionary."""
        result = pipeline.moderate("Test text.")
        result_dict = result.to_dict()
        assert "decision" in result_dict
        assert "confidence" in result_dict
        assert "layer_results" in result_dict
        assert "processing_time_ms" in result_dict

    def test_has_unique_id(self, pipeline: ModerationPipeline) -> None:
        """Each result should have a unique ID."""
        r1 = pipeline.moderate("First text.")
        r2 = pipeline.moderate("Second text.")
        assert r1.id != r2.id


class TestPipelineRulesOnly:
    """Tests for rules-only pipeline mode."""

    def test_rules_only_no_ml_layers(
        self, rules_only_pipeline: ModerationPipeline
    ) -> None:
        """Rules-only pipeline should not include ML layer results."""
        result = rules_only_pipeline.moderate("Hello!")
        assert "rules" in result.layer_results
        assert "toxicity" not in result.layer_results
        assert "sentiment" not in result.layer_results

    def test_rules_only_catches_keywords(
        self, rules_only_pipeline: ModerationPipeline
    ) -> None:
        """Rules-only pipeline should still catch blocked keywords."""
        result = rules_only_pipeline.moderate("I hate this product.")
        assert result.decision != Decision.APPROVE

    def test_rules_only_approves_clean(
        self, rules_only_pipeline: ModerationPipeline
    ) -> None:
        """Rules-only pipeline should approve clean text."""
        result = rules_only_pipeline.moderate("Great weather today!")
        assert result.decision == Decision.APPROVE


class TestPipelineBatch:
    """Tests for batch moderation."""

    def test_batch_returns_correct_count(
        self, pipeline: ModerationPipeline
    ) -> None:
        """Batch moderation should return one result per input."""
        texts = ["Hello!", "This is fine.", "Good message."]
        results = asyncio.get_event_loop().run_until_complete(
            pipeline.moderate_batch(texts)
        )
        assert len(results) == len(texts)

    def test_batch_all_have_ids(
        self, pipeline: ModerationPipeline
    ) -> None:
        """Every batch result should have a unique ID."""
        texts = ["Text one.", "Text two.", "Text three."]
        results = asyncio.get_event_loop().run_until_complete(
            pipeline.moderate_batch(texts)
        )
        ids = [r.id for r in results]
        assert len(set(ids)) == len(ids)

    def test_batch_empty_list(
        self, pipeline: ModerationPipeline
    ) -> None:
        """Batch moderation with empty input should return empty list."""
        results = asyncio.get_event_loop().run_until_complete(
            pipeline.moderate_batch([])
        )
        assert results == []
