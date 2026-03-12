"""Core moderation pipeline that orchestrates classifier layers.

The pipeline runs text through each enabled classifier layer, then passes
all layer results to the ensemble classifier to produce a final decision.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from modguard.classifiers.ensemble import EnsembleClassifier
from modguard.classifiers.rules import RuleBasedClassifier
from modguard.classifiers.sentiment import SentimentClassifier
from modguard.classifiers.toxicity import ToxicityClassifier
from modguard.config import PipelineConfig
from modguard.models import ModerationResult


class ModerationPipeline:
    """Multi-layered content moderation pipeline.

    Orchestrates rule-based, toxicity, and sentiment classifiers through
    an ensemble that produces a unified moderation decision.

    Attributes:
        config: Pipeline configuration object.

    Example:
        >>> pipeline = ModerationPipeline()
        >>> result = pipeline.moderate("Hello, this is a friendly message.")
        >>> print(result.decision)
        Decision.APPROVE
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """Initialize the moderation pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        self._rules_classifier = RuleBasedClassifier(self.config.rules)
        self._toxicity_classifier: Optional[ToxicityClassifier] = None
        self._sentiment_classifier: Optional[SentimentClassifier] = None
        self._ensemble = EnsembleClassifier(
            self.config.weights, self.config.thresholds
        )

        if self.config.enable_toxicity:
            self._toxicity_classifier = ToxicityClassifier(self.config.models)
        if self.config.enable_sentiment:
            self._sentiment_classifier = SentimentClassifier(self.config.models)

    def moderate(self, text: str) -> ModerationResult:
        """Run text through the full moderation pipeline.

        Processes the text through each enabled classifier layer and combines
        results using the ensemble classifier.

        Args:
            text: The text content to moderate.

        Returns:
            A ModerationResult containing the decision, confidence, layer
            results, processing time, and explanation.
        """
        start_time = time.perf_counter()
        layer_results: dict = {}

        # Layer 1: Rule-based classification (always enabled)
        rule_result = self._rules_classifier.classify(text)
        layer_results["rules"] = rule_result

        # Layer 2: Toxicity classification
        if self._toxicity_classifier is not None:
            toxicity_result = self._toxicity_classifier.classify(text)
            layer_results["toxicity"] = toxicity_result

        # Layer 3: Sentiment analysis
        if self._sentiment_classifier is not None:
            sentiment_result = self._sentiment_classifier.classify(text)
            layer_results["sentiment"] = sentiment_result

        # Ensemble decision
        result = self._ensemble.classify(text, layer_results)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        result.processing_time_ms = elapsed_ms
        result.text = text

        return result

    async def moderate_async(self, text: str) -> ModerationResult:
        """Async wrapper for the moderate method.

        Args:
            text: The text content to moderate.

        Returns:
            A ModerationResult with the moderation decision.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.moderate, text)

    async def moderate_batch(self, texts: list[str]) -> list[ModerationResult]:
        """Moderate multiple texts concurrently.

        Args:
            texts: List of text strings to moderate.

        Returns:
            List of ModerationResult objects in the same order as inputs.
        """
        tasks = [self.moderate_async(text) for text in texts]
        return await asyncio.gather(*tasks)
