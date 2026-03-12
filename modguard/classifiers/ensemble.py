"""Ensemble classifier that combines layer results into a final decision.

Applies configurable weights to each classifier layer's output and maps
the weighted score to a moderation decision using threshold boundaries.
Generates a human-readable explanation summarizing which layers contributed
to the final decision and why.
"""

from __future__ import annotations

from typing import Any, Optional

from modguard.config import ClassifierWeights, ThresholdConfig
from modguard.models import (
    Decision,
    ModerationResult,
    RuleResult,
    SentimentResult,
    ToxicityResult,
)


class EnsembleClassifier:
    """Weighted ensemble that produces a final moderation decision.

    Combines scores from rule-based, toxicity, and sentiment classifiers
    using configurable weights. The weighted score is then compared against
    thresholds to produce APPROVE, FLAG_FOR_REVIEW, or REJECT.

    Attributes:
        weights: Weight configuration for each classifier layer.
        thresholds: Decision boundary thresholds.
    """

    def __init__(
        self,
        weights: Optional[ClassifierWeights] = None,
        thresholds: Optional[ThresholdConfig] = None,
    ) -> None:
        """Initialize the ensemble classifier.

        Args:
            weights: Classifier weights. Uses defaults if not provided.
            thresholds: Decision thresholds. Uses defaults if not provided.
        """
        self.weights = weights or ClassifierWeights()
        self.thresholds = thresholds or ThresholdConfig()

    def _score_rules(self, result: RuleResult) -> float:
        """Extract a normalized score from rule-based results.

        Args:
            result: The rule-based classifier result.

        Returns:
            Score from 0.0 to 1.0 based on severity.
        """
        if not result.matched_rules:
            return 0.0
        return result.severity

    def _score_toxicity(self, result: ToxicityResult) -> float:
        """Extract a normalized score from toxicity results.

        Args:
            result: The toxicity classifier result.

        Returns:
            The overall toxicity score from 0.0 to 1.0.
        """
        return result.overall_score

    def _score_sentiment(self, result: SentimentResult) -> float:
        """Convert sentiment into a moderation-relevant score.

        Very negative sentiment increases the moderation score. Context
        flags like sarcasm also contribute a penalty.

        Args:
            result: The sentiment classifier result.

        Returns:
            Score from 0.0 to 1.0 where higher means more concerning.
        """
        # Convert sentiment_score (-1 to 1) to risk (0 to 1)
        # Very negative sentiment = high risk
        base_score = max(0.0, -result.sentiment_score)

        # Context flags add a penalty
        flag_penalty = len(result.context_flags) * 0.15

        return min(1.0, base_score + flag_penalty)

    def _generate_explanation(
        self,
        decision: Decision,
        weighted_score: float,
        layer_results: dict[str, Any],
        layer_scores: dict[str, float],
    ) -> str:
        """Generate a human-readable explanation of the decision.

        Args:
            decision: The final moderation decision.
            weighted_score: The combined weighted score.
            layer_results: Raw results from each classifier layer.
            layer_scores: Normalized scores from each layer.

        Returns:
            An explanation string describing what drove the decision.
        """
        parts: list[str] = []
        parts.append(
            f"Decision: {decision.value} (score: {weighted_score:.3f})"
        )

        # Rules explanation
        if "rules" in layer_results:
            rule_result: RuleResult = layer_results["rules"]
            if rule_result.matched_rules:
                rules_str = ", ".join(rule_result.matched_rules)
                parts.append(f"Rules triggered: {rules_str}")
            else:
                parts.append("No rule violations detected.")

        # Toxicity explanation
        if "toxicity" in layer_results:
            tox_result: ToxicityResult = layer_results["toxicity"]
            if tox_result.overall_score > 0.3:
                high_labels = [
                    f"{k}: {v:.2f}"
                    for k, v in tox_result.labels.items()
                    if v > 0.3
                ]
                if high_labels:
                    parts.append(
                        f"Toxicity signals: {', '.join(high_labels)}"
                    )
            else:
                parts.append("Toxicity levels within acceptable range.")

        # Sentiment explanation
        if "sentiment" in layer_results:
            sent_result: SentimentResult = layer_results["sentiment"]
            if sent_result.sentiment_score < -0.5:
                parts.append(
                    f"Strongly negative sentiment detected "
                    f"(score: {sent_result.sentiment_score:.2f})."
                )
            if sent_result.context_flags:
                flags_str = ", ".join(sent_result.context_flags)
                parts.append(f"Context flags: {flags_str}")

        return " | ".join(parts)

    def classify(
        self, text: str, layer_results: dict[str, Any]
    ) -> ModerationResult:
        """Combine all layer results into a final moderation decision.

        Computes a weighted score from each available layer, compares
        it against decision thresholds, and generates an explanation.

        Args:
            text: The original text (stored in the result).
            layer_results: Dictionary mapping layer names to their
                typed result objects.

        Returns:
            A ModerationResult with the final decision, confidence,
            and explanation.
        """
        layer_scores: dict[str, float] = {}
        weighted_score = 0.0
        total_weight = 0.0

        # Score each available layer
        if "rules" in layer_results:
            score = self._score_rules(layer_results["rules"])
            layer_scores["rules"] = score
            weighted_score += score * self.weights.rules
            total_weight += self.weights.rules

        if "toxicity" in layer_results:
            score = self._score_toxicity(layer_results["toxicity"])
            layer_scores["toxicity"] = score
            weighted_score += score * self.weights.toxicity
            total_weight += self.weights.toxicity

        if "sentiment" in layer_results:
            score = self._score_sentiment(layer_results["sentiment"])
            layer_scores["sentiment"] = score
            weighted_score += score * self.weights.sentiment
            total_weight += self.weights.sentiment

        # Normalize if not all layers are present
        if total_weight > 0 and total_weight < 1.0:
            weighted_score = weighted_score / total_weight

        # Determine decision based on thresholds
        if weighted_score >= self.thresholds.reject_above:
            decision = Decision.REJECT
        elif weighted_score >= self.thresholds.approve_below:
            decision = Decision.FLAG_FOR_REVIEW
        else:
            decision = Decision.APPROVE

        # Confidence reflects how far the score is from the nearest threshold
        if decision == Decision.APPROVE:
            confidence = 1.0 - (weighted_score / self.thresholds.approve_below)
        elif decision == Decision.REJECT:
            confidence = (weighted_score - self.thresholds.reject_above) / (
                1.0 - self.thresholds.reject_above
            )
        else:
            # For FLAG_FOR_REVIEW, confidence is distance to the nearest boundary
            mid = (self.thresholds.approve_below + self.thresholds.reject_above) / 2
            range_half = (self.thresholds.reject_above - self.thresholds.approve_below) / 2
            confidence = 1.0 - abs(weighted_score - mid) / range_half

        confidence = max(0.0, min(1.0, confidence))

        explanation = self._generate_explanation(
            decision, weighted_score, layer_results, layer_scores
        )

        return ModerationResult(
            text=text,
            decision=decision,
            confidence=confidence,
            layer_results=layer_results,
            explanation=explanation,
        )
