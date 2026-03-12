"""Configuration for the ModGuard pipeline.

Centralizes all tunable parameters including model names, thresholds,
weights, and server settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClassifierWeights:
    """Weights for the ensemble classifier.

    Attributes:
        rules: Weight for rule-based classifier output.
        toxicity: Weight for toxicity classifier output.
        sentiment: Weight for sentiment classifier output.
    """

    rules: float = 0.4
    toxicity: float = 0.4
    sentiment: float = 0.2

    def validate(self) -> None:
        """Validate that weights sum to 1.0 (within tolerance)."""
        total = self.rules + self.toxicity + self.sentiment
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Classifier weights must sum to 1.0, got {total:.2f}"
            )


@dataclass
class ThresholdConfig:
    """Decision thresholds for the ensemble classifier.

    Scores below approve_below result in APPROVE.
    Scores above reject_above result in REJECT.
    Scores in between result in FLAG_FOR_REVIEW.

    Attributes:
        approve_below: Threshold below which content is approved.
        reject_above: Threshold above which content is rejected.
    """

    approve_below: float = 0.3
    reject_above: float = 0.7


@dataclass
class ModelConfig:
    """ML model configuration.

    Attributes:
        toxicity_model: HuggingFace model identifier for toxicity classification.
        sentiment_model: HuggingFace model identifier for sentiment analysis.
        device: Device for model inference ('cpu', 'cuda', or 'auto').
    """

    toxicity_model: str = "martin-ha/toxic-comment-model"
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    device: str = "cpu"


@dataclass
class ServerConfig:
    """API server configuration.

    Attributes:
        host: Host address to bind the server to.
        port: Port number for the server.
        history_max_size: Maximum number of moderation results to keep in memory.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    history_max_size: int = 1000


@dataclass
class RulesConfig:
    """Configuration for the rule-based classifier.

    Attributes:
        blocked_keywords: List of keywords to block.
        spam_caps_ratio: Ratio of uppercase characters that triggers spam detection.
        spam_punctuation_ratio: Ratio of punctuation that triggers spam detection.
        max_url_count: Maximum allowed URLs in a single message.
        custom_patterns: Additional regex patterns to match. Each entry maps
            a rule name to a regex pattern string.
    """

    blocked_keywords: list[str] = field(default_factory=lambda: [
        "kill", "murder", "attack", "bomb", "terrorist",
        "hate", "slur", "die",
    ])
    spam_caps_ratio: float = 0.7
    spam_punctuation_ratio: float = 0.3
    max_url_count: int = 3
    custom_patterns: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level configuration for the moderation pipeline.

    Attributes:
        weights: Ensemble classifier weights.
        thresholds: Decision thresholds.
        models: ML model configuration.
        server: API server configuration.
        rules: Rule-based classifier configuration.
        enable_toxicity: Whether to enable the toxicity classifier.
        enable_sentiment: Whether to enable the sentiment classifier.
    """

    weights: ClassifierWeights = field(default_factory=ClassifierWeights)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    enable_toxicity: bool = True
    enable_sentiment: bool = True
