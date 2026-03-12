"""ModGuard: A multi-layered AI content moderation pipeline.

ModGuard combines rule-based filters, toxicity classification, and sentiment
analysis into a unified pipeline that produces actionable moderation decisions
with confidence scores and detailed explanations.
"""

__version__ = "0.1.0"

from modguard.models import (
    ModerationResult,
    RuleResult,
    SentimentResult,
    ToxicityResult,
)
from modguard.pipeline import ModerationPipeline

__all__ = [
    "ModerationPipeline",
    "ModerationResult",
    "RuleResult",
    "SentimentResult",
    "ToxicityResult",
]
