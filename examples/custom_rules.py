"""Custom rules example.

Demonstrates how to extend the pipeline with custom regex patterns
and modified thresholds. Shows how to add domain-specific rules
beyond the built-in keyword and spam detection.
"""

from modguard import ModerationPipeline
from modguard.config import (
    PipelineConfig,
    RulesConfig,
    ThresholdConfig,
)


def main() -> None:
    """Run moderation with custom rules and thresholds."""
    # Define custom rules for a hypothetical e-commerce platform
    rules_config = RulesConfig(
        # Add platform-specific blocked keywords
        blocked_keywords=[
            "scam", "fraud", "fake", "counterfeit",
            "hate", "kill", "attack",
        ],
        # Tighten spam detection
        spam_caps_ratio=0.6,
        max_url_count=2,
        # Add custom regex patterns
        custom_patterns={
            # Detect phone numbers (potential doxxing)
            "phone_number": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            # Detect competitor mentions (for brand-safety)
            "competitor_mention": r"\b(competitor_brand_a|competitor_brand_b)\b",
            # Detect price manipulation language
            "price_manipulation": r"\b(price\s+match|beat\s+any\s+price|lowest\s+price)\b",
        },
    )

    # Use stricter thresholds
    thresholds = ThresholdConfig(
        approve_below=0.2,
        reject_above=0.5,
    )

    config = PipelineConfig(
        rules=rules_config,
        thresholds=thresholds,
        enable_toxicity=False,
        enable_sentiment=False,
    )

    pipeline = ModerationPipeline(config)

    # Test with various inputs
    test_cases = [
        "Great product! Highly recommended.",
        "This is a total scam, do not buy!",
        "Call me at 555-123-4567 for a deal.",
        "competitor_brand_a has better prices than you.",
        "We will beat any price! Lowest price guaranteed!",
        "AMAZING DEAL CHECK OUT http://a.com http://b.com http://c.com",
        "Normal product review. Works as described.",
    ]

    print("ModGuard Custom Rules Example")
    print("=" * 50)
    print("Custom patterns: phone_number, competitor_mention, price_manipulation")
    print(f"Thresholds: approve < {thresholds.approve_below}, "
          f"reject > {thresholds.reject_above}")
    print()

    for text in test_cases:
        result = pipeline.moderate(text)
        rules = result.layer_results.get("rules")
        matched = rules.matched_rules if rules else []

        print(f"Text: {text[:65]}")
        print(f"  Decision: {result.decision.value}")
        if matched:
            print(f"  Rules:    {', '.join(matched)}")
        print()


if __name__ == "__main__":
    main()
