"""Basic moderation example.

Demonstrates the simplest usage of the ModGuard pipeline: create a
pipeline, moderate a few texts, and inspect the results.
"""

from modguard import ModerationPipeline
from modguard.config import PipelineConfig


def main() -> None:
    """Run basic moderation on a few sample texts."""
    # Create a pipeline (disable ML models for quick demo)
    config = PipelineConfig(enable_toxicity=False, enable_sentiment=False)
    pipeline = ModerationPipeline(config)

    # Sample texts to moderate
    texts = [
        "Hello! This is a friendly message.",
        "I hate everything about this terrible product!",
        "BUY NOW!!! FREE OFFER!!! CLICK HERE!!! LIMITED TIME!!!",
        "The weather is nice today. Perfect for a walk.",
        "You are the worst person I have ever met online.",
    ]

    print("ModGuard Basic Moderation Example")
    print("=" * 50)

    for text in texts:
        result = pipeline.moderate(text)
        print(f"\nText: {text[:60]}...")
        print(f"  Decision:   {result.decision.value}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Time:       {result.processing_time_ms:.1f}ms")
        print(f"  Explanation: {result.explanation}")


if __name__ == "__main__":
    main()
