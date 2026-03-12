"""Batch processing example.

Demonstrates how to moderate multiple texts concurrently using the
async batch API. Loads sample data from the demo module and processes
all items in a single call.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

from modguard import ModerationPipeline
from modguard.config import PipelineConfig


async def main() -> None:
    """Load sample data and moderate all items in batch."""
    # Create pipeline in rules-only mode for speed
    config = PipelineConfig(enable_toxicity=False, enable_sentiment=False)
    pipeline = ModerationPipeline(config)

    # Load sample data
    sample_path = Path(__file__).parent.parent / "modguard" / "demo" / "sample_content.json"
    with open(sample_path) as f:
        samples = json.load(f)

    texts = [s["text"] for s in samples]
    categories = [s["category"] for s in samples]

    print("ModGuard Batch Processing Example")
    print("=" * 50)
    print(f"Processing {len(texts)} items...\n")

    # Run batch moderation
    results = await pipeline.moderate_batch(texts)

    # Summarize results
    decision_counts: Counter = Counter()
    category_decisions: dict[str, Counter] = {}

    for result, category in zip(results, categories):
        decision = result.decision.value
        decision_counts[decision] += 1

        if category not in category_decisions:
            category_decisions[category] = Counter()
        category_decisions[category][decision] += 1

    # Print summary
    print("Overall Decision Distribution:")
    for decision, count in decision_counts.most_common():
        pct = count / len(results) * 100
        print(f"  {decision:20s}: {count:3d} ({pct:.1f}%)")

    print("\nDecisions by Content Category:")
    for category in sorted(category_decisions):
        counts = category_decisions[category]
        total = sum(counts.values())
        print(f"\n  {category} ({total} items):")
        for decision, count in counts.most_common():
            print(f"    {decision:20s}: {count}")

    # Show total processing time
    total_time = sum(r.processing_time_ms for r in results)
    avg_time = total_time / len(results) if results else 0
    print(f"\nTotal processing time: {total_time:.1f}ms")
    print(f"Average per item:      {avg_time:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
