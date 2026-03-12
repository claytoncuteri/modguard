# ModGuard

A multi-layered AI content moderation pipeline with real-time dashboard.

ModGuard combines rule-based filters, transformer-based toxicity classification, and sentiment analysis into a unified pipeline. Each piece of content flows through multiple independent classifier layers, and an ensemble step produces a final decision with a confidence score and detailed explanation.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the server (rules-only mode for fast startup)
python -m modguard serve --no-ml

# Run the server with all ML models
python -m modguard serve

# Open the dashboard
open http://localhost:8000/dashboard
```

## Usage

### Python API

```python
from modguard import ModerationPipeline
from modguard.config import PipelineConfig

# Create pipeline (disable ML models for quick testing)
config = PipelineConfig(enable_toxicity=False, enable_sentiment=False)
pipeline = ModerationPipeline(config)

result = pipeline.moderate("Hello, this is a friendly message.")
print(result.decision)      # Decision.APPROVE
print(result.confidence)    # 1.0
print(result.explanation)   # Human-readable explanation
```

### REST API

```bash
# Moderate a single item
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'

# Moderate a batch
curl -X POST http://localhost:8000/moderate/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello", "Goodbye"]}'

# Get statistics
curl http://localhost:8000/stats

# Get history (paginated)
curl "http://localhost:8000/history?page=1&page_size=20"
```

### WebSocket

Connect to `ws://localhost:8000/ws` for real-time moderation results as they are processed. You can also send text through the WebSocket for inline moderation.

## Architecture

```
Input Text
    |
    v
+---------------------------+
|   Layer 1: Rule-Based     |  Regex patterns, keyword blocklists,
|   (weight: 0.4)           |  spam heuristics (caps, punctuation, URLs)
+---------------------------+
    |
    v
+---------------------------+
|   Layer 2: Toxicity       |  HuggingFace transformer model
|   (weight: 0.4)           |  (martin-ha/toxic-comment-model)
+---------------------------+
    |
    v
+---------------------------+
|   Layer 3: Sentiment      |  DistilBERT sentiment analysis with
|   (weight: 0.2)           |  sarcasm and context flag detection
+---------------------------+
    |
    v
+---------------------------+
|   Ensemble Classifier     |  Weighted combination + threshold
|                           |  mapping to APPROVE / FLAG / REJECT
+---------------------------+
    |
    v
ModerationResult
  - decision (APPROVE | FLAG_FOR_REVIEW | REJECT)
  - confidence (0.0 to 1.0)
  - layer_results (per-layer breakdown)
  - processing_time_ms
  - explanation (human-readable)
```

## Design Decisions

### Why a Layered Pipeline?

A single classifier cannot reliably handle every type of problematic content. Keyword filters catch known bad patterns instantly but miss paraphrased abuse. ML models generalize better but can be fooled by adversarial inputs. By stacking multiple independent layers, the system achieves broader coverage and greater resilience. If one layer fails or returns an uncertain result, the other layers can compensate.

### Ensemble Weighting

The default weights (rules=0.4, toxicity=0.4, sentiment=0.2) reflect a practical balance. Rule-based and toxicity classifiers carry equal weight because they target different failure modes: rules handle deterministic patterns while toxicity models handle subtle language. Sentiment receives lower weight because negative sentiment alone is not necessarily harmful (a critical product review is negative but acceptable).

### Precision vs. Recall Tradeoff

The three-tier decision system (APPROVE, FLAG_FOR_REVIEW, REJECT) is designed to favor precision over recall for automated rejection. Only content scoring above 0.7 is auto-rejected. Content between 0.3 and 0.7 is flagged for human review, ensuring that borderline cases receive human judgment rather than automated decisions. This reduces false positives in the rejection category at the cost of more items requiring manual review.

### Confidence Scoring

Confidence reflects how far the weighted score falls from the nearest decision boundary. A score deep in APPROVE territory (near 0.0) produces high confidence. A score near a threshold boundary produces low confidence. This helps downstream systems prioritize which flagged items need the most urgent human review.

## Extending the Pipeline

### Adding a Custom Classifier

1. Create a new classifier that extends `BaseClassifier`:

```python
from modguard.classifiers.base import BaseClassifier
from dataclasses import dataclass

@dataclass
class MyResult:
    score: float = 0.0

    def to_dict(self):
        return {"score": self.score}

class MyClassifier(BaseClassifier):
    def classify(self, text: str) -> MyResult:
        # Your classification logic here
        return MyResult(score=0.5)
```

2. Add it to the pipeline by modifying `pipeline.py` to include your layer.

3. Update the ensemble weights in `config.py` to account for the new layer.

### Adding Custom Rules

Pass custom regex patterns through the configuration:

```python
from modguard.config import PipelineConfig, RulesConfig

config = PipelineConfig(
    rules=RulesConfig(
        custom_patterns={
            "phone_number": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "email_address": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
        },
        blocked_keywords=["spam", "scam", "fraud"],
    )
)
```

## Project Structure

```
modguard/
  modguard/
    __init__.py          # Package exports
    pipeline.py          # Core pipeline orchestration
    config.py            # All configuration dataclasses
    models.py            # Result data models
    __main__.py          # CLI entry point
    classifiers/
      base.py            # Abstract base classifier
      rules.py           # Regex and keyword classifier
      toxicity.py        # HuggingFace toxicity model wrapper
      sentiment.py       # Sentiment analysis with context flags
      ensemble.py        # Weighted decision combiner
    api/
      server.py          # FastAPI app factory
      routes.py          # HTTP endpoints
      websocket.py       # WebSocket handler
    dashboard/
      index.html         # Dashboard UI
      styles.css         # Dark theme styles
      app.js             # Dashboard logic and charts
    demo/
      generate_data.py   # Sample data generator
      sample_content.json
  tests/
    test_rules.py        # Rule classifier tests
    test_ensemble.py     # Ensemble weighting tests
    test_pipeline.py     # End-to-end pipeline tests
    test_api.py          # API endpoint tests
  examples/
    basic_moderation.py  # Simple usage example
    batch_processing.py  # Concurrent batch processing
    custom_rules.py      # Custom patterns and thresholds
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_rules.py
```

## Docker

```bash
# Build the image
docker build -t modguard .

# Run the container
docker run -p 8000:8000 modguard
```

## License

MIT License. See [LICENSE](LICENSE) for details.
