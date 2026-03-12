"""Tests for the FastAPI endpoints.

Verifies that API endpoints return correct status codes and response
structures. Uses mocked ML models for offline testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from modguard.api.server import create_app
from modguard.config import PipelineConfig
from modguard.models import SentimentResult, ToxicityResult


def _mock_toxicity_classify(text: str) -> ToxicityResult:
    """Produce a mock toxicity result."""
    return ToxicityResult(labels={"toxic": 0.05}, overall_score=0.05)


def _mock_sentiment_classify(text: str) -> SentimentResult:
    """Produce a mock sentiment result."""
    return SentimentResult(
        sentiment_score=0.7,
        subjectivity=0.3,
        raw_label="POSITIVE",
        raw_score=0.85,
    )


@pytest.fixture
def client() -> TestClient:
    """Create a test client with mocked ML classifiers."""
    config = PipelineConfig()
    app = create_app(config)

    # Mock ML classifiers on the pipeline stored in app state
    pipeline = app.state.pipeline
    if pipeline._toxicity_classifier is not None:
        pipeline._toxicity_classifier.classify = MagicMock(
            side_effect=_mock_toxicity_classify
        )
    if pipeline._sentiment_classifier is not None:
        pipeline._sentiment_classifier.classify = MagicMock(
            side_effect=_mock_sentiment_classify
        )

    return TestClient(app)


class TestModerateEndpoint:
    """Tests for POST /moderate."""

    def test_moderate_returns_200(self, client: TestClient) -> None:
        """Successful moderation should return HTTP 200."""
        response = client.post(
            "/moderate",
            json={"text": "Hello, world!"},
        )
        assert response.status_code == 200

    def test_moderate_response_structure(self, client: TestClient) -> None:
        """Response should contain all required fields."""
        response = client.post(
            "/moderate",
            json={"text": "Test message."},
        )
        data = response.json()
        assert "decision" in data
        assert "confidence" in data
        assert "layer_results" in data
        assert "processing_time_ms" in data
        assert "explanation" in data
        assert "id" in data

    def test_moderate_decision_values(self, client: TestClient) -> None:
        """Decision should be one of the valid enum values."""
        response = client.post(
            "/moderate",
            json={"text": "Some text."},
        )
        data = response.json()
        assert data["decision"] in ("APPROVE", "FLAG_FOR_REVIEW", "REJECT")

    def test_moderate_missing_text(self, client: TestClient) -> None:
        """Missing text field should return HTTP 422."""
        response = client.post("/moderate", json={})
        assert response.status_code == 422

    def test_moderate_empty_body(self, client: TestClient) -> None:
        """Empty request body should return HTTP 422."""
        response = client.post(
            "/moderate",
            content="",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


class TestBatchEndpoint:
    """Tests for POST /moderate/batch."""

    def test_batch_returns_200(self, client: TestClient) -> None:
        """Successful batch moderation should return HTTP 200."""
        response = client.post(
            "/moderate/batch",
            json={"texts": ["Hello!", "World!"]},
        )
        assert response.status_code == 200

    def test_batch_response_structure(self, client: TestClient) -> None:
        """Response should contain a results list."""
        response = client.post(
            "/moderate/batch",
            json={"texts": ["Text one.", "Text two."]},
        )
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

    def test_batch_each_result_has_id(self, client: TestClient) -> None:
        """Each result in the batch should have a unique ID."""
        response = client.post(
            "/moderate/batch",
            json={"texts": ["A", "B", "C"]},
        )
        data = response.json()
        ids = [r["id"] for r in data["results"]]
        assert len(set(ids)) == 3

    def test_batch_empty_list(self, client: TestClient) -> None:
        """An empty texts list should return 200 with empty results."""
        response = client.post(
            "/moderate/batch",
            json={"texts": []},
        )
        assert response.status_code == 200
        assert response.json()["results"] == []


class TestStatsEndpoint:
    """Tests for GET /stats."""

    def test_stats_returns_200(self, client: TestClient) -> None:
        """Stats endpoint should return HTTP 200."""
        response = client.get("/stats")
        assert response.status_code == 200

    def test_stats_initial_values(self, client: TestClient) -> None:
        """Initial stats should show zero processed."""
        response = client.get("/stats")
        data = response.json()
        assert data["total_processed"] == 0

    def test_stats_updates_after_moderation(self, client: TestClient) -> None:
        """Stats should update after processing a moderation request."""
        client.post("/moderate", json={"text": "Hello!"})
        response = client.get("/stats")
        data = response.json()
        assert data["total_processed"] == 1


class TestHistoryEndpoint:
    """Tests for GET /history."""

    def test_history_returns_200(self, client: TestClient) -> None:
        """History endpoint should return HTTP 200."""
        response = client.get("/history")
        assert response.status_code == 200

    def test_history_pagination(self, client: TestClient) -> None:
        """History should support pagination parameters."""
        response = client.get("/history?page=1&page_size=10")
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data

    def test_history_populates_after_moderation(
        self, client: TestClient
    ) -> None:
        """History should contain items after processing requests."""
        client.post("/moderate", json={"text": "First."})
        client.post("/moderate", json={"text": "Second."})
        response = client.get("/history")
        data = response.json()
        assert data["total"] == 2

    def test_history_filter_by_decision(self, client: TestClient) -> None:
        """History should support filtering by decision type."""
        client.post("/moderate", json={"text": "Clean message."})
        response = client.get("/history?decision=APPROVE")
        assert response.status_code == 200


class TestDashboardEndpoint:
    """Tests for GET /dashboard."""

    def test_dashboard_returns_200(self, client: TestClient) -> None:
        """Dashboard endpoint should return HTML with HTTP 200."""
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_dashboard_contains_modguard(self, client: TestClient) -> None:
        """Dashboard HTML should contain the ModGuard title."""
        response = client.get("/dashboard")
        assert "ModGuard" in response.text
