"""Tests for the rule-based classifier.

Verifies that known inputs trigger the correct rules with expected
severity scores and confidence values.
"""

from __future__ import annotations

import pytest

from modguard.classifiers.rules import RuleBasedClassifier
from modguard.config import RulesConfig


@pytest.fixture
def classifier() -> RuleBasedClassifier:
    """Create a rule-based classifier with default configuration."""
    return RuleBasedClassifier()


@pytest.fixture
def custom_classifier() -> RuleBasedClassifier:
    """Create a rule-based classifier with custom patterns."""
    config = RulesConfig(
        custom_patterns={"credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"}
    )
    return RuleBasedClassifier(config)


class TestBlockedKeywords:
    """Tests for blocked keyword detection."""

    def test_detects_blocked_keyword(self, classifier: RuleBasedClassifier) -> None:
        """A message containing a blocked keyword should be flagged."""
        result = classifier.classify("I hate everything about this.")
        assert "blocked_keyword" in result.matched_rules
        assert result.severity >= 0.8
        assert result.confidence == 1.0

    def test_case_insensitive_keyword(self, classifier: RuleBasedClassifier) -> None:
        """Blocked keywords should be matched case-insensitively."""
        result = classifier.classify("HATE is a strong word.")
        assert "blocked_keyword" in result.matched_rules

    def test_clean_text_no_keywords(self, classifier: RuleBasedClassifier) -> None:
        """Clean text should not trigger any keyword rules."""
        result = classifier.classify("This is a perfectly normal message.")
        assert "blocked_keyword" not in result.matched_rules

    def test_reports_which_keywords(self, classifier: RuleBasedClassifier) -> None:
        """The result details should list which keywords were found."""
        result = classifier.classify("I hate this and will attack the problem.")
        assert "blocked_keywords_found" in result.details
        found = result.details["blocked_keywords_found"]
        assert "hate" in found
        assert "attack" in found


class TestSpamDetection:
    """Tests for spam heuristic detection."""

    def test_excessive_caps(self, classifier: RuleBasedClassifier) -> None:
        """Text with too many uppercase letters should trigger caps rule."""
        result = classifier.classify("THIS IS ALL CAPS AND VERY SHOUTY TEXT")
        assert "excessive_caps" in result.matched_rules

    def test_normal_caps_ratio(self, classifier: RuleBasedClassifier) -> None:
        """Text with normal capitalization should not trigger caps rule."""
        result = classifier.classify("This is a normal sentence with Some Caps.")
        assert "excessive_caps" not in result.matched_rules

    def test_excessive_punctuation(self, classifier: RuleBasedClassifier) -> None:
        """Runs of 3+ punctuation marks should trigger the punctuation rule."""
        result = classifier.classify("What do you mean!!!! Are you serious???")
        assert "excessive_punctuation" in result.matched_rules

    def test_normal_punctuation(self, classifier: RuleBasedClassifier) -> None:
        """Standard punctuation should not trigger the rule."""
        result = classifier.classify("Hello! How are you? I am fine.")
        assert "excessive_punctuation" not in result.matched_rules

    def test_repeated_characters(self, classifier: RuleBasedClassifier) -> None:
        """Five or more repeated characters should trigger the rule."""
        result = classifier.classify("Noooooo way that happened!")
        assert "repeated_characters" in result.matched_rules

    def test_url_flooding(self, classifier: RuleBasedClassifier) -> None:
        """More than max_url_count URLs should trigger url_flooding."""
        text = (
            "Check http://a.com http://b.com http://c.com http://d.com"
        )
        result = classifier.classify(text)
        assert "url_flooding" in result.matched_rules
        assert result.details.get("url_count", 0) > 3

    def test_spam_phrases(self, classifier: RuleBasedClassifier) -> None:
        """Multiple spam phrases should trigger the spam_phrases rule."""
        result = classifier.classify(
            "Buy now! Free offer! Click here for a limited time!"
        )
        assert "spam_phrases" in result.matched_rules


class TestCustomPatterns:
    """Tests for custom regex patterns."""

    def test_custom_pattern_match(
        self, custom_classifier: RuleBasedClassifier
    ) -> None:
        """A custom pattern should match and produce a custom: rule."""
        result = custom_classifier.classify("My card is 1234-5678-9012-3456")
        custom_rules = [r for r in result.matched_rules if r.startswith("custom:")]
        assert len(custom_rules) > 0
        assert "custom:credit_card" in custom_rules

    def test_custom_pattern_no_match(
        self, custom_classifier: RuleBasedClassifier
    ) -> None:
        """Text without the custom pattern should not trigger it."""
        result = custom_classifier.classify("No card numbers here.")
        custom_rules = [r for r in result.matched_rules if r.startswith("custom:")]
        assert len(custom_rules) == 0


class TestCleanInput:
    """Tests confirming clean text passes without any rule triggers."""

    def test_clean_greeting(self, classifier: RuleBasedClassifier) -> None:
        """A friendly greeting should not trigger any rules."""
        result = classifier.classify("Hello, how are you doing today?")
        assert len(result.matched_rules) == 0
        assert result.severity == 0.0
        assert result.confidence == 0.0

    def test_clean_question(self, classifier: RuleBasedClassifier) -> None:
        """A normal question should pass cleanly."""
        result = classifier.classify("What time does the store close?")
        assert len(result.matched_rules) == 0

    def test_empty_string(self, classifier: RuleBasedClassifier) -> None:
        """Empty input should produce no matches."""
        result = classifier.classify("")
        assert len(result.matched_rules) == 0
        assert result.severity == 0.0
