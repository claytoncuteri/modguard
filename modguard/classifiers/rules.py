"""Rule-based classifier using regex patterns, keyword lists, and heuristics.

This classifier provides fast, deterministic filtering for content that
matches known patterns. It catches spam indicators (excessive caps,
punctuation floods, URL stuffing), blocked keywords, and custom regex
patterns. Because rules are deterministic, all matches produce a
confidence of 1.0.
"""

from __future__ import annotations

import re
from typing import Optional

from modguard.classifiers.base import BaseClassifier
from modguard.config import RulesConfig
from modguard.models import RuleResult


# Precompiled patterns for common spam and abuse indicators
_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    re.IGNORECASE,
)
_EXCESSIVE_PUNCTUATION = re.compile(r"[!?]{3,}")
_REPEATED_CHARS = re.compile(r"(.)\1{4,}")
_PHONE_PATTERN = re.compile(
    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
)
_EMAIL_SPAM_PATTERN = re.compile(
    r"\b(free|winner|congratulations|click here|act now|limited time|"
    r"buy now|order now|subscribe|unsubscribe)\b",
    re.IGNORECASE,
)


class RuleBasedClassifier(BaseClassifier):
    """Deterministic rule-based content classifier.

    Applies regex patterns, keyword blocklists, and spam heuristics to
    detect problematic content. Every match is returned with full
    confidence (1.0) since rules are deterministic.

    Attributes:
        config: Rules configuration object.
    """

    def __init__(self, config: Optional[RulesConfig] = None) -> None:
        """Initialize the rule-based classifier.

        Args:
            config: Rules configuration. Uses defaults if not provided.
        """
        self.config = config or RulesConfig()
        self._blocked_pattern = self._build_blocked_pattern()
        self._custom_patterns = self._compile_custom_patterns()

    def _build_blocked_pattern(self) -> re.Pattern:
        """Build a compiled regex from the blocked keyword list.

        Returns:
            A compiled regex pattern matching any blocked keyword as a
            whole word (case-insensitive).
        """
        escaped = [re.escape(kw) for kw in self.config.blocked_keywords]
        pattern = r"\b(" + "|".join(escaped) + r")\b"
        return re.compile(pattern, re.IGNORECASE)

    def _compile_custom_patterns(self) -> dict[str, re.Pattern]:
        """Compile custom regex patterns from configuration.

        Returns:
            Dictionary mapping rule names to compiled patterns.
        """
        compiled = {}
        for name, pattern in self.config.custom_patterns.items():
            compiled[name] = re.compile(pattern, re.IGNORECASE)
        return compiled

    def classify(self, text: str) -> RuleResult:
        """Classify text using rule-based heuristics.

        Checks for blocked keywords, spam indicators (all-caps ratio,
        excessive punctuation, URL flooding), and custom patterns.

        Args:
            text: The text content to classify.

        Returns:
            A RuleResult with matched rules, severity, and details.
        """
        matched_rules: list[str] = []
        details: dict = {}
        severity = 0.0

        # Check blocked keywords
        keyword_matches = self._blocked_pattern.findall(text)
        if keyword_matches:
            matched_rules.append("blocked_keyword")
            details["blocked_keywords_found"] = list(set(
                kw.lower() for kw in keyword_matches
            ))
            severity = max(severity, 0.8)

        # Check all-caps ratio (spam indicator)
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) > 10:
            caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if caps_ratio >= self.config.spam_caps_ratio:
                matched_rules.append("excessive_caps")
                details["caps_ratio"] = round(caps_ratio, 2)
                severity = max(severity, 0.4)

        # Check excessive punctuation
        if _EXCESSIVE_PUNCTUATION.search(text):
            matched_rules.append("excessive_punctuation")
            severity = max(severity, 0.3)

        # Check repeated characters
        if _REPEATED_CHARS.search(text):
            matched_rules.append("repeated_characters")
            severity = max(severity, 0.2)

        # Check URL count
        urls = _URL_PATTERN.findall(text)
        if len(urls) > self.config.max_url_count:
            matched_rules.append("url_flooding")
            details["url_count"] = len(urls)
            severity = max(severity, 0.5)

        # Check spam phrases
        spam_matches = _EMAIL_SPAM_PATTERN.findall(text)
        if len(spam_matches) >= 2:
            matched_rules.append("spam_phrases")
            details["spam_phrases_found"] = list(set(
                m.lower() for m in spam_matches
            ))
            severity = max(severity, 0.5)

        # Check custom patterns
        for rule_name, pattern in self._custom_patterns.items():
            if pattern.search(text):
                matched_rules.append(f"custom:{rule_name}")
                severity = max(severity, 0.6)

        return RuleResult(
            matched_rules=matched_rules,
            severity=severity,
            confidence=1.0 if matched_rules else 0.0,
            details=details,
        )
