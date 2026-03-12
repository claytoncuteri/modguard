"""Abstract base class for all moderation classifiers.

Every classifier layer in the pipeline must extend BaseClassifier and
implement the classify method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseClassifier(ABC):
    """Abstract base class for moderation classifiers.

    All classifier layers must implement the classify method, which takes
    raw text input and returns a typed result object.
    """

    @abstractmethod
    def classify(self, text: str) -> Any:
        """Classify the given text.

        Args:
            text: The text content to classify.

        Returns:
            A typed result object specific to the classifier implementation.
        """
        ...

    @property
    def name(self) -> str:
        """Return the classifier name derived from the class name."""
        return self.__class__.__name__
