"""FastAPI-based REST and WebSocket API for ModGuard.

Exposes endpoints for single and batch moderation, statistics,
history, and a real-time WebSocket feed. Also serves the browser
dashboard for monitoring moderation activity.
"""

from modguard.api.server import create_app

__all__ = ["create_app"]
