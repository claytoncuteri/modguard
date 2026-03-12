"""FastAPI application factory for ModGuard.

Creates and configures the FastAPI application with all routes, WebSocket
handlers, and static file serving for the dashboard.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from modguard.api.routes import create_router
from modguard.api.websocket import create_ws_router
from modguard.config import PipelineConfig
from modguard.pipeline import ModerationPipeline


def create_app(config: Optional[PipelineConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Pipeline configuration. Uses defaults if not provided.

    Returns:
        A configured FastAPI application instance.
    """
    config = config or PipelineConfig()

    app = FastAPI(
        title="ModGuard",
        description="Multi-layered AI content moderation API",
        version="0.1.0",
    )

    # CORS middleware for dashboard access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize shared pipeline and state
    pipeline = ModerationPipeline(config)

    app.state.pipeline = pipeline
    app.state.config = config
    app.state.history = []
    app.state.stats = {
        "total_processed": 0,
        "approved": 0,
        "flagged": 0,
        "rejected": 0,
        "total_confidence": 0.0,
        "total_processing_time_ms": 0.0,
    }
    app.state.ws_connections = set()

    # Register routers
    api_router = create_router()
    ws_router = create_ws_router()

    app.include_router(api_router)
    app.include_router(ws_router)

    return app
