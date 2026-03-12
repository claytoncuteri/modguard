"""HTTP route handlers for the ModGuard API.

Provides endpoints for single moderation, batch moderation, statistics,
history browsing, and serving the dashboard HTML.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


class ModerateRequest(BaseModel):
    """Request body for the single moderation endpoint.

    Attributes:
        text: The text content to moderate.
    """

    text: str


class BatchModerateRequest(BaseModel):
    """Request body for the batch moderation endpoint.

    Attributes:
        texts: List of text strings to moderate.
    """

    texts: list[str]


def create_router() -> APIRouter:
    """Create the API router with all HTTP endpoints.

    Returns:
        A configured APIRouter instance.
    """
    router = APIRouter()

    @router.post("/moderate")
    async def moderate(request: ModerateRequest, req: Request) -> JSONResponse:
        """Moderate a single text item.

        Args:
            request: The moderation request containing the text.
            req: The FastAPI request object for accessing app state.

        Returns:
            JSON response with the moderation result.
        """
        pipeline = req.app.state.pipeline
        result = pipeline.moderate(request.text)
        result_dict = result.to_dict()

        # Update stats and history
        _update_stats(req, result)
        _add_to_history(req, result_dict)

        # Notify WebSocket clients
        await _broadcast_ws(req, result_dict)

        return JSONResponse(content=result_dict)

    @router.post("/moderate/batch")
    async def moderate_batch(
        request: BatchModerateRequest, req: Request
    ) -> JSONResponse:
        """Moderate multiple text items concurrently.

        Args:
            request: The batch request containing a list of texts.
            req: The FastAPI request object for accessing app state.

        Returns:
            JSON response with a list of moderation results.
        """
        pipeline = req.app.state.pipeline
        results = await pipeline.moderate_batch(request.texts)

        result_dicts = []
        for result in results:
            result_dict = result.to_dict()
            _update_stats(req, result)
            _add_to_history(req, result_dict)
            await _broadcast_ws(req, result_dict)
            result_dicts.append(result_dict)

        return JSONResponse(content={"results": result_dicts})

    @router.get("/stats")
    async def get_stats(req: Request) -> JSONResponse:
        """Get aggregated moderation statistics.

        Args:
            req: The FastAPI request object for accessing app state.

        Returns:
            JSON response with moderation statistics.
        """
        stats = req.app.state.stats
        total = stats["total_processed"]

        response = {
            "total_processed": total,
            "approved": stats["approved"],
            "flagged": stats["flagged"],
            "rejected": stats["rejected"],
            "approval_rate": (
                stats["approved"] / total if total > 0 else 0.0
            ),
            "flagged_rate": (
                stats["flagged"] / total if total > 0 else 0.0
            ),
            "rejection_rate": (
                stats["rejected"] / total if total > 0 else 0.0
            ),
            "avg_confidence": (
                stats["total_confidence"] / total if total > 0 else 0.0
            ),
            "avg_processing_time_ms": (
                stats["total_processing_time_ms"] / total if total > 0 else 0.0
            ),
        }
        return JSONResponse(content=response)

    @router.get("/history")
    async def get_history(
        req: Request,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        decision: Optional[str] = Query(None),
    ) -> JSONResponse:
        """Get paginated moderation history.

        Args:
            req: The FastAPI request object for accessing app state.
            page: Page number (1-indexed).
            page_size: Number of items per page.
            decision: Optional filter by decision type.

        Returns:
            JSON response with paginated history and metadata.
        """
        history = req.app.state.history

        # Filter by decision if specified
        if decision:
            history = [
                item for item in history
                if item.get("decision") == decision.upper()
            ]

        # Most recent first
        history = list(reversed(history))

        total = len(history)
        start = (page - 1) * page_size
        end = start + page_size
        items = history[start:end]

        return JSONResponse(content={
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        })

    @router.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        """Serve the ModGuard dashboard HTML page.

        Returns:
            The dashboard HTML as an HTMLResponse.
        """
        dashboard_dir = Path(__file__).parent.parent / "dashboard"
        html_path = dashboard_dir / "index.html"

        if html_path.exists():
            return HTMLResponse(content=html_path.read_text())

        return HTMLResponse(
            content="<h1>Dashboard not found</h1>",
            status_code=404,
        )

    @router.get("/dashboard/styles.css")
    async def dashboard_css() -> Any:
        """Serve the dashboard CSS file."""
        from fastapi.responses import Response

        dashboard_dir = Path(__file__).parent.parent / "dashboard"
        css_path = dashboard_dir / "styles.css"

        if css_path.exists():
            return Response(
                content=css_path.read_text(),
                media_type="text/css",
            )
        return Response(content="", media_type="text/css", status_code=404)

    @router.get("/dashboard/app.js")
    async def dashboard_js() -> Any:
        """Serve the dashboard JavaScript file."""
        from fastapi.responses import Response

        dashboard_dir = Path(__file__).parent.parent / "dashboard"
        js_path = dashboard_dir / "app.js"

        if js_path.exists():
            return Response(
                content=js_path.read_text(),
                media_type="application/javascript",
            )
        return Response(
            content="", media_type="application/javascript", status_code=404
        )

    return router


def _update_stats(req: Request, result: Any) -> None:
    """Update in-memory moderation statistics.

    Args:
        req: The FastAPI request for accessing app state.
        result: The moderation result to record.
    """
    stats = req.app.state.stats
    stats["total_processed"] += 1
    stats["total_confidence"] += result.confidence
    stats["total_processing_time_ms"] += result.processing_time_ms

    decision = result.decision.value
    if decision == "APPROVE":
        stats["approved"] += 1
    elif decision == "FLAG_FOR_REVIEW":
        stats["flagged"] += 1
    elif decision == "REJECT":
        stats["rejected"] += 1


def _add_to_history(req: Request, result_dict: dict) -> None:
    """Add a moderation result to the in-memory history.

    Args:
        req: The FastAPI request for accessing app state.
        result_dict: The serialized moderation result.
    """
    history = req.app.state.history
    max_size = req.app.state.config.server.history_max_size

    history.append(result_dict)

    # Trim oldest entries if over capacity
    if len(history) > max_size:
        req.app.state.history = history[-max_size:]


async def _broadcast_ws(req: Request, result_dict: dict) -> None:
    """Broadcast a moderation result to all connected WebSocket clients.

    Args:
        req: The FastAPI request for accessing app state.
        result_dict: The serialized moderation result to broadcast.
    """
    message = json.dumps(result_dict)
    dead_connections = set()

    for ws in req.app.state.ws_connections:
        try:
            await ws.send_text(message)
        except Exception:
            dead_connections.add(ws)

    # Clean up dead connections
    req.app.state.ws_connections -= dead_connections
