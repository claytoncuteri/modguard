"""WebSocket handler for real-time moderation feed.

Clients connect to /ws and receive JSON-encoded moderation results
as they are processed. The connection is automatically cleaned up
on disconnect.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


def create_ws_router() -> APIRouter:
    """Create the WebSocket router.

    Returns:
        An APIRouter with the WebSocket endpoint registered.
    """
    router = APIRouter()

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """Handle a WebSocket connection for real-time updates.

        Accepts the connection, registers it for broadcasts, and keeps
        it alive until the client disconnects.

        Args:
            websocket: The WebSocket connection object.
        """
        await websocket.accept()
        websocket.app.state.ws_connections.add(websocket)
        logger.info("WebSocket client connected. Total: %d",
                     len(websocket.app.state.ws_connections))

        try:
            while True:
                # Keep the connection alive by reading messages
                # Clients can optionally send text for inline moderation
                data = await websocket.receive_text()

                # If the client sends text, moderate it and reply
                if data.strip():
                    pipeline = websocket.app.state.pipeline
                    result = pipeline.moderate(data)
                    await websocket.send_json(result.to_dict())

        except WebSocketDisconnect:
            websocket.app.state.ws_connections.discard(websocket)
            logger.info("WebSocket client disconnected. Total: %d",
                         len(websocket.app.state.ws_connections))
        except Exception as exc:
            logger.warning("WebSocket error: %s", exc)
            websocket.app.state.ws_connections.discard(websocket)

    return router
