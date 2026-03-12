"""CLI entry point for ModGuard.

Provides the ``python -m modguard serve`` command to start the API server
with the real-time moderation dashboard.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Parse CLI arguments and dispatch the requested command."""
    parser = argparse.ArgumentParser(
        prog="modguard",
        description="ModGuard: AI content moderation pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve", help="Start the ModGuard API server"
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number (default: 8000)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    serve_parser.add_argument(
        "--no-ml",
        action="store_true",
        help="Disable ML models (rules-only mode for quick testing)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve":
        _run_server(args)


def _run_server(args: argparse.Namespace) -> None:
    """Start the FastAPI server with uvicorn.

    Args:
        args: Parsed command-line arguments.
    """
    import uvicorn

    from modguard.api.server import create_app
    from modguard.config import PipelineConfig

    config = PipelineConfig()
    config.server.host = args.host
    config.server.port = args.port

    if args.no_ml:
        config.enable_toxicity = False
        config.enable_sentiment = False

    app = create_app(config)

    print(f"Starting ModGuard server on {args.host}:{args.port}")
    print(f"Dashboard: http://{args.host}:{args.port}/dashboard")
    print(f"API docs:  http://{args.host}:{args.port}/docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
