"""
Module serving as the entry point for the mini-rag-mcp command.
This will run an LLM's MCP server using lightrag-hku.
"""

import sys
import argparse


def main():
    """
    Entry point for the mini-rag-mcp command.
    This function will initialize and run the MCP server using lightrag-hku.
    """
    parser = argparse.ArgumentParser(
        description="Run an LLM's MCP server with lightrag-hku"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")

    args = parser.parse_args()

    print(f"Starting MCP server with lightrag-hku")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Model: {args.model}")

    try:
        # This is a placeholder for the actual implementation
        # In a real implementation, you would import lightrag-hku
        # and start the MCP server here
        print("MCP server is running. Press Ctrl+C to stop.")
    except KeyboardInterrupt:
        print("\nShutting down MCP server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
