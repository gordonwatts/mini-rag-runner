import asyncio
import sys
from typing import List

import typer
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo")


@mcp.tool()
def get_conference_titles() -> List[str]:
    """Returns the list of conferences this server is serving info about.
    Use these as the `conference_name` in other tool calls."""
    return ["European Strategy Update Document Database"]


def main(
    host: str = typer.Option("localhost", help="Host to bind the server to"),
    port: int = typer.Option(8000, help="Port to bind the server to"),
    model: str = typer.Option("gpt-4o-mini", help="LLM model to use"),
):
    """
    Run an LLM's MCP server with conference information.
    """
    print(f"Starting MCP Conference Info server")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Model: {model}")

    try:
        print(mcp.settings.host)
        print(mcp.settings.port)
        mcp.settings.host = host
        mcp.settings.port = port
        print(mcp.settings.host)
        print(mcp.settings.port)
        mcp.run("sse")
    except KeyboardInterrupt:
        print("\nShutting down MCP server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting MCP server: {e}", file=sys.stderr)
        sys.exit(1)


def start_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
