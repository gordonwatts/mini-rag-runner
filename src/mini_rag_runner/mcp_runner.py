import sys

import typer


def main(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(8000, help="Port to bind the server to"),
    model: str = typer.Option("gpt-4o-mini", help="LLM model to use"),
):
    """
    Run an LLM's MCP server with lightrag-hku.
    """
    print(f"Starting MCP server with lightrag-hku")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Model: {model}")

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


def start_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
