# Mini RAG Runner

A package that runs an LLM's Model Context Protocol (MCP) server with the lightrag-hku library, and runs in-process.

## Motivation

LightRAG has a very nice API/server that can run in a second process. Unfortunately, I can't setup a docker on the server I need to run. This enables an MCP server, read-only, interfaced to lightrag, with github authentication and keys.

## Running

```bash
 uvx --python=3.13 --from=git+<https://github.com/gordonwatts/mini-rag-runner.git@main> mini-rag-mcp
```

## Installation

You can install this package directly from the source:

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# OR
.\.venv\Scripts\activate  # On Windows

# Install the package in development mode
pip install -e .
```

## Usage

Once installed, you can run the MCP server using the `mini-rag-mcp` command:

```bash
mini-rag-mcp --host 0.0.0.0 --port 8000 --model gpt-4o-mini
```

### Available Options

- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8000)
- `--model`: LLM model to use (default: gpt-4o-mini)

## Environment Variables

The package uses the following environment variables that can be set in the `.env` file:

- `LLM_BINDING`: The LLM binding to use (default: openai)
- `LLM_MODEL`: The LLM model to use (default: gpt-4o-mini)
- `EMBEDDING_BINDING`: The embedding binding to use (default: openai)
- `EMBEDDING_MODEL`: The embedding model to use (default: text-embedding-3-small)
- `EMBEDDING_DIM`: The embedding dimension (default: 1536)

## Docker

You can also run the MCP server using Docker:

```bash
# Use the provided docker-compose files
docker-compose -f docker-compose-mini.yml up
```

## Development

This package uses [Hatch](https://hatch.pypa.io/) as its build system. To set up a development environment:

```bash
pip install hatch
hatch shell
```
