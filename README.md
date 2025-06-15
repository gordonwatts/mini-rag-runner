# Mini RAG Runner

A package that runs an LLM's Model Context Protocol (MCP) server with the lightrag-hku library, and runs in-process.

## Motivation

LightRAG has a very nice API/server that can run in a second process. Unfortunately, I can't setup a docker on the server I need to run. This enables an MCP server, read-only, interfaced to lightrag, with github authentication and keys.

## Running

### Light Rag WebAPI

```bash
 light-rag-webapi ../azure-light-rag/storage-esu.tar.gz "European Union Strategy Update 2025 Submissions" --openai-key <key>
```

### MCP

```bash
 uvx --python=3.13 --from=git+<https://github.com/gordonwatts/mini-rag-runner.git@main> mini-rag-mcp
```

### Running from docker

Building the docker container:

```bash
docker build --rm -f 'Dockerfile' -t 'miniragrunner:latest' '.' 
```

Often it makes sense to build this for more than one image type. To do this, you'll need a one-time setup:

```bash
docker run --privileged --rm tonistiigi/binfmt --install all
docker buildx create --use
```

And now you can do the build for multiple platforms. Note: some cloud services can only run `amd64`.

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t gordonwatts/miniragrunner:1.0.0a1 --push .
```

And running it. Note you need to mount the database inside the container (somehow).

```bash
docker run -p8001:8001 -v ${PWD}/../storage-esu:/db --rm -it gordonwatts/miniragrunner:1.0.0a1 --rag-db /db --openai-key <api-key>
```

### How to generate the database

```bash
docker run -p8001:8001 -v ${PWD}\..\rag-db\eusu-2025:/db -v ${PWD}\..\data\ingest-temp:/ingest --rm -it miniragrunner:latest /db "European Union Strategy Update 2025" --ingest-dir /ingest --openai-key <key>
```

WARNING: files are removed from the ingestion directory as they are successfully ingested.

### How to prepare the tar.gz (or ZIP file) of the database

You can give this an upacked directory or a tar.gz file that can be unpacked. To prep the file:

1. `cd` into the database directory
1. Compress: `tar -czvf ../storage-esu.tar.gz .`

```bash
$ tar -czvf ../storage-esu.tar.gz .
./
./vdb_chunks.json
./vdb_relationships.json
./kv_store_full_docs.json
./vdb_entities.json
./graph_chunk_entity_relation.graphml
./kv_store_doc_status.json
./kv_store_text_chunks.json
```

### The `--rag-db` argument

This can be:

* A local directory with the light rag database
* A `http` pointer to a tar/gz or zip file (the URL must end with those file extensions)
* An Azure blob (e.g. `az://lightrag-data/esu.tar.gz`) - again, note that the name ends with the file type!

Example of the command line that will use a tar/gz file from an azure container:

```bash
light-rag-webapi az://lightrag-data/esu.tar.gz "European Particle Physics Strategy Update 2025 Document Database"  --openai-key <key> --account-name <az-storage-account> --account-key <az-storage-key>
```

`fsspec` is used under the hood to fetch the compressed file.

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

* `--host`: Host to bind the server to (default: 0.0.0.0)
* `--port`: Port to bind the server to (default: 8000)
* `--model`: LLM model to use (default: gpt-4o-mini)

## Environment Variables

The package uses the following environment variables that can be set in the `.env` file:

* `LLM_BINDING`: The LLM binding to use (default: openai)
* `LLM_MODEL`: The LLM model to use (default: gpt-4o-mini)
* `EMBEDDING_BINDING`: The embedding binding to use (default: openai)
* `EMBEDDING_MODEL`: The embedding model to use (default: text-embedding-3-small)
* `EMBEDDING_DIM`: The embedding dimension (default: 1536)

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

Note this will not build on raw windows - `gensim` does not build on most raw installations of windows.
