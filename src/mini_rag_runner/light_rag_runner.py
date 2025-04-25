from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, cast

import typer
from fastapi import FastAPI, Request
from lightrag import LightRAG, QueryParam
from pydantic import BaseModel


@dataclass
class rag_context:
    rag: LightRAG


class RagResponse(BaseModel):
    chunk: str
    document_reference: str


def create_app(working_dir: Path) -> FastAPI:
    "Define the complete app here"

    @asynccontextmanager
    async def rag_context_setup(app: FastAPI):
        "Create the RAG application"
        from lightrag.kg.shared_storage import initialize_pipeline_status
        from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
        from lightrag.utils import setup_logger

        # setup_logger("lightrag", level="INFO")

        assert working_dir.exists(), f"Failed to find working directory {working_dir}"

        rag = LightRAG(
            working_dir=str(working_dir),
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
        )

        await rag.initialize_storages()
        await initialize_pipeline_status()

        # Make sure everyone can use it.
        app.state.context = rag_context(rag)

        yield
        # Add clean up code here if we need it.

    app = FastAPI(
        lifespan=rag_context_setup,
        title="European Strategy Update 2025 Document Database",
        description="Endpoint access a vector database with entity relationships of the 264 documents.",
        version="1.0.0",
    )

    @app.post(
        "/get_rag_data",
        response_model=List[RagResponse],
        summary="Return list of documents from the vector db that are most closely matched to the `question`",
    )
    def get_rag_data(question: str, fastapi_request: Request, top_k: int = 20):
        r_context = cast(rag_context, fastapi_request.app.state.context)
        if r_context is None:
            raise RuntimeError("The RAG Context is `None` - should never happen.")
        mode = "mix"
        logging.debug(f"Querying for the question: '{question}'.")
        q_params = QueryParam(mode=mode, only_need_context=True)
        result = r_context.rag.query(question, param=q_params)
        logging.debug(f"  Result: '{result}'.")
        return [
            RagResponse(chunk=f"Example chunk 1 {result}", document_reference="doc1.pdf"),
        ]

    return app


def main(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(8001, help="Port to bind the server to"),
    rag_db: Path = typer.Option(
        ..., "--rag-db", help="Path to the RAG database directory (must exist)"
    ),
    openai_key: str = typer.Option(
        None, "--openai-key", help="OpenAI API key to set as OPEN_API_KEY environment variable"
    ),
):
    import uvicorn
    from pathlib import Path
    import os

    os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    if not rag_db.exists() or not (rag_db / "graph_chunk_entity_relation.graphml").exists():
        raise ValueError(f"Failed to find rag database in directory {rag_db}")

    app = create_app(rag_db.absolute())
    uvicorn.run(app, host=host, port=port)


def start_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
