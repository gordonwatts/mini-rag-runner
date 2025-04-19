from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, cast

import typer
from fastapi import FastAPI, Request
from lightrag import LightRAG, QueryParam
from pydantic import BaseModel


@dataclass
class rag_context:
    rag: LightRAG


@asynccontextmanager
async def rag_context_setup(app: FastAPI):
    "Create the RAG application"
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
    from lightrag.utils import setup_logger

    # setup_logger("lightrag", level="INFO")

    working_dir = Path("/home/gwatts/code/llm/storage-esu")
    assert working_dir.exists()

    rag = LightRAG(
        working_dir=str(working_dir),
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    # Make sure everyone can use it.
    app.state.context = rag_context(rag)

    # Let it go. No need for any teardown afterwards.
    yield


class RagResponse(BaseModel):
    chunk: str
    document_reference: str


class RagRequest(BaseModel):
    question: str


app = FastAPI(lifespan=rag_context_setup)


@app.post("/get_rag_data", response_model=List[RagResponse])
def get_rag_data(request: RagRequest, fastapi_request: Request):
    r_context = cast(rag_context, fastapi_request.app.state.context)

    # # Perform naive search
    # mode = "naive"
    # # Perform local search
    # mode = "local"
    # # Perform global search
    # mode = "global"
    # # Perform hybrid search
    # mode = "hybrid"
    # Mix mode Integrates knowledge graph and vector retrieval.
    mode = "mix"

    q_params = QueryParam(mode=mode, only_need_context=True)

    result = r_context.rag.query(request.question, param=q_params)

    # Placeholder logic: return dummy data
    return [
        RagResponse(chunk=f"Example chunk 1 {result}", document_reference="doc1.pdf"),
    ]


def main(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(8001, help="Port to bind the server to"),
):
    # imports here to make sure command line interactions are fast.
    import lightrag
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def start_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
