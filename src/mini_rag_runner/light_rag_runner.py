import logging
import os
import tarfile
import tempfile
import zipfile
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, cast

import fsspec
import typer
from fastapi import FastAPI, Query, Request
from lightrag import LightRAG, QueryParam
from lightrag import prompt as lg_prompt
from pydantic import BaseModel


@dataclass
class rag_context:
    rag: LightRAG


class RagResponse(BaseModel):
    chunk: str
    document_reference: str


def create_app(
    working_dir: Path,
    title: str,
    servers: Optional[List[Dict[str, str]]] = None,
    ingest_dir: Optional[Path] = None,
) -> FastAPI:
    "Define the complete app here"

    import asyncio

    @asynccontextmanager
    async def rag_context_setup(app: FastAPI):
        "Create the RAG application"
        from lightrag.kg.shared_storage import initialize_pipeline_status
        from lightrag.llm.openai import openai_complete, openai_embed

        # setup_logger("lightrag", level="INFO")

        assert working_dir.exists(), f"Failed to find working directory {working_dir}"

        rag = LightRAG(
            working_dir=str(working_dir),
            embedding_func=openai_embed,
            llm_model_func=openai_complete,
            addon_params={
                "entity_types": [
                    "physics detector/experiment",
                    "physics concept or theory",
                    "country",
                    "organization",
                    "person",
                    "geo",
                    "event",
                    "category",
                ],
            },
            llm_model_name="gpt-4.1-nano",
            chunk_token_size=600,
            chunk_overlap_token_size=50,
        )

        await rag.initialize_storages()
        await initialize_pipeline_status()

        # Make sure everyone can use it.
        app.state.context = rag_context(rag)
        app.state.ingest_lock = asyncio.Lock()
        app.state.stop_ingest = False

        async def ingest_watcher():
            while not app.state.stop_ingest:
                if ingest_dir and ingest_dir.exists():
                    files = [f for f in ingest_dir.iterdir() if f.is_file()]
                    for file in files:
                        async with app.state.ingest_lock:
                            try:
                                with open(file, "r", encoding="utf-8", errors="ignore") as fin:
                                    content = fin.read()
                                logging.warning(f"Ingesting new file: {file}")
                                await rag.ainsert(content, file_paths=str(file))
                                logging.warning(f"Ingested and removed file: {file}")
                            except Exception as e:
                                logging.error(f"Failed to ingest {file}: {e}")
                            finally:
                                try:
                                    file.unlink()
                                except Exception as unlink_err:
                                    logging.error(f"Failed to remove file {file}: {unlink_err}")
                await asyncio.sleep(5)

        # Start watcher if ingest_dir is set
        if ingest_dir:
            app.state.ingest_task = asyncio.create_task(ingest_watcher())
        yield
        # Cleanup
        app.state.stop_ingest = True
        if ingest_dir and hasattr(app.state, "ingest_task"):
            await app.state.ingest_task

    # Use provided servers or default to localhost
    app = FastAPI(
        lifespan=rag_context_setup,
        title=title,
        description="Endpoint access a vector database with entity relationships of the "
        "documents.",
        version="1.0.0",
        servers=servers,
    )

    @app.post(
        "/get_rag_data",
        response_model=List[RagResponse],
        summary="Return list, as a single string, of documents from the vector db that are most"
        " closely matched to the `question`",
    )
    def get_rag_data(
        fastapi_request: Request,
        question: str = Query(..., description="text to use as vector db lookup"),
        top_k: int = Query(20, description="Top N documents to return"),
    ):
        r_context = cast(rag_context, fastapi_request.app.state.context)
        if r_context is None:
            raise RuntimeError("The RAG Context is `None` - should never happen.")
        mode = "hybrid"
        logging.debug(f"Querying for the question: '{question}'.")
        q_params = QueryParam(
            mode=mode,
            top_k=top_k,
            response_type=(
                "All useful quotes from supplied text as bullet points "
                "with a source at the end of each quote. Clearly indicating "
                "whether each source is from Knowledge Graph (KG) or Document "
                "Chunks (DC), and include the file path if available, in the "
                "following format: [KG/DC] file_path. No longer than 4000 characters."
            ),
            user_prompt="Don't draw conclusions or give intro text. Just the list of quotes.",
        )
        result = r_context.rag.query(question, param=q_params)
        logging.debug(f"  Result: '{result}'.")
        return [
            RagResponse(chunk=f"{result}", document_reference="??"),
        ]

    return app


@contextmanager
def resolve_rag_db(
    rag_db: str, account_name: Optional[str] = None, account_key: Optional[str] = None
):
    """
    Context manager: yields the path to the RAG DB directory, whether local or extracted from an
        archive/remote.
    Cleans up tempdir automatically if used.
    """
    if Path(rag_db).is_dir():
        yield Path(rag_db)
    else:
        tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(tmpdir.name)
        # Use provided account_name/key or fallback to environment variables
        account_name = account_name or os.getenv("LIGHTRAG_ACCOUNT_NAME")
        account_key = account_key or os.getenv("LIGHTRAG_ACCOUNT_KEY")
        # Only set storage_options if both are present and non-blank
        if account_name and account_key:
            storage_options = {
                "account_name": account_name,
                "account_key": account_key,
            }
            open_args = dict(**storage_options)
        else:
            open_args = {}
        with fsspec.open(str(rag_db), "rb", **open_args) as f:
            if str(rag_db).endswith(".zip"):
                with zipfile.ZipFile(f) as zf:  # type: ignore
                    zf.extractall(tmp_path)
            elif str(rag_db).endswith((".tar.gz", ".tgz")):
                with tarfile.open(fileobj=f, mode="r:gz") as tf:  # type: ignore
                    tf.extractall(tmp_path)
            else:
                raise ValueError("Unsupported archive format for rag_db")
        try:
            yield tmp_path
        finally:
            tmpdir.cleanup()


def main(
    rag_db: str = typer.Argument(
        ...,
        help="Path to the RAG database directory (must exist). If empty, will create a new database.",
    ),
    title: str = typer.Argument(..., help="Title for the FastAPI app"),
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(8001, help="Port to bind the server to"),
    openai_key: str = typer.Option(
        None, "--openai-key", help="OpenAI API key to set as OPEN_API_KEY environment variable"
    ),
    server: List[str] = typer.Option(
        None,
        "--server",
        help="Server URL to inject into the OpenAPI servers list. Repeat for multiple servers.",
    ),
    account_name: Optional[str] = typer.Option(
        None,
        "--account-name",
        help="Azure Storage account name (or set LIGHTRAG_ACCOUNT_NAME env var)",
    ),
    account_key: Optional[str] = typer.Option(
        None,
        "--account-key",
        help="Azure Storage account key (or set LIGHTRAG_ACCOUNT_KEY env var)",
    ),
    ingest_dir: Optional[Path] = typer.Option(
        None,
        "--ingest-dir",
        help="Path to a local directory to watch for ingestion.",
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
):
    # Configure lightrag a little bit before anything else gets going.
    def do_replace(source, s, d) -> str:
        assert s in source, f"Did not found replacement for {s} in {d}"
        return source.replace(s, d)

    lg_prompt.PROMPTS["rag_response"] = do_replace(
        lg_prompt.PROMPTS["rag_response"],
        "Use markdown formatting with appropriate section headings",
        "Use markdown formatting",
    )
    lg_prompt.PROMPTS["rag_response"] = do_replace(
        lg_prompt.PROMPTS["rag_response"],
        "Generate a concise response",
        "Generate a response",
    )
    lg_prompt.PROMPTS["rag_response"] = do_replace(
        lg_prompt.PROMPTS["rag_response"],
        "- Ensure the response maintains continuity with the conversation history.",
        "",
    )
    lg_prompt.PROMPTS["rag_response"] = do_replace(
        lg_prompt.PROMPTS["rag_response"],
        '- List up to 5 most important reference sources at the end under "References" section. '
        "Clearly indicating whether each source is from Knowledge Graph (KG) or Document Chunks "
        "(DC), and include the file path if available, in the following format: [KG/DC] file_path",
        "",
    )

    logging.info(f"rag_response prompt: {lg_prompt.PROMPTS['rag_response']}")

    # Next, get the server up and running
    import os

    import uvicorn

    os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    with resolve_rag_db(rag_db, account_name=account_name, account_key=account_key) as rag_db_path:
        if not (rag_db_path / "graph_chunk_entity_relation.graphml").exists():
            logging.warning(
                f"Failed to find graph_chunk_entity_relation.graphml in {rag_db_path}. "
                "Creating empty database."
            )
        servers_list = [{"url": url} for url in server] if server else None
        app = create_app(
            rag_db_path.absolute(), title=title, servers=servers_list, ingest_dir=ingest_dir
        )
        uvicorn.run(app, host=host, port=port)


def start_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
