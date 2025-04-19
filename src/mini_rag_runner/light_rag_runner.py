from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import typer

app = FastAPI()


class RagResponse(BaseModel):
    chunk: str
    document_reference: str


class RagRequest(BaseModel):
    question: str


@app.post("/get_rag_data", response_model=List[RagResponse])
def get_rag_data(request: RagRequest):
    # Placeholder logic: return dummy data
    return [
        RagResponse(chunk="Example chunk 1", document_reference="doc1.pdf"),
        RagResponse(chunk="Example chunk 2", document_reference="doc2.pdf"),
    ]


def main(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(8001, help="Port to bind the server to"),
):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def start_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
