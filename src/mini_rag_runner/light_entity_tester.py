import os
import asyncio
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from tempfile import TemporaryDirectory
from numpy import add
import typer
from lightrag.kg.shared_storage import initialize_pipeline_status

app = typer.Typer()

@app.command()
def main(file: typer.FileText,
         openai_key: str = typer.Option(
        None, "--openai-key", help="OpenAI API key to set as OPENAI_API_KEY environment variable"
    ),
):
    """
    Run entity extraction on an input file and dump output on everything found.
    """
    async def async_main():
        # Get openai key set
        os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

        # Create a temp directory for the working directory
        with TemporaryDirectory() as working_dir:
            # Create a LightRAG instance
            rag = LightRAG(
                working_dir=str(working_dir),
                embedding_func=openai_embed,
                llm_model_func=gpt_4o_mini_complete,
                addon_params={
                    "entity_types": ["experiment", "physics concept or theory", "organization", "person", "geo", "event", "category"]
                },
                chunk_token_size=600,
                chunk_overlap_token_size=50,
            )

            await rag.initialize_storages()
            await initialize_pipeline_status()

            # Insert the text into the store.

            # Run entity extraction
            input_text = file.read()
            await rag.ainsert(input_text)

            # Get all the entities we found, and the relationships, back out.
            await rag.aexport_data("entity_info.md", "md")
            await rag.aexport_data("entity_info.xlsx", "excel")

    asyncio.run(async_main())

def start_main():
    app()
    # typer.run(main)

if __name__ == "__main__":
    app()