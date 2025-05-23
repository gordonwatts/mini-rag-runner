import asyncio
import logging
import os
import time
from tempfile import TemporaryDirectory
from typing import AsyncIterator, Union

import typer
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete, openai_embed

app = typer.Typer()


async def local_openai_interface(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    # logging.warning(f"Extra ars are **kwargs: {kwargs}")
    extra_args = {"base_url": "http://localhost:12434/engines/llama.cpp/v1"}
    # Merge extra_args into kwargs, giving precedence to values in kwargs
    merged_kwargs = {**extra_args, **kwargs}
    return await openai_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **merged_kwargs,
    )


@app.command()
def main(
    file: typer.FileText,
    openai_key: str = typer.Option(
        None,
        "--openai-key",
        help="OpenAI API key to set as OPENAI_API_KEY environment variable",
        rich_help_panel="LLM Options",
    ),
    llm_model_name: str = typer.Option(
        "gpt-4o-mini",
        "--llm-model-name",
        help="LLM model name to use (default: gpt-4o-mini)",
        rich_help_panel="LLM Options",
    ),
    llm_source: str = typer.Option(
        "openai",
        "--llm-source",
        help="LLM source to use: 'openai' or 'docker'",
        case_sensitive=False,
        show_choices=True,
        rich_help_panel="LLM Options",
        prompt=False,
        metavar="[openai|docker]",
        callback=lambda v: v.lower() if v else v,
    ),
):
    """
    Run entity extraction on an input file and dump output on everything found.
    """

    async def async_main():
        # Get openai key & location set
        os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

        # If we are doing a local run, we need to use a different openai interface.
        # Note that for a text embedding that still uses openai, we need the
        # default key still.
        # Select the LLM function based on llm_source
        if llm_source == "docker":
            llm_model_func = local_openai_interface
        else:
            llm_model_func = openai_complete

        # Create a temp directory for the working directory
        with TemporaryDirectory() as working_dir:
            # Create a LightRAG instance
            rag = LightRAG(
                working_dir=str(working_dir),
                embedding_func=openai_embed,
                llm_model_func=llm_model_func,
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
                llm_model_name=llm_model_name,
                chunk_token_size=600,
                chunk_overlap_token_size=50,
                llm_model_max_async=1,
            )

            await rag.initialize_storages()
            await initialize_pipeline_status()

            # Insert the text into the store.

            # Run entity extraction
            input_text = file.read()
            start_time = time.time()
            await rag.ainsert(input_text)
            elapsed_time = time.time() - start_time
            logging.warning(f"Entity extraction took {elapsed_time:.2f} seconds")

            # Get all the entities we found, and the relationships, back out.
            await rag.aexport_data("entity_info.md", "md")
            await rag.aexport_data("entity_info.xlsx", "excel")

    asyncio.run(async_main())


def start_main():
    app()
    # typer.run(main)


if __name__ == "__main__":
    app()
