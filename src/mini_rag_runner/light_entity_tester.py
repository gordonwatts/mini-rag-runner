from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from tempfile import TemporaryDirectory
import typer

app = typer.Typer()

@app.command()
def main(file: typer.FileText):
    """
    Run entity extraction on an input file and dump output on everything found.
    """
    # Create a temp directory for the working directory
    with TemporaryDirectory() as working_dir:
        # Create a LightRAG instance
        rag = LightRAG(
            working_dir=str(working_dir),
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
        )

        # Insert the text into the store.

        # Run entity extraction
        input_text = file.read()
        rag.insert(input_text)

        # Get all the entities we found, and the relationships, back out.
        rag.export_data("entity_info.md", "md")

def start_main():
    app()
    # typer.run(main)

if __name__ == "__main__":
    app()