import argparse
import sys
from typing import Optional

from src.rag_app.rag.pipeline import ingest_pdf_path, answer_question
from src.rag_app.utils.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

def main() -> None:
    """
    Main entry point for the PDF RAG flow.
    Wires together ingestion and Q&A using the shared pipeline components.
    """
    configure_logging()
    
    parser = argparse.ArgumentParser(description="PDF-based Retrieval-Augmented Generation (RAG) CLI.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file for ingestion.")
    parser.add_argument("--collection", type=str, default="default_collection", help="Name of the vector store collection.")
    parser.add_argument("--question", type=str, help="The question or prompt to ask the RAG system.")
    parser.add_argument("--model", type=str, default=None, help="Optional Ollama chat model to use for answer generation.")

    args = parser.parse_args()

    try:
        # 1. Ingestion Phase
        ingest_pdf_path(
            pdf_path=args.pdf_path,
            collection_name=args.collection
        )
    except Exception as e:
        logger.exception("Failed to ingest PDF path: %s", args.pdf_path)
        print(f"Error during ingestion: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Q&A Phase (only if a question is provided via CLI)
    if args.question:
        try:
            generator = None
            if getattr(args, "model", None):
                from src.rag_app.rag.generator import RagGenerator
                from src.rag_app.models.ollama_chat import OllamaChatClient
                generator = RagGenerator(chat_client=OllamaChatClient(model=args.model))

            kwargs = {"question": args.question, "collection_name": args.collection}
            if generator is not None:
                kwargs["generator"] = generator

            result = answer_question(**kwargs)
            
            # Output the result and sources if available
            answer_text = result["answer"] if isinstance(result, dict) or hasattr(result, "__getitem__") else result.answer
            print(f"\nAnswer:\n{answer_text}")
            
            sources = result.get("sources", []) if hasattr(result, "get") else getattr(result, "sources", [])
            if sources:
                print("\nSources:")
                for idx, source in enumerate(sources, 1):
                    if isinstance(source, dict):
                        source_path = source.get("source") or source.get("metadata", {}).get("source", "Unknown Source")
                        page = source.get("page") if "page" in source else source.get("metadata", {}).get("page")
                        source_info = f"{source_path} (page {page})" if page is not None else str(source_path)
                    else:
                        source_info = str(source)
                    print(f"{idx}. {source_info}")
        except Exception as e:
            logger.exception("Failed to generate answer for question: %s", args.question)
            print(f"Error during Q&A generation: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()