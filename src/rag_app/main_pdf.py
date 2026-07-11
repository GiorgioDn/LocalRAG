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
            result = answer_question(
                question=args.question,
                collection_name=args.collection
            )
            
            # Output the result and sources if available
            print(f"\nAnswer:\n{result['answer']}")
            
            if "sources" in result and result["sources"]:
                print("\nSources:")
                for idx, source in enumerate(result["sources"], 1):
                    source_info = source.get("metadata", {}).get("source", "Unknown Source")
                    print(f"{idx}. {source_info}")
        except Exception as e:
            logger.exception("Failed to generate answer for question: %s", args.question)
            print(f"Error during Q&A generation: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()