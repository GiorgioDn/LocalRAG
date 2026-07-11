import argparse
from typing import Optional

from src.rag_app.rag.pipeline import ingest_pdf_path, answer_question

def main() -> None:
    """
    Main entry point for the PDF RAG flow.
    Wires together ingestion and Q&A using the shared pipeline components.
    """
    parser = argparse.ArgumentParser(description="PDF-based Retrieval-Augmented Generation (RAG) CLI.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file for ingestion.")
    parser.add_argument("--collection", type=str, default="default_collection", help="Name of the vector store collection.")
    parser.add_argument("--question", type=str, help="The question or prompt to ask the RAG system.")

    args = parser.parse_args()

    # 1. Ingestion Phase
    ingest_pdf_path(
        pdf_path=args.pdf_path,
        collection_name=args.collection
    )

    # 2. Q&A Phase (only if a question is provided via CLI)
    if args.question:
        result = answer_question(
            question=args.question,
            collection_name=args.collection
        )
        
        # Output the result and sources if available
        print(f"\nAnswer:\n{result['answer']}")
        
        if "sources" in result and result["sources"]:
            print("\nSources:")
            for idx, source in enumerate(result["sources"], 1):
                # Format metadata/source info succinctly
                source_info = source.get("metadata", {}).get("source", "Unknown Source")
                print(f"{idx}. {source_info}")

if __name__ == "__main__":
    main()