import argparse
import sys
from pathlib import Path
from typing import Optional

# Importing from shared pipelines
from src.rag_app.rag.pipeline import ingest_text_path, answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple CLI for text-based RAG operations.")
    parser.add_argument("path", type=str, help="Path to the input .txt file")
    parser.add_argument("--collection", type=str, default=None, help="Optional name for the collection")
    parser.add_argument("--question", type=str, default=None, help="The question to ask based on the text content")

    args = parser.parse_args()

    # 1. Ingestion phase
    try:
        ingest_text_path(path=args.path, collection_name=args.collection)
    except Exception as e:
        print(f"Error during ingestion: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. QA phase (only if a question is provided)
    if args.question:
        try:
            response = answer_question(question=args.question, collection_name=args.collection)
            
            # Print the primary answer
            print(f"\nAnswer:\n{response.answer}")
            
            # Print sources if available in a concise format
            if hasattr(response, "sources") and response.sources:
                print("\nSources:")
                for idx, source in enumerate(response.sources, 1):
                    # Assuming 'source' has an identifier or name attribute
                    name = getattr(source, 'name', str(source))
                    print(f"{idx}. {name}")
        except Exception as e:
            print(f"Error during QA generation: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()