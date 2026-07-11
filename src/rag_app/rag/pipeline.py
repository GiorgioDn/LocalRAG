from pathlib import Path
from typing import List, Optional

from src.rag_app.config import settings
from src.rag_app.ingestion.pdf_loader import load_pdf_file
from src.rag_app.ingestion.text_loader import load_text_file
from src.rag_app.rag.generator import RagGenerator
from src.rag_app.rag.indexer import DocumentIndexer
from src.rag_app.rag.prompt_builder import build_rag_prompt
from src.rag_app.rag.retriever import Retriever
from src.rag_app.types import DocumentChunk, RagAnswer

def ingest_text_path(
    path: str | Path,
    collection_name: str | None = None,
    *,
    indexer: DocumentIndexer | None = None,
) -> list[DocumentChunk]:
    """
    Ingests a text file into the specified (or default) collection.
    """
    collection_name = collection_name or settings.default_collection
    
    # Load documents
    loaded_documents = load_text_file(path)
    
    # Get indexer
    active_indexer = indexer or DocumentIndexer()
    
    # Index documents
    chunks = active_indexer.index_documents(loaded_documents, collection_name)
    
    return list(chunks)

def ingest_pdf_path(
    path: str | Path,
    collection_name: str | None = None,
    *,
    indexer: DocumentIndexer | None = None,
) -> list[DocumentChunk]:
    """
    Ingests a PDF file into the specified (or default) collection.
    """
    collection_name = collection_name or settings.default_collection
    
    # Load documents
    loaded_documents = load_pdf_file(path)
    
    # Get indexer
    active_indexer = indexer or DocumentIndexer()
    
    # Index documents
    chunks = active_indexer.index_documents(loaded_documents, collection_name)
    
    return list(chunks)

def answer_question(
    question: str,
    collection_name: str | None = None,
    top_k: int | None = None,
    *,
    retriever: Retriever | None = None,
    generator: RagGenerator | None = None,
) -> RagAnswer:
    """
    Answers a question using the RAG pipeline.
    """
    # Resolve defaults
    collection_name = collection_name or settings.default_collection
    top_k = top_k if top_k is not None else settings.top_k

    # Validations
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if not collection_name.strip():
        raise ValueError("Collection name must not be empty.")
    if top_k <= 0:
        raise ValueError(f"top_k must be greater than zero (received {top_k}).")

    # Get components
    active_retriever = retriever or Retriever()
    active_generator = generator or RagGenerator()

    # Step 1: Retrieve chunks
    retrieved_chunks = active_retriever.retrieve(question, collection_name, top_k)
    
    # Step 2: Build prompt
    prompt = build_rag_prompt(question, retrieved_chunks)
    
    # Step 3: Generate answer
    return active_generator.generate(prompt, retrieved_chunks)