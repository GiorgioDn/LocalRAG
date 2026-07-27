from typing import List, Optional

from src.rag_app.config import settings
from src.rag_app.ingestion.chunker import chunk_documents
from src.rag_app.models.ollama_embeddings import OllamaEmbeddingClient
from src.rag_app.types import DocumentChunk, LoadedDocument
from src.rag_app.vectorstore.chroma_connector import ChromaConnector

class DocumentIndexer:
    """
    Responsible for processing raw documents through chunking and embedding, 
    and persisting them to a vector store.
    """

    def __init__(
        self,
        chunker_module=None,
        embedding_client: Optional[OllamaEmbeddingClient] = None,
        vector_store: Optional[ChromaConnector] = None,
    ) -> None:
        """
        Initialize the indexer with injected dependencies or default components.

        Args:
            chunker_module: The module containing the chunking logic (optional).
            embedding_client: An instance of OllamaEmbeddingClient (optional).
            vector_store: An instance of ChromaConnector (optional).
        """
        # Note: Since 'chunk_documents' is a standalone function in chunker.py, 
        # if the user wants to inject a different chunking implementation, they would 
        # typically pass an object with a .chunk_documents() method or we keep it simple.
        # Based on requirements, we provide standard defaults for missing components.
        self._chunk_func = chunk_documents
        self.embedding_client = embedding_client or OllamaEmbeddingClient()
        self.vector_store = vector_store or ChromaConnector()

    def index_documents(
        self,
        documents: List[LoadedDocument],
        collection_name: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Process a list of loaded documents into chunks, embed them, and upsert to Chroma.

        Args:
            documents: A list of raw document contents with metadata.
            collection_name: The name of the collection in the vector store. 
                               Defaults to settings.default_collection if None.

        Returns:
            A list of successfully indexed DocumentChunk objects.

        Raises:
            ValueError: If the collection name is empty or whitespace only.
            RuntimeError: If the number of embeddings received does not match chunk count.
        """
        # 1. Resolve and validate collection name
        target_collection = collection_name if collection_name is not None else settings.default_collection
        if not target_collection or not target_collection.strip():
            raise ValueError("Collection name must not be empty.")
        
        target_collection = target_collection.strip()

        # 2. Handle empty input early
        if not documents:
            return []

        # 3. Chunk the documents
        chunks = self._chunk_func(documents)
        
        # 4. If chunking produces no results, exit early
        if not chunks:
            return []

        # 5. Prepare and embed text in a single batch
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_client.embed_texts(texts)

        # 6. Verify embedding consistency
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embeddings count ({len(embeddings)}) does not match "
                f"chunk count ({len(chunks)})."
            )

        # 7. Upsert to the vector store
        self.vector_store.upsert_documents(
            collection_name=target_collection,
            chunks=chunks,
            embeddings=embeddings,
        )

        return chunks