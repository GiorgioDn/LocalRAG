from typing import List, Optional

from src.rag_app.config import settings
from src.rag_app.models.ollama_embeddings import OllamaEmbeddingClient
from src.rag_app.types import RetrievedChunk
from src.rag_app.vectorstore.chroma_connector import ChromaConnector


class Retriever:
    """
    Handles the retrieval of relevant document chunks from a vector store 
    based on a provided query string.
    """

    def __init__(
        self,
        embedding_client: Optional[OllamaEmbeddingClient] = None,
        vector_store: Optional[ChromaConnector] = None,
    ) -> None:
        """
        Initializes the Retriever with optional dependency injection for 
        the embedding client and vector store connector.

        Args:
            embedding_client: The client used to generate embeddings for queries.
            vector_store: The connector used to perform similarity searches in the vector database.
        """
        self.embedding_client = embedding_client or OllamaEmbeddingClient()
        self.vector_store = vector_store or ChromaConnector()

    def retrieve(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieves a list of RelevantChunk objects from the vector store.

        Args:
            query: The raw search query string.
            collection_name: The name of the collection to search. 
                               Defaults to settings.default_collection if None.
            top_k: The maximum number of results to return. 
                   Defaults to settings.top_k if None.

        Returns:
            A list of RetrievedChunk objects.

        Raises:
            ValueError: If the query is empty, collection name is invalid, or top_k <= 0.
            RuntimeError: If the vector store returns malformed results.
        """
        # 1. Resolve configuration and validate inputs
        resolved_collection = collection_name or settings.default_collection
        resolved_top_k = top_k if top_k is not None else settings.top_k

        if not query or not query.strip():
            raise ValueError("Query must not be empty or whitespace-only.")

        if not resolved_collection or not resolved_collection.strip():
            raise ValueError("Collection name must not be empty or whitespace-only.")

        if resolved_top_k is None or resolved_top_k <= 0:
            raise ValueError(f"top_k must be a positive integer (received {resolved_top_k}).")

        # 2. Generate embedding for the query
        # This call occurs exactly once as per requirements.
        query_embedding = self.embedding_client.embed_query(query)

        # 3. Perform similarity search in the vector store
        # This call occurs exactly once as per requirements.
        results = self.vector_store.search(
            collection_name=resolved_collection,
            query_embedding=query_embedding,
            top_k=resolved_top_k,
        )

        # 4. Map results to RetrievedChunk objects
        retrieved_chunks: List[RetrievedChunk] = []
        for item in results:
            # Validate that the result contains all required components
            if not all(k in item for k in ("text", "metadata", "score")):
                raise RuntimeError(
                    f"Vector store returned a malformed record missing one or more fields: {item}"
                )

            # Map to internal model, ensuring metadata is copied
            retrieved_chunks.append(
                RetrievedChunk(
                    text=str(item["text"]),
                    metadata=dict(item["metadata"]),  # Copy the dict
                    score=float(item["score"]),
                )
            )

        return retrieved_chunks