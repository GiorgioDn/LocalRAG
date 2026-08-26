import chromadb
from pathlib import Path
from typing import Any, Sequence

from src.rag_app.config import settings
from src.rag_app.types import DocumentChunk

class ChromaConnector:
    """
    Adapter for managing a local ChromaDB instance and its collections.
    """

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        client: chromadb.ClientAPI | None = None,
    ) -> None:
        """
        Initializes the Chroma connector with a persistent client and directory.

        Args:
            persist_directory: The path to the local storage. Defaults to settings.chroma_persist_dir.
            client: An optional pre-configured chromadb ClientAPI for injection (e.g., for testing).
        """
        if persist_directory is None:
            # Fallback to a default directory if the config is missing
            persist_directory = "storage/chroma" 
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        if client is not None:
            self.client = client
        else:
            # Default to a PersistentClient if no client is injected.
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))

    def get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """
        Retrieves an existing collection or creates a new one with cosine similarity settings.

        Args:
            collection_name: The name of the collection (e.g., "documents").

        Returns:
            A chromadb.Collection object.

        Raises:
            ValueError: If the provided collection_name is empty or only whitespace.
        """
        clean_name = collection_name.strip()
        if not clean_name:
            raise ValueError("Collection name must not be empty.")

        return self.client.get_or_create_collection(
            name=clean_name, 
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_documents(
        self,
        collection_name: str,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """
        Upserts a batch of documents into the specified collection.

        Args:
            collection_name: The name of the target collection.
            chunks: A sequence of DocumentChunk objects containing id, text, and metadata.
            embeddings: A sequence of embedding vectors for each chunk.

        Raises:
            ValueError: If validation fails (e.g., mismatch in counts, empty IDs/text, 
                        non-numeric embeddings, or duplicate IDs within the batch).
        """
        clean_name = collection_name.strip()
        if not clean_name:
            raise ValueError("Collection name must not be empty.")

        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatched count: {len(chunks)} chunks vs {len(embeddings)} embeddings.")

        prepared_ids = []
        prepared_docs = []
        prepared_metadatas = []
        prepared_embeddings = []
        
        seen_ids = set()
        first_dim = None

        for i, chunk in enumerate(chunks):
            cid = str(chunk.id).strip()
            if not cid:
                raise ValueError(f"Chunk at index {i} has an empty or whitespace-only ID.")
            if cid in seen_ids:
                raise ValueError(f"Duplicate ID detected within batch: {cid}")
            seen_ids.add(cid)

            text = str(chunk.text).strip()
            if not text:
                raise ValueError(f"Chunk at index {i} has empty or whitespace-only text.")

            metadata = chunk.metadata
            if not isinstance(metadata, dict):
                raise ValueError(f"Metadata for chunk at index {i} must be a dictionary.")

            embedding = embeddings[i]
            if not embedding:
                raise ValueError(f"Embedding for chunk at index {i} is empty.")
            
            current_dim = len(embedding)
            if first_dim is None:
                first_dim = current_dim
            elif current_dim != first_dim:
                raise ValueError(f"Inconsistent embedding dimension at index {i}: "
                                 f"{current_dim} vs {first_dim}")

            if not all(isinstance(x, (int, float)) for x in embedding):
                raise ValueError(f"Embedding at index {i} contains non-numeric values.")

            prepared_ids.append(cid)
            prepared_docs.append(text)
            prepared_metadatas.append(dict(metadata))
            prepared_embeddings.append([float(x) for x in embedding])

        collection = self.get_or_create_collection(clean_name)
        collection.upsert(
            ids=prepared_ids,
            documents=prepared_docs,
            embeddings=prepared_embeddings,
            metadatas=prepared_metadatas,
        )

    def search(
        self,
        collection_name: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Searches the collection for relevant documents using a query embedding.

        Args:
            collection_name: The name of the target collection.
            query_embedding: A sequence of floats representing the query vector.
            top_k: The maximum number of results to return.

        Returns:
            A list of dictionaries, each containing 'id', 'text', 'metadata', and 'score'.

        Raises:
            ValueError: If collection name is empty, top_k <= 0, or query embedding is invalid.
            RuntimeError: If the Chroma response structure is malformed.
        """
        clean_name = collection_name.strip()
        if not clean_name:
            raise ValueError("Collection name must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if not query_embedding or not all(isinstance(x, (int, float)) for x in query_embedding):
            raise ValueError("Query embedding must be non-empty and contain only numeric values.")

        normalized_query = [float(x) for x in query_embedding]
        collection = self.get_or_create_collection(clean_name)

        response = collection.query(
            query_embeddings=[normalized_query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns a list of lists for each field. 
        # We need to flatten the first result since we only passed one query embedding.
        if not isinstance(response["documents"], list) or not response["documents"]:
            return []

        results = []
        for i in range(len(response["documents"])):
            # Validate that metadata and distances exist at this index
            if (i < len(response["metadatas"]) and 
                i < len(response["distances"]) and 
                response["documents"][i] is not None):
                
                results.append({
                    "id": response["ids"][i],
                    "text": response["documents"][i],
                    "metadata": dict(response["metadatas"][i]),
                    "score": response["distances"][i],
                })
        
        return results

    def delete_collection(self, collection_name: str) -> None:
        """
        Deletes the specified collection from Chroma.

        Args:
            collection_name: The name of the collection to delete.

        Raises:
            ValueError: If the collection name is empty or whitespace-only.
        """
        clean_name = collection_name.strip()
        if not clean_name:
            raise ValueError("Collection name must not be empty.")

        self.client.delete_collection(name=clean_name)