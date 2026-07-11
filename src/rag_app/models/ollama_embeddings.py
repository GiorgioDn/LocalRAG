from typing import List
import ollama
from src.rag_app.config import settings

class OllamaEmbeddingClient:
    """
    A stateless adapter for generating embeddings using the Ollama service.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        client: ollama.Client | None = None,
    ) -> None:
        """
        Initialize the client with optional overrides for model and base URL.

        Args:
            model: The Ollama model identifier to use for embeddings. Defaults to settings.ollama_embed_model.
            base_url: The base URL of the Ollama service. Defaults to settings.ollama_base_url.
            client: An optional pre-configured ollama.Client instance.
        """
        self.model = model or settings.ollama_embed_model
        self.base_url = base_url or settings.ollama_base_url
        
        if client is None:
            # Initialize the client with the resolved base_url if not provided via injection
            self.client = ollama.Client(host=self.base_url)
        else:
            self.client = client

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of strings.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of lists of floats representing the embeddings.

        Raises:
            ValueError: If any input string is empty or only contains whitespace.
            RuntimeError: If the Ollama API returns an unexpected number/format of results.
        """
        if not texts:
            return []

        for text in texts:
            if not text or text.isspace():
                raise ValueError("Each input text must be a non-empty string.")

        try:
            response = self.client.embeddings(
                model=self.model,
                prompt=texts
            )
            embeddings = self._extract_embeddings(response)
            
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Expected {len(texts)} embeddings but received {len(embeddings)}."
                )
            
            return embeddings

        except Exception as e:
            raise RuntimeError(f"Embedding failed for model '{self.model}'.") from e

    def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a single query string.

        Args:
            query: The query string to embed.

        Returns:
            A list of floats representing the embedding.

        Raises:
            ValueError: If the query is empty or only contains whitespace.
        """
        if not query or query.isspace():
            raise ValueError("Query must be a non-empty string.")

        # Re-use embed_texts to ensure consistent logic and avoid redundant code.
        embeddings = self.embed_texts([query])
        return embeddings[0]

    def _extract_embeddings(self, response: object) -> List[List[float]]:
        """
        Helper method to extract embedding data from the Ollama response.
        Ensures the result is a list of lists of floats regardless of SDK-specific types.

        Args:
            response: The raw response from the ollama.Client.embeddings call.

        Returns:
            A nested list of floats.
        """
        # Ensure we handle the dictionary-like access typical in recent 'ollama' python libs
        # or list-of-objects depending on specific internal implementation.
        # Typically, response.embeddings is a list of objects where .embedding is a list of floats.
        if hasattr(response, "embeddings") and response.embeddings:
            return [[e.embedding for e in response.embeddings] if hasattr(e, 'embedding') 
                    else [e] for e in response.embeddings]
        
        # Fallback/Alternative if the object is a dict or direct list of lists
        try:
            if isinstance(response, dict) and "embeddings" in response:
                return [[float(x) for x in emb] for emb in response["embeddings"]]
            return [list(map(float, emb)) for emb in response.embeddings]
        except (AttributeError, TypeError):
            # If we can't determine the type easily but it is iterable
            return [list(map(float, emb)) for emb in response.embeddings]
