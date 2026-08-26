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

        embeddings = []
        for text in texts:
            try:
                response = self.client.embeddings(
                    model=self.model,
                    prompt=text
                )
                embeddings.append(self._extract_embeddings(response))
            except Exception as e:
                raise RuntimeError(f"Embedding failed for model '{self.model}'.") from e

        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Expected {len(texts)} embeddings but received {len(embeddings)}."
            )
        
        return embeddings

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

    def _extract_embeddings(self, response: object) -> List[float]:
        """
        Helper method to extract embedding data from the Ollama response.
        Ensures the result is a list of floats.

        Args:
            response: The raw response from the Ollama.Client.embeddings call.

        Returns:
            A list of floats representing the embedding.
        """
        # Handle the different ways the 'ollama' library might return data
        if hasattr(response, "embeddings") and response.embeddings:
            # If multiple were returned, take the first (relevant if iterating)
            return list(map(float, response.embeddings[0]))
        
        if hasattr(response, "embedding") and response.embedding:
            # Single embedding result (expected when prompt is a single string)
            return list(map(float, response.embedding))

        # Fallback for dictionary-like access
        if isinstance(response, dict) and "embeddings" in response:
            return list(map(float, response["embeddings"]))
        
        # Final fallback to direct cast
        try:
            return list(map(float, response))
        except (TypeError, ValueError):
            # If it fails, attempt to fall back to known properties
            if hasattr(response, "embeddings") and response.embeddings:
                return list(map(float, response.embeddings[0]))
            if hasattr(response, "embedding") and response.embedding:
                return list(map(float, response.embedding))
            return []
