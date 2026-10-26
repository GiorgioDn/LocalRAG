import ollama
from typing import Optional
from src.rag_app.config import settings

class OllamaChatClient:
    """Adapter for interacting with the Ollama chat API."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[ollama.Client] = None,
    ) -> None:
        """
        Initialize the client with model details and an optional pre-configured Ollama Client.

        Args:
            model: The name of the model to use. Defaults to settings.ollama_chat_model.
            base_url: The base URL for the Ollama service. Defaults to settings.ollama_base_url.
            client: A pre-configured ollama.Client instance for dependency injection.
        """
        self.model = model or settings.ollama_chat_model
        # Note: some clients might use 'host' or 'base_url', but the 
        # standard way to pass a base URL via construction if not using 
        # default env vars is often handled by the client initialization.
        self._base_url = base_url or settings.ollama_base_url
        
        if client is None:
            self.client = ollama.Client(host=self._base_url)
        else:
            self.client = client

    def answer(self, prompt: str) -> str:
        """
        Send a single prompt to the Ollama chat model and return the generated text.

        Args:
            prompt: The user input string.

        Returns:
            The stripped response from the model.

        Raises:
            ValueError: If the prompt is empty or contains only whitespace.
            RuntimeError: If an error occurs during communication with Ollama 
                         or if the response is invalid.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace-only.")

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            return self._extract_answer(response)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve response from Ollama: {e}") from e

    def _extract_answer(self, response: ollama.ChatResponse) -> str:
        """Extract and format the assistant's message from the raw Ollama response."""
        message = getattr(response, "message", None)
        content = getattr(message, "content", None) if message else None
        if not message or content is None:
            raise RuntimeError("Ollama returned an empty response.")
        
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty response.")
        
        return content.strip()