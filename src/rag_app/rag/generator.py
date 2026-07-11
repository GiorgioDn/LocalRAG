from typing import Any, Sequence
from dataclasses import dataclass

from src.rag_app.models.ollama_chat import OllamaChatClient
from src.rag_app.types import RagAnswer, RetrievedChunk
from src.rag_app.config import settings


@dataclass
class RagGenerator:
    """
    Handles the generation of responses by interacting with a chat client 
    using a provided prompt and context.
    """

    chat_client: OllamaChatClient

    def __init__(
        self,
        chat_client: OllamaChatClient | None = None,
    ) -> None:
        """
        Initialize the generator with an optional chat client.
        
        Args:
            chat_client: An instance of OllamaChatClient. If None, 
                         a default client is initialized.
        """
        self.chat_client = chat_client or OllamaChatClient()

    def generate(
        self,
        prompt: str,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> RagAnswer:
        """
        Generate a response based on the provided prompt and retrieved chunks.

        Args:
            prompt: The constructed RAG prompt to send to the model.
            retrieved_chunks: A sequence of chunks retrieved from the vector store.

        Returns:
            RagAnswer: An object containing the LLM's response and a list 
                       of unique source metadata dictionaries.

        Raises:
            ValueError: If the prompt is empty or contains only whitespace.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty or whitespace-only.")

        # Call the chat adapter exactly once
        answer_text = self.chat_client.answer(prompt)

        # Construct unique sources from retrieved chunks
        sources = self._build_unique_sources(retrieved_chunks)

        return RagAnswer(answer=answer_text, sources=sources)

    def _build_unique_sources(self, chunks: Sequence[RetrievedChunk]) -> list[dict[str, Any]]:
        """
        Deduplicate source information from the provided chunks.

        A source is considered unique based on the combination of its 
        identifier and page number (if applicable).
        """
        seen_identifiers = set()
        unique_sources: list[dict[str, Any]] = []

        for chunk in chunks:
            # Use a tuple of (source_id, page) as a unique key.
            # For text files where page is None, the key will be (source_id, None).
            source_id = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            
            identifier = (source_id, page)

            if identifier not in seen_identifiers:
                seen_identifiers.add(identifier)
                # Append a copy of the metadata dictionary to ensure 
                # results are independent of original objects.
                unique_sources.append(dict(chunk.metadata))

        return unique_sources