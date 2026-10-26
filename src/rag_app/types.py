from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """Represents a normalized chunk ready to be embedded and stored."""
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """Represents one relevant result returned by retrieval code. It contains text, source metadata, and a score."""
    text: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        text: str = "",
        score: float = 0.0,
        metadata: dict[str, Any] | None = None,
        *,
        content: str | None = None,
    ) -> None:
        self.text = content if content is not None else text
        self.score = score
        self.metadata = metadata if metadata is not None else {}

    @property
    def content(self) -> str:
        return self.text

    @content.setter
    def content(self, value: str) -> None:
        self.text = value


@dataclass
class RagAnswer:
    """Represents the final answer returned by the RAG pipeline, including source metadata."""
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class LoadedDocument:
    """Extracted document content before chunking."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)