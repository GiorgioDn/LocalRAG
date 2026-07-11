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
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RagAnswer:
    """Represents the final answer returned by the RAG pipeline, including source metadata."""
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LoadedDocument:
    """Extracted document content before chunking."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)