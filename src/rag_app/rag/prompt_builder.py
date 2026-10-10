from typing import Any, Mapping, Sequence

from src.rag_app.types import RetrievedChunk

def format_source(metadata: Mapping[str, Any]) -> str:
    """
    Formats source information from metadata into a string for the prompt.

    Args:
        metadata: A mapping containing 'source', 'page', and 'source_type'.

    Returns:
        A formatted string like "source: manual.pdf | page: 12" or just "source: manual.pdf".
    """
    parts = []
    if "source" in metadata:
        source = metadata["source"]
        parts.append(f"source: {source}" if source else "source: unknown source")
    else:
        parts.append("source: unknown source")

    if "page" in metadata and metadata["page"] is not None:
        parts.append(f"page: {metadata['page']}")

    if "source_type" in metadata and metadata["source_type"]:
        parts.append(f"type: {metadata['source_type']}")

    return " | ".join(parts)


def build_rag_prompt(
    question: str,
    retrieved_chunks: Sequence[RetrievedChunk],
) -> str:
    """
    Builds a grounded RAG prompt from a question and retrieved chunks.

    Args:
        question: The user's query.
        retrieved_chunks: A sequence of RetreivedChunk objects.

    Returns:
        A plain string containing the full prompt for the LLM.

    Raises:
        ValueError: If the question is empty or only whitespace.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    prompt_parts = [
        "You are a grounded assistant. Answer the question using only the context below.",
        "If the context does not contain enough information, say so clearly.",
        "Do not invent facts or sources.",
        "",
        "Context:",
    ]

    if not retrieved_chunks:
        prompt_parts.append("No relevant context was found.")
    else:
        for i, chunk in enumerate(retrieved_chunks, start=1):
            source_info = format_source(chunk.metadata)
            prompt_parts.append(f"[Context {i} | {source_info}]")
            prompt_parts.append(f"{chunk.text}")
            prompt_parts.append("")

    prompt_parts.append(f"Question:\n{question.strip()}")
    prompt_parts.append("")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)