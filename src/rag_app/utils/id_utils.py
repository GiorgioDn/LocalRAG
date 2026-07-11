from collections import Counter
import re
from pathlib import Path
from typing import Any, Mapping

from rag_app.constants import METADATA_CHUNK_INDEX
from rag_app.constants import METADATA_PAGE
from rag_app.constants import METADATA_SOURCE

def build_chunk_id(
    metadata: Mapping[str, Any],
    chunk_index: int,
) -> str:
    """
    Builds a deterministic, database-safe ID for a document chunk.

    Args:
        metadata: A mapping containing metadata about the source document.
        chunk_index: The zero-based index of the chunk within the document.

    Returns:
        A string representing the unique identifier for the chunk.

    Raises:
        TypeError: If metadata is not a Mapping or if chunk_index is not an integer.
        ValueError: If required fields are missing or values are invalid.
    """
    # 1. Validate inputs
    if not isinstance(metadata, Mapping):
        raise TypeError(f"Metadata must be a mapping, got {type(metadata).__name__}")

    if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
        raise TypeError(f"Chunk index must be an integer (not boolean), got {type(chunk_index).__name__}")

    if chunk_index < 0:
        raise ValueError(f"Chunk index must be non-negative, got {chunk_index}")

    # 2. Get and validate source
    raw_source = metadata.get(METADATA_SOURCE)
    if not isinstance(raw_source, str) or not raw_source.strip():
        raise ValueError(f"Metadata '{METADATA_SOURCE}' must be a non-empty string.")

    # 3. Build source component
    # Use Path().name to get the filename and discard path info
    path_obj = Path(raw_source)
    filename = path_obj.name if path_obj.name else raw_source
    
    # Sanitize: keep lowercase ASCII letters, digits, '.', '_', '-'
    # Replace anything else with '-'
    sanitized = "".join(
        c if (c.isalnum() or c in "._-") else "-" 
        for c in filename.lower()
    )
    
    # Collapse repeated hyphens and strip leading/trailing ones
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    
    if not sanitized:
        safe_source = "document"
    else:
        safe_source = sanitized

    # 4. Build page component
    page_val = metadata.get(METADATA_PAGE)
    
    # Logic for page: must be an integer > 0 if it exists as a non-None value.
    if page_val is None:
        page_component = "pnone"
    elif isinstance(page_val, bool) or not isinstance(page_val, int):
        raise ValueError(f"Metadata '{METADATA_PAGE}' must be an integer (not boolean), got {type(page_val).__name__}")
    elif page_val <= 0:
        # Reject booleans, zero, negative values, non-integers, and blank strings with clear errors.
        raise ValueError(f"Page number must be greater than zero, got {page_val}")
    else:
        page_component = f"p{page_val}"

    # 5. Return final string
    return f"{safe_source}-{page_component}-c{chunk_index}"