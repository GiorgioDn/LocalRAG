from pathlib import Path
from typing import Union

from src.rag_app.constants import METADATA_SOURCE, METADATA_SOURCE_TYPE, SOURCE_TYPE_TEXT, SUPPORTED_TEXT_EXTENSIONS
from src.rag_app.ingestion.normalizer import normalize_text
from src.rag_app.types import LoadedDocument
from src.rag_app.utils.file_utils import require_existing_file, require_supported_extension

def load_text_file(path: Union[str, Path]) -> list[LoadedDocument]:
    """
    Safely loads a text-based file, normalizes its content, and returns 
    a list containing one LoadedDocument.
    """
    # Use new utilities for path validation
    file_path = require_existing_file(path)
    file_path = require_supported_extension(file_path, SUPPORTED_TEXT_EXTENSIONS)

    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Failed to decode file {file_path} using UTF-8 encoding.") from exc

    normalized_text = normalize_text(raw_text)

    if not normalized_text.strip():
        raise ValueError(f"The content of {file_path} is empty or only whitespace after normalization.")

    return [
        LoadedDocument(
            text=normalized_text,
            metadata={
                METADATA_SOURCE: str(file_path),
                METADATA_SOURCE_TYPE: SOURCE_TYPE_TEXT,
            },
        )
    ]