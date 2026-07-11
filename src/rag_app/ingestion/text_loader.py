from pathlib import Path
from typing import Union

from rag_app.constants import METADATA_SOURCE, METADATA_SOURCE_TYPE, SOURCE_TYPE_TEXT, SUPPORTED_TEXT_EXTENSIONS
from rag_app.ingestion.normalizer import normalize_text
from rag_app.types import LoadedDocument

def load_text_file(path: Union[str, Path]) -> list[LoadedDocument]:
    """
    Safely loads a text-based file, normalizes its content, and returns 
    a list containing one LoadedDocument.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"The specified path does not exist: {file_path}")
    
    if file_path.is_dir():
        raise IsADirectoryError(f"The path is a directory, not a file: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_TEXT_EXTENSIONS:
        supported = ", ".join(sorted(list(SUPPORTED_TEXT_EXTENSIONS)))
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            f"Supported extensions are: {supported}"
        )

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