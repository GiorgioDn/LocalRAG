"""
This module provides tools for loading and processing PDF files within 
the ingestion pipeline. It converts raw PDF pages into a list of 
Normalized documents, ensuring that page-level metadata is preserved.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Union, List

from src.rag_app.ingestion.normalizer import normalize_text
from src.rag_app.constants import (
    SUPPORTED_PDF_EXTENSIONS,
    SOURCE_TYPE_PDF,
    METADATA_SOURCE,
    METADATA_SOURCE_TYPE,
    METADATA_PAGE,
)
from src.rag_app.types import LoadedDocument
from src.rag_app.utils.file_utils import require_existing_file, require_supported_extension

def load_pdf_file(path: Union[str, Path]) -> List[LoadedDocument]:
    """
    Opens a PDF file and extracts text from each page.

    Each page is normalized and wrapped in a LoadedDocument object 
    containing the original source path, type, and page number.

    Args:
        path (str | Path): The filesystem path to the PDF file.

    Returns:
        List[LoadedDocument]: A list of documents, one for each 
            page containing extractable text.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        IsADirectoryError: If the provided path is a directory.
        ValueError: If the file extension is not supported, if the 
                    PDF is corrupted/encrypted, or if the PDF contains 
                    no extractable text.
    """
    # Use new utilities for validation
    path_obj = require_existing_file(path)
    path_obj = require_supported_extension(path_obj, SUPPORTED_PDF_EXTENSIONS)

    documents: List[LoadedDocument] = []

    # Try/finally ensures the PyMuPDF document is closed regardless of errors
    doc = None
    try:
        doc = fitz.open(path_obj)
    except Exception as e:
        raise ValueError(f"Failed to open PDF at {path_obj}. The file may be "
                         "corrupted, encrypted, or otherwise unreadable.") from e

    try:
        for page_num, page in enumerate(doc, start=1):
            # Extract raw text
            raw_text = page.get_text("text")
            
            # Normalize the text using shared logic
            normalized_text = normalize_text(raw_text)

            if normalized_text:
                documents.append(LoadedDocument(
                    text=normalized_text,
                    metadata={
                        METADATA_SOURCE: str(path_obj.absolute()),
                        METADATA_SOURCE_TYPE: SOURCE_TYPE_PDF,
                        METADATA_PAGE: page_num,
                    }
                ))
    finally:
        if doc:
            doc.close()

    # If no text was extracted from any page, the PDF might be scanned/image-based
    if not documents:
        raise ValueError(
            f"The PDF at {path_obj} contains no extractable text. "
            "It may be a scanned document or image-based."
        )

    return documents